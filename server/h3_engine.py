"""Resident ComfyUI runtime: model lifecycle, caches, cancellation and pipeline dispatch.

All generation runs on the owning worker thread. Each pipeline lives in pipelines/;
model_catalog contains only configuration and standard-library helpers.
"""

import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "comfy_launcher"))
from run_h3 import bootstrap_comfy, load_larryvrh_nodes
from engine_errors import Canceled
from runtime_memory import MemoryPolicy
from audio_runtime import AudioCppPool
from audio_catalog import native_id as audio_native_id
from pipelines import audio_cpp, llm
from llm_runtime import StrataServer
from model_catalog import AUDIO_VAE, CLIP, DEFAULT_LORAS, LORAS, VIDEO_VAE, unet_file
from pipelines import video, picture, music, seedvr2, flashvsr, relayvsr, pixal, media


class H3Engine:
    def __init__(self, comfyui_path, comfy_argv, *, audio_settings=None):
        self.comfyui_path = comfyui_path
        self.comfy_argv = comfy_argv
        self.memory = MemoryPolicy(self)
        self._models = {}
        self.family = None
        self.unet = None
        self.base = None
        self._moss = None
        self._picture_model = self._picture_models = None
        self._music = None
        self._seedvr2 = None
        self._flashvsr = None
        self._relayvsr = None
        self._pixal = None
        self._cancel = False
        self.audio_cpp = AudioCppPool(audio_settings.audio_root, audio_settings.data / "audio_cpp",
            port=audio_settings.audio_cpp_port, threads=audio_settings.audio_cpp_threads,
            timeout=audio_settings.audio_cpp_timeout,
            instances=audio_settings.audio_cpp_instances) if audio_settings is not None else None
        self.llm = StrataServer(audio_settings.strata_dir, audio_settings.strata_config, audio_settings.data / "strata",
            port=audio_settings.strata_port, parallel=audio_settings.strata_parallel,
            timeout=audio_settings.strata_timeout) if audio_settings is not None and audio_settings.strata_dir and audio_settings.strata_config else None

    def bootstrap(self):
        """ComfyUI's process-level setup (CLI args, DynamicVRAM). Call from the main thread."""
        bootstrap_comfy(self.comfyui_path, self.comfy_argv)

    def load(self):
        """Create the model objects. Call from the thread that will run generate()."""
        import nodes
        import comfy.model_management
        import comfy.utils
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from comfy_extras.nodes_custom_sampler import (
            BasicGuider,
            BasicScheduler,
            KSamplerSelect,
            RandomNoise,
            SamplerCustomAdvanced,
        )
        from comfy_extras.nodes_minimax_h3 import (
            MiniMaxH3AddGuide,
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
            MiniMaxH3SigmaShift,
        )
        from comfy_extras.nodes_video import CreateVideo
        from comfy_api.latest import Types

        self.nodes = nodes
        self.mm = comfy.model_management
        self.utils = comfy.utils
        self.n = {
            "VAEDecodeAudio": VAEDecodeAudio,
            "BasicGuider": BasicGuider,
            "BasicScheduler": BasicScheduler,
            "RandomNoise": RandomNoise,
            "SamplerCustomAdvanced": SamplerCustomAdvanced,
            "ImageToVideo": MiniMaxH3ImageToVideo,
            "ReferenceToVideo": MiniMaxH3ReferenceToVideo,
            "AddGuide": MiniMaxH3AddGuide,
            "SigmaShift": MiniMaxH3SigmaShift,
            "CreateVideo": CreateVideo,
            "KSamplerSelect": KSamplerSelect,
        }
        self.Types = Types

        self.clip = nodes.CLIPLoader().load_clip(CLIP, "minimax", "default")[0]
        self.vae = nodes.VAELoader().load_vae(VIDEO_VAE)[0]
        self.audio_vae = nodes.VAELoader().load_vae(AUDIO_VAE)[0]
        self._samplers = {}
        self._turbo = None
        self._use_base(DEFAULT_LORAS["fl2va"]["544p"])

    def _prepare_task(self, task):
        self._cancel = False
        self.mm.interrupt_current_processing(False)
        self.memory.prepare(task)
        # A cancel received during a slow offload must survive pipeline setup.
        self._check()

    def _sampler(self, name):
        if name not in self._samplers:
            self._samplers[name] = self.n["KSamplerSelect"].execute(name)[0]
        return self._samplers[name]

    def _use_base(self, lora):
        """Make the base DiT of model choice `lora` the resident one, unloading the previous base first."""
        unet = unet_file(lora)
        if self.unet == unet:
            return False
        if self.base is not None:
            # Drop every reference to the old DiT (its clones share its weights)
            # so ComfyUI can release the VRAM and the process can release the RAM.
            self.mm.unload_model_and_clones(self.base)
            self._models.clear()
            self.base = None
            self.unet = None
            self.family = None
            gc.collect()
            self.mm.soft_empty_cache()
        self.base = self.nodes.UNETLoader().load_unet(unet, "default")[0]
        self.unet = unet
        self.family = LORAS[lora]["family"]
        for key in DEFAULT_LORAS[self.family].values():
            if unet_file(key) == unet:
                self._lora_model(key)
        return True

    def unload(self):
        """Free the VRAM for another program: drop the base DiT, its clones and the lazily loaded models, and move
        everything else off the GPU. The next job reloads what it needs. Call from the worker thread when idle."""
        import torch

        if self.audio_cpp is not None:
            self.audio_cpp.stop()
        if self.llm is not None:
            self.llm.stop()
        self.memory.reset()
        if self._flashvsr is not None:
            self._flashvsr.release()
        self._models.clear()
        self.base = self.unet = self.family = None
        self._moss = self._music = self._seedvr2 = self._flashvsr = self._relayvsr = self._pixal = None
        self._picture_model = self._picture_models = None
        self.mm.unload_all_models()
        gc.collect()
        self.mm.soft_empty_cache()
        free, total = torch.cuda.mem_get_info()
        return {"vram_free_mb": free // 2**20, "vram_total_mb": total // 2**20}

    def _lora_model(self, key, strength=1.0):
        """(model, sigmas, sampler) for a model choice at a LoRA strength on the resident DiT."""
        import torch

        if (key, strength) not in self._models:
            # Keep the full-strength clones plus only the latest other strength: each holds its LoRA weights.
            for cached in [c for c in self._models if c[1] != 1.0]:
                del self._models[cached]
            p = LORAS[key]
            model = self.n["SigmaShift"].execute(
                self.base, p["shift_video"], p["shift_audio"]
            )[0]
            if p["loader"] == "larryvrh":
                turbo = self._larryvrh_nodes()
                model = turbo.MiniMaxH3TurboLoRA().apply_lora(
                    model, p["file"], strength
                )[0]
                sampler = turbo.MiniMaxH3TurboSampler().get_sampler()[0]
            else:
                if p["file"]:
                    model = self.nodes.LoraLoaderModelOnly().load_lora_model_only(
                        model, p["file"], strength
                    )[0]
                sampler = self._sampler(p.get("sampler", "euler"))
            if p.get("vsa") or p.get("sol"):
                from comfy_extras.nodes_model_advanced import ModelAttentionBackend
                from comfy_extras.nodes_sparse_attention import BlockSparseAttention

                model = ModelAttentionBackend.execute(
                    model=model, attention="comfy kitchen attention"
                )[0]
                if p.get("vsa"):
                    selection = {"selection": "vsa", "keep_percent": p["vsa"]["keep"]}
                    start = p["vsa"]["start"]
                else:  # Sol-Attn: training-free, for bases without VSA's learned coarse branch
                    selection = {"selection": "sol-attn", "tau": p["sol"]["tau"]}
                    start = p["sol"]["start"]
                model = BlockSparseAttention.execute(
                    model=model,
                    selection=selection,
                    start_percent=start,
                    end_percent=1.0,
                    dense_blocks="",
                    min_tokens=12288,
                    extra_tokens=256,
                    sink_conditioning="exact_kv_and_rows",
                )[0]
            if p.get("sigmas"):
                # Already shifted for the video stream; the DiT derives the audio sigmas from them.
                sigmas = torch.tensor(p["sigmas"], dtype=torch.float32)
            else:
                sigmas = self.n["BasicScheduler"].execute(
                    model, "simple", p["steps"], 1.0
                )[0]
            self._models[(key, strength)] = (model, sigmas, sampler)
        return self._models[(key, strength)]

    def _larryvrh_nodes(self):
        if self._turbo is None:
            self._turbo = load_larryvrh_nodes(self.comfyui_path)
        return self._turbo

    def cancel(self):
        self._cancel = True
        if self.audio_cpp is not None and self.memory.task and self.memory.task[0] == "audio_cpp":
            self.audio_cpp.request_stop()
        if self.llm is not None and self.memory.task and self.memory.task[0] == "llm":
            self.llm.request_stop()
        self.mm.interrupt_current_processing(True)

    def _check(self):
        if self._cancel:
            raise Canceled()

    def _after_node(self):
        """What ComfyUI's executor runs after every node under DynamicVRAM. Autoregressive text encoders
        (MiniMax Music 3) record a per-thread CUDA malloc graph; without this the next job replays a stale one
        and fails with a device-side assert."""
        import comfy.memory_management
        import comfy.model_prefetch

        if comfy.memory_management.aimdo_enabled:
            import comfy_aimdo.model_vbar

            comfy.model_prefetch.cleanup_prefetch_queues()
            self.mm.reset_cast_buffers()
            comfy_aimdo.model_vbar.vbars_reset_watermark_limits()

    def generate(
        self,
        *,
        lora,
        prompt,
        width,
        height,
        duration,
        seed,
        lora_strength=1.0,
        sampler=None,
        steps=None,
        image_path=None,
        last_image_path=None,
        ref_paths=(),
        ref_video_paths=(),
        audio_path=None,
        audio_start=0.0,
        video_path,
        thumb_path,
        on_progress,
    ):
        self._prepare_task(("video", unet_file(lora)))
        return video.generate(
            self,
            lora=lora,
            prompt=prompt,
            width=width,
            height=height,
            duration=duration,
            seed=seed,
            lora_strength=lora_strength,
            sampler=sampler,
            steps=steps,
            image_path=image_path,
            last_image_path=last_image_path,
            ref_paths=ref_paths,
            ref_video_paths=ref_video_paths,
            audio_path=audio_path,
            audio_start=audio_start,
            video_path=video_path,
            thumb_path=thumb_path,
            on_progress=on_progress,
        )

    def generate_image(
        self,
        *,
        prompt,
        width,
        height,
        seed,
        image_path,
        thumb_path,
        on_progress,
        image_model="moss",
        ref_paths=(),
    ):
        self._prepare_task(("image", image_model))
        return picture.generate_image(
            self,
            image_model=image_model,
            ref_paths=ref_paths,
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            image_path=image_path,
            thumb_path=thumb_path,
            on_progress=on_progress,
        )

    def generate_music(
        self, *, caption, lyrics, duration, seed, audio_path, on_progress
    ):
        self._prepare_task(("music",))
        return music.generate_music(
            self,
            caption=caption,
            lyrics=lyrics,
            duration=duration,
            seed=seed,
            audio_path=audio_path,
            on_progress=on_progress,
        )

    def upscale_video(
        self, *, source_path, width, height, seed, video_path, thumb_path, on_progress
    ):
        self._prepare_task(("seedvr2",))
        return seedvr2.upscale_video(
            self,
            source_path=source_path,
            width=width,
            height=height,
            seed=seed,
            video_path=video_path,
            thumb_path=thumb_path,
            on_progress=on_progress,
        )

    def flashvsr_video(
        self, *, source_path, width, height, seed, video_path, thumb_path, on_progress
    ):
        self._prepare_task(("flashvsr",))
        return flashvsr.flashvsr_video(
            self,
            source_path=source_path,
            width=width,
            height=height,
            seed=seed,
            video_path=video_path,
            thumb_path=thumb_path,
            on_progress=on_progress,
        )

    def relayvsr_video(
        self, *, source_path, target, seed, video_path, thumb_path, on_progress
    ):
        self._prepare_task(("relayvsr", target))
        return relayvsr.relayvsr_video(
            self,
            source_path=source_path,
            target=target,
            seed=seed,
            video_path=video_path,
            thumb_path=thumb_path,
            on_progress=on_progress,
        )

    def _ref_video(self, path, width, height, frames):
        return media._ref_video(path=path, width=width, height=height, frames=frames)

    def _audio_clip(self, path, start, duration):
        return media._audio_clip(path=path, start=start, duration=duration)

    def _thumb_from_video(self, video_path, thumb_path):
        return media._thumb_from_video(video_path=video_path, thumb_path=thumb_path)

    def _save_mp3(self, audio, path):
        return media._save_mp3(audio=audio, path=path)

    def _save_picture(self, frame, path):
        return media._save_picture(frame=frame, path=path)

    def _save_thumb(self, frame, path):
        return media._save_thumb(frame=frame, path=path)

    def generate_3d(self, **kwargs):
        self._prepare_task(("pixal3d",))
        return pixal.generate_3d(self, **kwargs)

    def speech_task(self, model, options):
        return ("audio_cpp", audio_native_id(model, options["speech_mode"]))

    def generate_speech(self, *, model, options, prepared=False, **kwargs):
        """prepared=True: the worker already ran _prepare_task for this speech batch."""
        if not prepared:
            self._prepare_task(self.speech_task(model, options))
        return audio_cpp.speech(self, model=model, options=options, **kwargs)

    def transcribe(self, *, model, **kwargs):
        self._prepare_task(("audio_cpp", model))
        return audio_cpp.transcribe(self, model=model, **kwargs)

    def run_llm(self, **kwargs):
        self._prepare_task(("llm", self.llm.model if self.llm else None))
        return llm.batch(self, **kwargs)
