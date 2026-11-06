"""RelayVSR pipeline: one-step Wan on every 15th frame, Flash decoder fills in the rest."""

import time
from model_catalog import RELAYVSR, RELAYVSR_MODELS, upscale_model_info
from engine_errors import Canceled


def _relayvsr_models(engine, target):
    """(model, vae) of one scale factor, loaded on first use and kept resident."""
    if engine._relayvsr is None:
        engine._relayvsr = {}
    if target not in engine._relayvsr:
        files = RELAYVSR_MODELS[target]
        model = engine.nodes.UNETLoader().load_unet(files["unet"], "default")[0]
        vae = engine.nodes.VAELoader().load_vae(files["vae"])[0]
        engine._relayvsr[target] = (model, vae)
    return engine._relayvsr[target]


def relayvsr_video(
    engine, *, source_path, target, seed, video_path, thumb_path, on_progress
):
    """Upscale a finished video by the checkpoint's native factor, keeping its audio and frame rate."""
    import torch
    from comfy_api.latest import InputImpl
    from comfy_extras.nodes_wan import RelayVSRConditioning, VSRDecode

    engine._cancel = False
    engine.mm.interrupt_current_processing(False)
    timings = {}
    stage_name = {"value": None}

    def stage(name):
        stage_name["value"] = name
        on_progress(name, 0.0)
        timings[name] = time.perf_counter()

    def hook(value, total, preview, node_id=None):
        engine._check()
        if stage_name["value"] in ("upscaling", "decoding") and total:
            on_progress(stage_name["value"], value / total)

    engine.utils.set_progress_bar_global_hook(hook)
    try:
        with torch.inference_mode():
            if target not in (engine._relayvsr or {}):
                stage("loading")
            model, vae = _relayvsr_models(engine, target)
            stage("reading")
            source = InputImpl.VideoFromFile(str(source_path)).get_components()
            positive, latent = RelayVSRConditioning.execute(
                model=model, images=source.images
            )
            engine._check()

            stage("upscaling")
            # The workflow wires the conditioning node's one output to both inputs; cfg 1 ignores the negative.
            sampled = engine.nodes.KSampler().sample(
                model,
                seed,
                RELAYVSR["steps"],
                RELAYVSR["cfg"],
                RELAYVSR["sampler"],
                RELAYVSR["scheduler"],
                positive,
                positive,
                latent,
                denoise=1.0,
            )[0]
            engine._after_node()  # The model records a per-thread malloc graph while it samples.
            engine._check()

            stage("decoding")
            images = VSRDecode.execute(
                vae=vae,
                samples=sampled,
                images=source.images,
                detail=RELAYVSR["detail"],
            )[0]
            engine._check()

            stage("saving")
            video = engine.n["CreateVideo"].execute(
                images, float(source.frame_rate), source.audio
            )[0]
            video.save_to(
                str(video_path),
                format=engine.Types.VideoContainer("mp4"),
                codec=engine.Types.VideoCodec("auto"),
            )
            engine._save_thumb(images[images.shape[0] // 2], thumb_path)
    except engine.mm.InterruptProcessingException:
        raise Canceled()
    finally:
        engine.utils.set_progress_bar_global_hook(None)
        engine.mm.interrupt_current_processing(False)
        engine._after_node()

    end = time.perf_counter()
    names = list(timings)
    durations = {
        n: round((timings[names[i + 1]] if i + 1 < len(names) else end) - timings[n], 1)
        for i, n in enumerate(names)
    }
    return {
        "frames": int(images.shape[0]),
        "steps": RELAYVSR["steps"],
        "timings": durations,
        "models": upscale_model_info("relayvsr", target),
    }
