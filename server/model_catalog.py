"""Model registry, inference defaults and size/prompt helpers. Standard library only."""

import math
from image_models import (IMAGE, IMAGE_AREA, MOSS_REPO, MOSS_UNET, MOSS_CLIP, MOSS_VAE, image_model_info)  # noqa: F401

# Pixal3D native single-image pipeline (official Comfy-Org template).
PIXAL_REPO = "Comfy-Org/Pixal3D"
PIXAL_FILES = [
    (PIXAL_REPO, "diffusion_models", "pixal3d_int8_convrot.safetensors"),
    (PIXAL_REPO, "clip_vision", "dino_v3_L_naf_fp32.safetensors"),
    (PIXAL_REPO, "vae", "trellis_2_shape_vae_bf16.safetensors"),
    (PIXAL_REPO, "vae", "trellis_2_texture_vae_bf16.safetensors"),
    ("Comfy-Org/MoGe", "geometry_estimation", "moge_2_vitl_normal_fp16.safetensors"),
    ("Comfy-Org/BiRefNet", "background_removal", "birefnet.safetensors"),
]


def pixal_model_info():
    return {
        "unet": PIXAL_FILES[0][2],
        "sampler": "euler",
        "steps": [12, 20, 12, 12],
        "shape_vae": PIXAL_FILES[2][2],
        "texture_vae": PIXAL_FILES[3][2],
    }


FPS = 24

RESOLUTIONS = {"544p": 960 * 544, "768p": 1344 * 768}

# Model groups shown to users: H3 bases take a LoRA, FastH3 bases are distilled and take none.
GROUPS = {
    "h3": "MiniMax-H3 + LoRA",
    "fasth3": "FastVideo FastH3",
    "pika": "Pika H3 Flex",
}

# Model choices. H3 entries are a LoRA on the family's base in UNETS; FastH3 entries
# name their own `unet` and have no LoRA (`file` None). `resolutions` lists where each
# one was trained or tested; `source` / `unet_source` are the Hugging Face repos
# scripts/prepare_models.py downloads from. The shift has to match the checkpoint.
# `vsa` turns on comfy_kitchen attention plus Video Sparse Attention as in the FastH3
# ComfyUI templates. `fixed` keeps the template's sampler and steps whatever a job asks: FastH3 V2 is
# distilled on a fixed 8-rung ladder and OmniRef fuses 4 of its 32 PDD output heads per Euler step,
# so another sampler or step count leaves what they were trained for. The 4-step lightx2v ref2v LoRAs were removed: they
# smeared large motion into streaks even at 6-8 steps. `sigmas` replaces the simple schedule with fixed (already shifted)
# values, as Pika H3 Flex needs; `sol` turns on training-free Sol-Attn block-sparse attention (the same ComfyUI node as
# VSA, whose method needs FastH3's learned coarse branch and so cannot run on the official bases); `source_path` is the file's path in its repo when not at the top level, `sha256` its hash.
# Pika H3 Flex's six-step schedule: its knots 1000/750/500/250/100/40/0 after the video shift 12.
PIKA_SIGMAS = (1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0)

LORAS = {
    # Needs its own LoRA node (adaLN on the pruned base) and its own 4-step sampler.
    "larryvrh-v4": {
        "group": "h3",
        "family": "fl2va",
        "source": "larryvrh/MiniMax-H3-Turbo-Lora",
        "label": "larryvrh v4 4-step",
        "loader": "larryvrh",
        "file": "minimax_h3_turbo_v4_step600_ema.safetensors",
        "steps": 4,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "resolutions": ["544p", "768p"],
    },
    # FastVideo's 8-step distilled bases, settings from ComfyUI's FastH3 templates.
    "fasth3-v2": {
        "group": "fasth3",
        "family": "fl2va",
        "label": "FastH3 V2 8-step",
        "loader": "none",
        "file": None,
        "unet": "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
        "unet_source": "FastVideo/FastVideo-FastH3-Comfy",
        "sampler": "res_multistep",
        "steps": 8,
        "shift_video": 10.0,
        "shift_audio": 3.0,
        "vsa": {"keep": 10, "start": 0.2},
        "fixed": True,
        "resolutions": ["544p", "768p"],
    },
    "fasth3-omniref": {
        "group": "fasth3",
        "family": "ref2va",
        "label": "FastH3 OmniRef 8-step",
        "loader": "none",
        "file": None,
        "unet": "fasth3_omniref_8step_pruned_int8_convrot.safetensors",
        "unet_source": "FastVideo/FastH3-Omni-Comfy",
        "sampler": "euler",
        "steps": 8,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "vsa": {"keep": 10, "start": 0.0},
        "fixed": True,
        "resolutions": ["544p", "768p"],
    },
    # Pika H3 Flex: Comfy-format BF16 LoRAs on the official H3 bases, sampled like Comfy-Org's video_pika_h3_flex_*
    # templates: lcm (fresh noise every step) over ManualSigmas, the knots 1000/750/500/250/100/40/0 at shift 12. Pika
    # trains for 480p / 768p; generic 6-step sampling is not equivalent, so sampler and steps are fixed.
    "pika-flex-fl2va": {
        "group": "pika",
        "family": "fl2va",
        "source": "Pika-Labs/Pika-H3-Flex",
        "source_path": "comfy/loras/pika_h3_flex_fl2va_comfy_bf16.safetensors",
        "sha256": "68783e6447696ec365d13c0554941e070bee4cfea13c37da52498ec4f9a53f89",
        "label": "Pika H3 Flex 6-step",
        "loader": "core",
        "file": "pika_h3_flex_fl2va_comfy_bf16.safetensors",
        "sampler": "lcm",
        "steps": 6,
        "sigmas": PIKA_SIGMAS,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "fixed": True,
        "resolutions": ["544p", "768p"],
    },
    "pika-flex-ref2va": {
        "group": "pika",
        "family": "ref2va",
        "source": "Pika-Labs/Pika-H3-Flex",
        "source_path": "comfy/loras/pika_h3_flex_ref2va_comfy_bf16.safetensors",
        "sha256": "e4d058fbd027377b925a0ee607b514ddd9037873fee3afa05d73b302e6f7f8e1",
        "label": "Pika H3 Flex 6-step",
        "loader": "core",
        "file": "pika_h3_flex_ref2va_comfy_bf16.safetensors",
        "sampler": "lcm",
        "steps": 6,
        "sigmas": PIKA_SIGMAS,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "fixed": True,
        "resolutions": ["544p", "768p"],
    },
}
# Pika with sparse attention: the same LoRAs plus Sol-Attn (tau 1.3 keeps ~10% of key blocks exact). `-sparse-all` runs
# every step sparse (start 0); `-sparse` keeps the first of the six steps dense (start 0.2 at shift 12); the plain keys
# stay dense as in Pika's templates. Pika was not trained for sparse attention; it is a speed choice. The frontend
# recommends the first choice of a group, so each family lists all-sparse, first-step dense, dense, in that order.
for _key in ("pika-flex-fl2va", "pika-flex-ref2va"):
    _dense = LORAS.pop(_key)
    LORAS[_key + "-sparse-all"] = {
        **_dense,
        "label": "Pika H3 Flex 6-step · sparse",
        "sol": {"tau": 1.3, "start": 0.0},
    }
    LORAS[_key + "-sparse"] = {
        **_dense,
        "label": "Pika H3 Flex 6-step · sparse, dense 1st step",
        "sol": {"tau": 1.3, "start": 0.2},
    }
    LORAS[_key] = {**_dense, "label": "Pika H3 Flex 6-step · dense"}

# larryvrh with every step sparse, listed right after its dense entry; same Sol-Attn settings as Pika. Sampling halved
# (49 -> 27 s for 5 s at 768p) with comparable quality, so it is the recommended H3 choice.
_order = list(LORAS)
for _key in ("larryvrh-v4",):
    LORAS[_key + "-sparse"] = {
        **LORAS[_key],
        "label": LORAS[_key]["label"] + " · sparse",
        "sol": {"tau": 1.3, "start": 0.0},
    }
    _order.insert(_order.index(_key) + 1, _key + "-sparse")
LORAS = {k: LORAS[k] for k in _order}

# Recommended model choice per (family, resolution), used when a request names none.
DEFAULT_LORAS = {
    "fl2va": {"544p": "larryvrh-v4-sparse", "768p": "larryvrh-v4-sparse"},
    "ref2va": {"544p": "fasth3-omniref", "768p": "fasth3-omniref"},
}

# Jobs created before LoRA selection stored a combined preset id instead. All of these name
# LoRAs that are gone (lightx2v was removed); those jobs keep the models they ran on in result.models.
LEGACY_PRESETS = {
    "544p": ("544p", "lightx2v-8step-v1.0"),
    "768p": ("768p", "lightx2v-4step-v1.3"),
    "ref-544p": ("544p", "lightx2v-ref-v0.1"),
    "ref-768p": ("768p", "lightx2v-ref-v1.0"),
}

ASPECT_RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "1:1": 1.0}

# Base checkpoints, all from the Comfy-Org/MiniMax-H3 repo (see scripts/prepare_models.py).
UNETS = {
    "fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}
MAX_REFS = 6
# Reference videos (ref2va): the node takes up to 3, 2-15 s at 24 fps. Their tokens join every attention call,
# so the engine trims them to the job's length and scales them to its pixel area (see _ref_video).
MAX_REF_VIDEOS = 3
REF_VIDEO_MIN_SECONDS = 2.0
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

# Music generation: MiniMax Music 3 with the settings of ComfyUI's text-to-music template. The text encoder
# writes the acoustic tokens autoregressively from caption + lyrics; the DiT then renders them.
MUSIC_REPO = "Comfy-Org/MiniMax-Music-3"
MUSIC_UNET = "minimax_music3_dit_int8_convrot.safetensors"
MUSIC_CLIP = "minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
MUSIC_VAE = "minimax_music3_dav.safetensors"
MUSIC = {
    "steps": 30,
    "cfg": 1.7,
    "sampler": "euler",
    "scheduler": "simple",
    "encode_cfg": 1.7,
    "encode_top_k": 50,
    "tile": 1536,
    "overlap": 64,
}
MUSIC_DURATION_RANGE = (10, 300)

# Upscaling finished videos: SeedVR2 3B with the settings of ComfyUI's SeedVR2 video upscale template
# (one euler step, lanczos pre-resize, no color correction). Two departures: the template samples the whole
# clip at once, here it is split into temporal chunks sized to free VRAM, overlapping by two latent frames
# that are crossfaded; and the template's VAEEncodeTiled / VAEDecodeTiled at 512 px are replaced by the plain
# VAE nodes, whose SeedVR2 path sizes the work from measured memory figures and runs whole frames when they
# fit (tiling itself only when not). The forced 512 px tiles made a 10 s 1080p clip spend 480 of 510 s in the VAE.
SEEDVR2_REPO = "Comfy-Org/SeedVR2"
SEEDVR2_UNET = "seedvr2_3b_int8_convrot.safetensors"
SEEDVR2_VAE = "seedvr2_ema_vae_fp16.safetensors"
UPSCALE = {
    "steps": 1,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "color": "none",
    "overlap": 2,
}
# FlashVSR v1.1 through T8star's FlashVSR+ node (custom_nodes/Comfyui-Flashvsr-Plus-T8), which streams the
# source mp4 through the one-step DiT in chunks, decodes with the Tiny (TC) decoder and copies the source audio.
# The node's defaults otherwise (temporal compression 4, decode every 28 latent frames, no color correction);
# int8 ConvRot DiT through comfy_kitchen. Its pipeline lives on the CPU between jobs.
FLASHVSR_REPO = "t8star/Flashvsr-Plus-Comfy"
FLASHVSR_NODE = "Comfyui-Flashvsr-Plus-T8"
FLASHVSR_DIR = "flashvsr_plus"  # under models/
FLASHVSR_FILES = (
    "v1/dit_int8_convrot.safetensors",
    "v1/LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
)
FLASHVSR = {
    "mode": "v1",
    "precision": "int8_convrot",
    "attention": "auto",
    "decoder": "tiny",
}
# RelayVSR with the settings of the workflow attached to ComfyUI's PR #17511: the source goes in at its native
# size and the checkpoint decides the factor. One euler step on every 15th frame, then the Flash decoder restores
# all frames from those latents and the source frames. Needs comfy_kitchen's window_attention (0.2.41+).
# Natively sized output has no room for a target resolution, so its targets are the factors.
RELAYVSR_REPO = "Comfy-Org/RelayVSR"
RELAYVSR_MODELS = {
    "2x": {
        "factor": 2,
        "unet": "relayvsr_x2_lightx2v_int8_convrot.safetensors",
        "vae": "relayvsr_flash_vae_x2_fp32.safetensors",
    },
    "4x": {
        "factor": 4,
        "unet": "relayvsr_x4_int8_convrot.safetensors",
        "vae": "relayvsr_flash_vae_x4_fp32.safetensors",
    },
}
RELAYVSR_CHECKSUMS = {
    "relayvsr_x2_lightx2v_int8_convrot.safetensors": "240e75e0c17cf2845d559b25254a230ff7458f81449d7a8c5e7e3fb8275c7b3c",
    "relayvsr_flash_vae_x2_fp32.safetensors": "b5bb0339acb5b52545911ea29a51b4c5b805776a779222b5a14591571f42e21c",
    "relayvsr_x4_int8_convrot.safetensors": "21303b3c5f9ab4e1fd3fe00df5dd2e0b343f0cf2e3d269e03a9872000a313a2e",
    "relayvsr_flash_vae_x4_fp32.safetensors": "7b8b8c99275fd5c4501b0589cc7b4c4506456a5df83160ef4e207e147b30f714",
}
RELAYVSR = {
    "steps": 1,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "detail": 1.0,
}
# Largest output short side: 4x of 544p. Decoded frames sit in system RAM until they are encoded.
RELAYVSR_MAX_SHORT_SIDE = 2176
# Upscalers a finished video can go through.
UPSCALERS = {
    "seedvr2": {"label": "SeedVR2 3B"},
    "flashvsr": {"label": "FlashVSR v1.1"},
    "relayvsr": {"label": "RelayVSR"},
}
# Target short side of each upscale choice.
UPSCALE_TARGETS = {"1080p": 1080, "1440p": 1440}

# Keyframe tags the I2VA prompts in the repo and the ComfyUI docs open with.
I2V_PREFIX = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)
LAST_FRAME_TAG = "At {:.2f} seconds into the target video, <Picture 2> (from [Shot 1]) is fully referenced."


def frame_count(duration):
    """Frames the model renders for `duration` seconds: the 17k+5 grid at 24 fps."""
    n = max(5, round(duration * FPS))
    while n % 17 != 5:
        n += 1
    return n


# Per-job LoRA strength; 1.0 is what the LoRAs were trained for.
LORA_STRENGTH_RANGE = (0.0, 2.0)

# Samplers a job may pick instead of its model choice's own; ComfyUI's H3 templates use res_multistep.
SAMPLERS = ("euler", "res_multistep", "er_sde")
# Per-job step count instead of the model choice's own; turbo LoRAs also run at more steps than they were distilled for.
STEPS_RANGE = (1, 30)


def unet_file(lora):
    """The base checkpoint a model choice runs on."""
    p = LORAS[lora]
    return p.get("unet") or UNETS[p["family"]]


def fixed_sampling(lora):
    """Whether a model choice ignores a job's sampler and steps: larryvrh's turbo sampler and the FastH3 bases."""
    p = LORAS[lora]
    return p["loader"] == "larryvrh" or p.get("fixed", False)


def sampler_name(lora, sampler=None):
    """The sampler a job runs: its own pick, else the model choice's. larryvrh always uses its turbo sampler."""
    p = LORAS[lora]
    if p["loader"] == "larryvrh":
        return "larryvrh turbo"
    return (not fixed_sampling(lora) and sampler) or p.get("sampler", "euler")


def model_info(lora, strength=1.0, sampler=None, steps=None):
    """The checkpoints and sampling settings a model choice runs on, as shown to users."""
    p = LORAS[lora]
    vsa, sol = p.get("vsa"), p.get("sol")
    return {
        "unet": unet_file(lora),
        "lora": p["file"],
        "lora_strength": strength if p["file"] else None,
        "steps": (not p.get("fixed") and steps) or p["steps"],
        "sampler": sampler_name(lora, sampler),
        "shift_video": p["shift_video"],
        "shift_audio": p["shift_audio"],
        "attention": (
            f"VSA {vsa['keep']}%"
            if vsa
            else f"Sol-Attn τ{sol['tau']}{', dense 1st step' if sol['start'] else ''}" if sol else None
        ),
    }


def music_model_info():
    return {
        "unet": MUSIC_UNET,
        "lora": None,
        "lora_strength": None,
        "steps": MUSIC["steps"],
        "sampler": MUSIC["sampler"],
        "cfg": MUSIC["cfg"],
        "text_encoder": MUSIC_CLIP,
    }


def upscale_targets(upscaler):
    """The targets an upscaler offers: fixed short sides, or RelayVSR's native factors."""
    return list(RELAYVSR_MODELS if upscaler == "relayvsr" else UPSCALE_TARGETS)


def upscale_model_info(upscaler="seedvr2", target=None):
    """What an upscale job runs on, as shown to users."""
    if upscaler == "relayvsr":
        files = RELAYVSR_MODELS.get(target, RELAYVSR_MODELS["2x"])
        return {
            "unet": files["unet"],
            "lora": None,
            "lora_strength": None,
            "steps": RELAYVSR["steps"],
            "sampler": RELAYVSR["sampler"],
            "vae": files["vae"],
        }
    if upscaler == "flashvsr":
        return {
            "unet": "flashvsr_v1.1_dit_int8_convrot",
            "lora": None,
            "lora_strength": None,
            "steps": 1,
            "sampler": None,
            "vae": "TCDecoder",
        }
    return {
        "unet": SEEDVR2_UNET,
        "lora": None,
        "lora_strength": None,
        "steps": UPSCALE["steps"],
        "sampler": UPSCALE["sampler"],
        "vae": SEEDVR2_VAE,
    }


def upscale_size(width, height, target):
    """Width/height of a `width`×`height` video at `target`: a native factor, or a short side with both even (H.264)."""
    if target in RELAYVSR_MODELS:
        factor = RELAYVSR_MODELS[target]["factor"]
        return width * factor, height * factor
    scale = UPSCALE_TARGETS[target] / min(width, height)
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def image_size(ratio):
    """Picture width/height of about one megapixel at `ratio`, each a multiple of 16."""
    width = max(16, round(math.sqrt(IMAGE_AREA * ratio) / 16) * 16)
    height = max(16, round(math.sqrt(IMAGE_AREA / ratio) / 16) * 16)
    return width, height


def canvas_size(resolution, ratio):
    """Width/height with the resolution's pixel area at `ratio`, each a multiple of 32."""
    area = RESOLUTIONS[resolution]
    width = max(32, round(math.sqrt(area * ratio) / 32) * 32)
    height = max(32, round(math.sqrt(area / ratio) / 32) * 32)
    return width, height


def with_keyframe_prefix(prompt, duration, last_frame=False):
    """Reference the keyframes the way the official I2VA prompts do, unless the prompt already does."""
    tags = []
    if "<Picture 1>" not in prompt:
        tags.append(I2V_PREFIX)
    if last_frame and "<Picture 2>" not in prompt:
        tags.append(LAST_FRAME_TAG.format((frame_count(duration) - 1) / FPS))
    if not tags:
        return prompt
    prefix = " ".join(tags)
    label = "integrated_multimodal_description:"
    if prompt.lstrip().startswith(label):
        body = prompt.lstrip()[len(label) :].lstrip()
        return f"{label} {prefix}\n\n{body}"
    return f"{prefix}\n\n{prompt}"
