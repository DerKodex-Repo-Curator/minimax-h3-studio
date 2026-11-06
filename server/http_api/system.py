"""System endpoints."""

import logging
from aiohttp import web
from image_models import IMAGE_MODELS, picture_size
from audio_catalog import SPEECH_MODELS, ASR_MODELS, LANGUAGES, OMNI_INSTRUCTION_GROUPS
import asyncio
from model_catalog import (
    ASPECT_RATIOS,
    DEFAULT_LORAS,
    GROUPS,
    LORAS,
    LORA_STRENGTH_RANGE,
    MAX_REFS,
    MAX_REF_VIDEOS,
    MUSIC_DURATION_RANGE,
    RESOLUTIONS,
    SAMPLERS,
    STEPS_RANGE,
    RELAYVSR_MAX_SHORT_SIDE,
    UPSCALERS,
    UPSCALE_TARGETS,
    upscale_targets,
    canvas_size,
    fixed_sampling,
    image_model_info,
    image_size,
    model_info,
    music_model_info,
    pixal_model_info,
    upscale_model_info,
)
from server_settings import DURATION_RANGE

log = logging.getLogger("h3")


async def health(request):
    worker = request.app["worker"]
    return web.json_response(
        {
            "state": worker.state,
            "load_error": worker.load_error,
            "running": worker.running_id,
            "family": worker.engine.family,
            "unet": worker.engine.unet,
            "loaded_models": [
                label
                for attr, label in (
                    ("_picture_models", IMAGE_MODELS.get(getattr(worker.engine, "_picture_model", None), {}).get("label", "Picture model")),
                    ("_music", "MiniMax Music 3"),
                    ("_pixal", "Pixal3D"),
                    ("_seedvr2", "SeedVR2"),
                    ("_flashvsr", "FlashVSR"),
                    ("_relayvsr", "RelayVSR"),
                )
                if getattr(worker.engine, attr, None) is not None
            ],
            "queued": len(request.app["store"].queued_ids()),
            "loaded_at": worker.loaded_at,
            "audio_cpp": worker.engine.audio_cpp.status if getattr(worker.engine, "audio_cpp", None) else None,
            "llm": worker.engine.llm.status if getattr(worker.engine, "llm", None) else None,
            "memory": dict(worker.engine.memory.status) if hasattr(worker.engine, "memory") else None,
        }
    )


async def presets(request):
    return web.json_response(
        {
            "resolutions": [
                {
                    "id": k,
                    "sizes": {a: canvas_size(k, r) for a, r in ASPECT_RATIOS.items()},
                }
                for k in RESOLUTIONS
            ],
            "groups": [{"id": k, "label": v} for k, v in GROUPS.items()],
            "loras": [
                {
                    "id": k,
                    "group": v["group"],
                    "family": v["family"],
                    "label": v["label"],
                    "steps": v["steps"],
                    "fixed": fixed_sampling(k),
                    "resolutions": v["resolutions"],
                    "models": model_info(k),
                }
                for k, v in LORAS.items()
            ],
            "default_loras": DEFAULT_LORAS,
            "aspect_ratios": list(ASPECT_RATIOS),
            "duration_range": DURATION_RANGE,
            "max_refs": MAX_REFS,
            "max_ref_videos": MAX_REF_VIDEOS,
            "lora_strength_range": LORA_STRENGTH_RANGE,
            "samplers": SAMPLERS,
            "steps_range": STEPS_RANGE,
            "mesh": {
                "resolutions": [1024, 1536],
                "texture_sizes": [1024, 2048, 4096],
                "models": pixal_model_info(),
            },
            "image": {
                "sizes": {a: image_size(r) for a, r in ASPECT_RATIOS.items()},
                "models": image_model_info(),
                "default_model": "moss",
                "choices": [{"id": key, "label": p["label"], "max_refs": p["max_refs"],
                             "edit": bool(p["max_refs"]), "models": image_model_info(key),
                             "sizes": {a: picture_size(key, ratio) for a, ratio in ASPECT_RATIOS.items()}}
                            for key, p in IMAGE_MODELS.items()],
            },
            "speech": {"instruction_groups": OMNI_INSTRUCTION_GROUPS, "models": [{"id": k, **v} for k, v in SPEECH_MODELS.items()], "languages": LANGUAGES, "reference_seconds": [0.2, 30]},
            "asr": {"models": [{"id": k, **v} for k, v in ASR_MODELS.items()], "languages": LANGUAGES, "max_seconds": 7200},
            "llm": {"model": request.app["worker"].engine.llm.model, "concurrency": request.app["settings"].strata_concurrency}
            if getattr(request.app["worker"].engine, "llm", None) else None,
            "music": {
                "duration_range": MUSIC_DURATION_RANGE,
                "default_duration": 120,
                "models": music_model_info(),
            },
            "upscale": {
                "targets": list(UPSCALE_TARGETS),
                "relayvsr_max_short_side": RELAYVSR_MAX_SHORT_SIDE,
                "upscalers": [
                    {
                        "id": k,
                        "label": v["label"],
                        "targets": upscale_targets(k),
                        "models": upscale_model_info(k),
                    }
                    for k, v in UPSCALERS.items()
                ],
            },
        }
    )


async def unload(request):
    """Free the GPU for another program while idle; the next job reloads its models."""
    store, worker = request.app["store"], request.app["worker"]
    if worker.state != "ready":
        return web.json_response({"error": f"the engine is {worker.state}"}, status=503)
    if worker.running_id or store.queued_ids():
        return web.json_response({"error": "jobs are running or queued"}, status=409)
    result = await asyncio.wrap_future(worker.call(worker.engine.unload))
    log.info("models unloaded %s", result)
    return web.json_response(result)
