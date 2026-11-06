"""Jobs endpoints."""

from aiohttp import web
from image_models import IMAGE_MODELS
from audio_catalog import SPEECH_MODELS, ASR_MODELS
import random
import time
from model_catalog import (
    RELAYVSR_MAX_SHORT_SIDE,
    RELAYVSR_MODELS,
    UPSCALERS,
    UPSCALE_TARGETS,
    upscale_size,
    upscale_targets,
)
from server_settings import HISTORY_KINDS, MAX_SEED
from job_views import job_view, queue_positions
from .common import bad_request
from h3_jobs import ACTIVE


async def list_jobs(request):
    store = request.app["store"]
    kind = request.query.get("kind")
    if kind is not None and kind not in HISTORY_KINDS:
        return bad_request("kind must be one of " + ", ".join(HISTORY_KINDS))
    try:
        limit = max(1, min(int(request.query.get("limit", "60")), 500))
        offset = max(int(request.query.get("offset", "0")), 0)
    except ValueError:
        return bad_request("limit and offset must be integers")
    image_model = request.query.get("image_model")
    if image_model is not None and (kind != "image" or image_model not in IMAGE_MODELS):
        return bad_request("image_model requires kind=image and a known image model")
    audio_model = request.query.get("audio_model")
    if audio_model is not None and (kind not in ("speech", "asr") or audio_model not in (SPEECH_MODELS if kind == "speech" else ASR_MODELS)):
        return bad_request("audio_model requires a matching speech/asr history kind")
    modes = HISTORY_KINDS[kind] if kind else None
    positions = queue_positions(store)
    return web.json_response(
        [job_view(j, positions) for j in store.list(limit, offset, modes=modes, image_model=image_model, audio_model=audio_model)],
        headers={"X-Total-Count": str(store.count(modes=modes, image_model=image_model, audio_model=audio_model))},
    )


async def get_job(request):
    store = request.app["store"]
    job = store.get(request.match_info["id"])
    if not job:
        raise web.HTTPNotFound()
    return web.json_response(job_view(job, queue_positions(store)))


async def upscale_job(request):
    """Queue an upscale of a finished video job; the new job keeps the source's prompt for display."""
    store, worker = request.app["store"], request.app["worker"]
    source = store.get(request.match_info["id"])
    if not source:
        raise web.HTTPNotFound()
    form = await request.post()
    upscaler = form.get("upscaler") or "seedvr2"
    if upscaler not in UPSCALERS:
        return bad_request(f"upscaler must be one of {', '.join(UPSCALERS)}")
    targets = upscale_targets(upscaler)
    target = form.get("target") or targets[0]
    if target not in targets:
        return bad_request(f"target must be one of {', '.join(targets)}")
    if source["status"] != "done" or not source["video"]:
        return bad_request("only finished videos can be upscaled")
    short_side = min(source["width"], source["height"])
    if upscaler == "relayvsr":
        if short_side * RELAYVSR_MODELS[target]["factor"] > RELAYVSR_MAX_SHORT_SIDE:
            return bad_request(
                f"{target} would exceed the {RELAYVSR_MAX_SHORT_SIDE} px short side limit"
            )
    elif short_side >= UPSCALE_TARGETS[target]:
        return bad_request(f"the video is already {target} or larger")
    width, height = upscale_size(source["width"], source["height"], target)
    raw_seed = (form.get("seed") or "").strip()
    seed = (
        int(raw_seed) % MAX_SEED
        if raw_seed.lstrip("-").isdigit()
        else random.randrange(MAX_SEED)
    )
    job = store.create(
        mode="upscale",
        source=source["id"],
        upscaler=upscaler,
        preset=target,
        prompt=source["prompt"],
        width=width,
        height=height,
        duration=source["duration"],
        seed=seed,
        refs=[],
    )
    if worker.state == "ready":
        worker.queue.put(job["id"])
    return web.json_response(job_view(job, queue_positions(store)), status=201)


async def cancel_job(request):
    store, worker = request.app["store"], request.app["worker"]
    job = store.get(request.match_info["id"])
    if not job:
        raise web.HTTPNotFound()
    if job["status"] == "queued":
        store.update(job["id"], status="canceled", finished_at=time.time())
    elif job["status"] == "running":
        worker.cancel(job["id"])
    return web.json_response({"ok": True})


async def delete_job(request):
    settings = request.app["settings"]
    store = request.app["store"]
    job = store.get(request.match_info["id"])
    if not job:
        raise web.HTTPNotFound()
    if job["status"] in ACTIVE:
        return bad_request("cancel the job before deleting it")
    for folder, name in (
        (settings.models3d, job["model_3d"]),
        (settings.videos, job["video"]),
        (settings.pictures, job["picture"]),
        (settings.speech if job["mode"] == "tts" else settings.music, job["audio"]),
        (settings.uploads, job["input_media"]),
        (settings.llm, (job["result"] or {}).get("output") if job["mode"] == "llm" else None),
        *((settings.transcripts, n) for n in (job["result"] or {}).get("transcript", {}).get("exports", {}).values()),
        (settings.thumbs, job["thumb"]),
        (settings.uploads, job["image"]),
        (settings.uploads, job["guide_audio"]),
        (settings.uploads, job["last_image"]),
        *((settings.uploads, n) for n in job["refs"]),
        *((settings.uploads, n) for n in job["ref_videos"]),
    ):
        if name:
            (folder / name).unlink(missing_ok=True)
    store.delete(job["id"])
    return web.json_response({"ok": True})
