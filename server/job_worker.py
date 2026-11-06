"""Single-threaded inference queue and engine lifecycle ownership."""

import concurrent.futures
import logging
import queue
import threading
import time
import traceback
from engine_errors import Canceled
from runtime_memory import is_memory_error
from job_views import job_strength

log = logging.getLogger("h3")


class Worker(threading.Thread):
    """Owns the engine; executes queued jobs strictly one at a time."""

    def __init__(self, store, engine, settings):
        super().__init__(daemon=True, name="h3-worker")
        self.settings = settings
        self.store = store
        self.engine = engine
        self.queue = queue.Queue()
        self.running_id = None
        self.speech_jobs = {}  # running pooled speech job id -> cancel requested
        self.state = "loading"
        self.load_error = None
        self.loaded_at = None

    def run(self):
        try:
            t0 = time.perf_counter()
            self.engine.load()
            self.loaded_at = time.time()
            self.state = "ready"
            log.info("engine ready in %.1fs", time.perf_counter() - t0)
        except Exception:
            self.state = "error"
            self.load_error = traceback.format_exc()
            log.error("engine failed to load:\n%s", self.load_error)
            return
        for job_id in self.store.queued_ids():
            self.queue.put(job_id)
        while True:
            item = self.queue.get()
            if callable(item):
                item()
            else:
                self._run_one(item)
                if self.state == "error":
                    return

    def call(self, fn):
        """Run fn on the worker thread after the jobs queued before it; returns a concurrent Future."""
        future = concurrent.futures.Future()

        def task():
            try:
                future.set_result(fn())
            except Exception as exc:
                future.set_exception(exc)

        self.queue.put(task)
        return future

    def cancel(self, job_id):
        """Cancel one running job; pooled speech jobs stop only their own audio.cpp instance."""
        if job_id in self.speech_jobs:
            self.speech_jobs[job_id] = True
            self.engine.audio_cpp.request_stop(owner=job_id)
        elif self.running_id == job_id:
            self.engine.cancel()

    def _next_speech(self):
        """Pop the queue head if it is a speech job (any speech model); keeps queue order."""
        with self.queue.mutex:
            while self.queue.queue:
                head = self.queue.queue[0]
                if callable(head):
                    return None
                job = self.store.get(head)
                if not job or job["status"] != "queued":
                    self.queue.queue.popleft()  # _run_one would skip it as well
                    continue
                if job["mode"] != "tts":
                    return None
                self.queue.queue.popleft()
                return job
        return None

    def _run_speech_batch(self, first):
        """Run consecutive queued speech jobs on the audio.cpp pool, one per instance, rolling."""
        task = self.engine.speech_task(first["audio_model"], first["audio_options"] or {})
        try:
            self.engine._prepare_task(task)
        except Exception:
            return False  # the single-job path reports the failure with memory recovery
        size = self.engine.audio_cpp.size
        self.running_id = first["id"]
        running, job = {}, first
        with concurrent.futures.ThreadPoolExecutor(size, thread_name_prefix="h3-speech") as pool:
            while True:
                while len(running) < size:
                    job = job or self._next_speech()
                    if not job:
                        break
                    self.speech_jobs[job["id"]] = False
                    running[pool.submit(self._run_speech, job)] = job["id"]
                    job = None
                if not running:
                    break
                done, _ = concurrent.futures.wait(list(running), timeout=0.5,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    running.pop(future)
        self.running_id = None
        try:
            self.engine.memory.observe()
        except Exception:
            log.exception("failed to record GPU memory status")
        return True

    def _run_speech(self, job):
        job_id = job["id"]

        def check():
            if self.speech_jobs.get(job_id) or self.engine._cancel:
                raise Canceled()

        last = {"t": 0.0}

        def on_progress(stage, fraction):
            now = time.monotonic()
            if fraction in (0.0, 1.0) or now - last["t"] > 0.5:
                last["t"] = now
                self.store.update(job_id, stage=stage, progress=round(fraction, 3))

        self.store.update(job_id, status="running", started_at=time.time(), stage="encoding", progress=0.0)
        try:
            source = self.settings.uploads / job["input_media"] if job["input_media"] else None
            output = f"{job_id}.wav"
            result = self.engine.generate_speech(model=job["audio_model"], text=job["prompt"],
                options=job["audio_options"] or {}, source_path=source, seed=job["seed"],
                audio_path=self.settings.speech / output, on_progress=on_progress,
                prepared=True, check=check, owner=job_id)
            self.store.update(job_id, status="done", stage="done", progress=1.0, audio=output,
                duration=result["seconds"], result=result, finished_at=time.time())
            log.info("job %s done tts", job_id)
        except Canceled:
            self.store.update(job_id, status="canceled", finished_at=time.time())
            log.info("job %s canceled", job_id)
        except Exception:
            err = traceback.format_exc()
            self.store.update(job_id, status="failed", error=err[-4000:], finished_at=time.time())
            log.error("job %s failed:\n%s", job_id, err)
        finally:
            self.speech_jobs.pop(job_id, None)

    def _run_one(self, job_id):
        job = self.store.get(job_id)
        if not job or job["status"] != "queued":
            return
        pool = getattr(self.engine, "audio_cpp", None)
        if job["mode"] == "tts" and pool is not None and pool.size > 1 and self._run_speech_batch(job):
            return
        self.running_id = job_id
        self.store.update(
            job_id,
            status="running",
            started_at=time.time(),
            stage="encoding",
            progress=0.0,
        )
        last = {"t": 0.0}

        def on_progress(stage, fraction):
            now = time.monotonic()
            if fraction in (0.0, 1.0) or now - last["t"] > 0.5:
                last["t"] = now
                self.store.update(job_id, stage=stage, progress=round(fraction, 3))

        recover_memory = False
        video, thumb = f"{job_id}.mp4", f"{job_id}.jpg"
        try:
            if job["mode"] in ("tts", "asr"):
                options = job["audio_options"] or {}
                source = self.settings.uploads / job["input_media"] if job["input_media"] else None
                output = f"{job_id}.wav" if job["mode"] == "tts" else None
                if job["mode"] == "tts":
                    result = self.engine.generate_speech(model=job["audio_model"], text=job["prompt"], options=options,
                        source_path=source, seed=job["seed"], audio_path=self.settings.speech / output, on_progress=on_progress)
                else:
                    result = self.engine.transcribe(model=job["audio_model"], source_path=source, options=options,
                        folder=self.settings.transcripts, job_id=job_id, on_progress=on_progress)
                self.store.update(job_id, status="done", stage="done", progress=1.0, audio=output,
                    duration=result["seconds"], result=result, finished_at=time.time())
                log.info("job %s done %s", job_id, job["mode"])
                return
            if job["mode"] == "llm":
                output = f"{job_id}.jsonl"
                result = self.engine.run_llm(requests_path=self.settings.uploads / job["input_media"],
                    output_path=self.settings.llm / output, concurrency=(job["llm_options"] or {}).get("concurrency", 16),
                    on_progress=on_progress)
                self.store.update(job_id, status="done", stage="done", progress=1.0, result=result, finished_at=time.time())
                log.info("job %s done llm %s", job_id, {k: result[k] for k in ("requests", "failed", "tokens_per_second")})
                return
            if job["mode"] == "i23d":
                model = f"{job_id}.glb"
                result = self.engine.generate_3d(
                    image_path=self.settings.uploads / job["image"],
                    model_path=self.settings.models3d / model,
                    thumb_path=self.settings.thumbs / thumb,
                    seed=job["seed"],
                    **job["mesh_options"],
                    on_progress=on_progress,
                )
                self.store.update(
                    job_id,
                    status="done",
                    stage="done",
                    progress=1.0,
                    model_3d=model,
                    thumb=thumb,
                    result=result,
                    finished_at=time.time(),
                )
                log.info("job %s done %s", job_id, result)
                return
            if job["mode"] in ("t2i", "i2i"):
                picture = f"{job_id}.png"
                result = self.engine.generate_image(
                    image_model=job["image_model"] or "moss",
                    ref_paths=[self.settings.uploads / name for name in job["refs"]],
                    prompt=job["prompt"],
                    width=job["width"],
                    height=job["height"],
                    seed=job["seed"],
                    image_path=self.settings.pictures / picture,
                    thumb_path=self.settings.thumbs / thumb,
                    on_progress=on_progress,
                )
                self.store.update(
                    job_id,
                    status="done",
                    stage="done",
                    progress=1.0,
                    picture=picture,
                    thumb=thumb,
                    result=result,
                    finished_at=time.time(),
                )
                log.info("job %s done %s", job_id, result)
                return
            if job["mode"] == "t2m":
                audio = f"{job_id}.mp3"
                result = self.engine.generate_music(
                    caption=job["prompt"],
                    lyrics=job["lyrics"] or "",
                    duration=job["duration"],
                    seed=job["seed"],
                    audio_path=self.settings.music / audio,
                    on_progress=on_progress,
                )
                self.store.update(
                    job_id,
                    status="done",
                    stage="done",
                    progress=1.0,
                    audio=audio,
                    result=result,
                    finished_at=time.time(),
                )
                log.info("job %s done %s", job_id, result)
                return
            if job["mode"] == "upscale":
                source = self.store.get(job["source"])
                if (
                    not source
                    or not source["video"]
                    or not (self.settings.videos / source["video"]).is_file()
                ):
                    raise ValueError("the source video has been deleted")
                paths = dict(
                    source_path=self.settings.videos / source["video"],
                    seed=job["seed"],
                    video_path=self.settings.videos / video,
                    thumb_path=self.settings.thumbs / thumb,
                    on_progress=on_progress,
                )
                if job["upscaler"] == "relayvsr":
                    result = self.engine.relayvsr_video(target=job["preset"], **paths)
                else:
                    upscale = (
                        self.engine.flashvsr_video
                        if job["upscaler"] == "flashvsr"
                        else self.engine.upscale_video
                    )
                    result = upscale(width=job["width"], height=job["height"], **paths)
                self.store.update(
                    job_id,
                    status="done",
                    stage="done",
                    progress=1.0,
                    video=video,
                    thumb=thumb,
                    result=result,
                    finished_at=time.time(),
                )
                log.info("job %s done %s", job_id, result)
                return
            result = self.engine.generate(
                lora=job["lora"],
                prompt=job["prompt"],
                width=job["width"],
                height=job["height"],
                duration=job["duration"],
                seed=job["seed"],
                lora_strength=job_strength(job),
                sampler=job["sampler"],
                steps=job["steps"],
                image_path=self.settings.uploads / job["image"]
                if job["image"]
                else None,
                last_image_path=self.settings.uploads / job["last_image"]
                if job["last_image"]
                else None,
                ref_paths=[self.settings.uploads / name for name in job["refs"]],
                ref_video_paths=[
                    self.settings.uploads / name for name in job["ref_videos"]
                ],
                audio_path=self.settings.uploads / job["guide_audio"]
                if job["guide_audio"]
                else None,
                audio_start=job["audio_start"] or 0.0,
                video_path=self.settings.videos / video,
                thumb_path=self.settings.thumbs / thumb,
                on_progress=on_progress,
            )
            self.store.update(
                job_id,
                status="done",
                stage="done",
                progress=1.0,
                video=video,
                thumb=thumb,
                result=result,
                finished_at=time.time(),
            )
            log.info("job %s done %s", job_id, result)
        except Canceled:
            self.store.update(job_id, status="canceled", finished_at=time.time())
            log.info("job %s canceled", job_id)
        except Exception as exc:
            recover_memory = is_memory_error(exc, getattr(self.engine, "mm", None))
            err = traceback.format_exc()
            self.store.update(
                job_id, status="failed", error=err[-4000:], finished_at=time.time()
            )
            log.error("job %s failed:\n%s", job_id, err)
        finally:
            if hasattr(self.engine, "memory") and not recover_memory:
                try:
                    self.engine.memory.observe()
                except Exception:
                    log.exception("failed to record GPU memory status")
            self.running_id = None
        # The except block must be exited first: traceback frames can otherwise
        # keep failed inference tensors alive while we try to release VRAM.
        if recover_memory:
            try:
                state = self.engine.memory.recover()
                self.store.update(
                    job_id,
                    error="GPU memory exhausted. Task stopped and cached models released. "
                    "Try fewer references, a smaller output, or close other GPU programs.",
                )
                log.warning("job %s memory recovered: %s", job_id, state)
            except Exception:
                self.state = "error"
                self.load_error = "GPU memory recovery failed; restart the server before running more jobs."
                log.exception("GPU recovery failed; stopping inference worker")
                self.store.update(job_id, error=self.load_error)
