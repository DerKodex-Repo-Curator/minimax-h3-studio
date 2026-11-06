"""Worker regression tests."""

from api_support import APIAppTestCase
from types import SimpleNamespace
from unittest.mock import Mock
from job_worker import Worker
from engine_errors import Canceled


class WorkerTests(APIAppTestCase):
    async def test_worker_dispatch(self):
        job = self.store.create(
            mode="i23d",
            prompt="object",
            image="input.png",
            seed=42,
            mesh_options={
                "resolution": 1024,
                "texture_size": 2048,
                "remove_background": True,
            },
        )
        calls = []
        engine = SimpleNamespace(
            generate_3d=lambda **kw: calls.append(kw) or {"faces": 100}
        )
        worker = Worker(self.store, engine, self.settings)
        worker._run_one(job["id"])
        result = self.store.get(job["id"])
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["model_3d"], job["id"] + ".glb")
        self.assertEqual(calls[0]["resolution"], 1024)
        self.assertIsNone(worker.running_id)

    async def test_worker_dispatches_every_pipeline_with_settings_paths(self):
        methods = (
            "generate",
            "generate_image",
            "generate_music",
            "generate_3d",
            "upscale_video",
            "flashvsr_video",
            "relayvsr_video",
        )
        engine = SimpleNamespace(
            **{method: Mock(return_value={"test": method}) for method in methods}
        )
        worker = Worker(self.store, engine, self.settings)
        source = self.store.create(
            mode="t2v", prompt="source", video="source.mp4", width=960, height=544
        )
        (self.settings.videos / "source.mp4").write_bytes(b"fixture")
        for mode, method, result_column, folder, extra in (
            ("t2v", "generate", "video", self.settings.videos, {}),
            ("t2i", "generate_image", "picture", self.settings.pictures, {}),
            ("t2m", "generate_music", "audio", self.settings.music, {}),
            (
                "i23d",
                "generate_3d",
                "model_3d",
                self.settings.models3d,
                {
                    "mesh_options": {
                        "resolution": 1024,
                        "texture_size": 2048,
                        "remove_background": True,
                    }
                },
            ),
            (
                "upscale",
                "upscale_video",
                "video",
                self.settings.videos,
                {"source": source["id"]},
            ),
            (
                "upscale",
                "flashvsr_video",
                "video",
                self.settings.videos,
                {"source": source["id"], "upscaler": "flashvsr"},
            ),
            (
                "upscale",
                "relayvsr_video",
                "video",
                self.settings.videos,
                {"source": source["id"], "upscaler": "relayvsr", "preset": "2x"},
            ),
        ):
            job = self.store.create(
                mode=mode,
                prompt="fixture",
                seed=42,
                image="input.png",
                refs=[],
                **extra,
            )
            worker._run_one(job["id"])
            result = self.store.get(job["id"])
            self.assertEqual(result["status"], "done", result["error"])
            self.assertEqual(result["result"]["test"], method)
            args = getattr(engine, method).call_args.kwargs
            path_key = {
                "video": "video_path",
                "picture": "image_path",
                "audio": "audio_path",
                "model_3d": "model_path",
            }[result_column]
            self.assertEqual(args[path_key], folder / result[result_column])
            self.assertIsNone(worker.running_id)
        engine.generate_music.side_effect = Canceled()
        job = self.store.create(mode="t2m", prompt="cancel", seed=42)
        worker._run_one(job["id"])
        self.assertEqual(self.store.get(job["id"])["status"], "canceled")


class PooledSpeechTests(APIAppTestCase):
    async def test_consecutive_speech_jobs_run_one_per_instance(self):
        import threading
        import time

        options = {"speech_mode": "generate", "voice": "Ryan"}
        jobs = [self.store.create(mode="tts", prompt=f"line {i}", seed=i, audio_model="qwen3_tts",
                                  audio_options=options) for i in range(7)]
        other = self.store.create(mode="t2i", prompt="picture", seed=1, width=512, height=512)
        state = {"now": 0, "peak": 0, "prepared": 0}
        lock = threading.Lock()

        def generate_speech(*, audio_path, prepared, check, owner, **kw):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return {"seconds": 1.0}

        engine = SimpleNamespace(
            audio_cpp=SimpleNamespace(size=3, request_stop=Mock()),
            speech_task=lambda model, opts: ("audio_cpp", model),
            _prepare_task=lambda task: state.__setitem__("prepared", state["prepared"] + 1),
            _cancel=False,
            generate_speech=generate_speech,
            generate_image=Mock(return_value={}),
            memory=SimpleNamespace(observe=Mock()),
        )
        worker = Worker(self.store, engine, self.settings)
        for job in jobs[1:]:
            worker.queue.put(job["id"])
        worker.queue.put(other["id"])
        worker._run_one(jobs[0]["id"])
        self.assertEqual([self.store.get(j["id"])["status"] for j in jobs], ["done"] * 7)
        self.assertEqual(state["peak"], 3)
        self.assertEqual(state["prepared"], 1)
        self.assertEqual(self.store.get(other["id"])["status"], "queued")  # left for the main loop
        self.assertEqual(worker.queue.get(), other["id"])
        self.assertIsNone(worker.running_id)
        self.assertEqual(worker.speech_jobs, {})
