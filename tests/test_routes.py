"""Routes regression tests."""

from api_support import APIAppTestCase


class RoutesTests(APIAppTestCase):
    async def test_cancel_delete_and_upscale_routes_use_injected_paths(self):
        response = await self.client.post("/api/jobs", data=self.form(seed=42))
        job = await response.json()
        uploaded = self.settings.uploads / job["image"]
        response = await self.client.post(f"/api/jobs/{job['id']}/cancel")
        self.assertEqual(response.status, 200)
        self.assertEqual(self.store.get(job["id"])["status"], "canceled")
        response = await self.client.delete(f"/api/jobs/{job['id']}")
        self.assertEqual(response.status, 200)
        self.assertFalse(uploaded.exists())
        source = self.store.create(
            mode="t2v",
            prompt="source",
            status="done",
            video="source.mp4",
            width=960,
            height=544,
        )
        self.store.update(source["id"], status="done")
        response = await self.client.post(
            f"/api/jobs/{source['id']}/upscale",
            data={"upscaler": "flashvsr", "target": "1080p"},
        )
        self.assertEqual(response.status, 201, await response.text())
        upscale = await response.json()
        self.assertEqual(upscale["source"], source["id"])
        self.assertEqual(upscale["upscaler"], "flashvsr")
        response = await self.client.post(
            f"/api/jobs/{source['id']}/upscale",
            data={"upscaler": "relayvsr", "target": "4x"},
        )
        self.assertEqual(response.status, 201, await response.text())
        upscale = await response.json()
        self.assertEqual((upscale["width"], upscale["height"]), (3840, 2176))
        self.assertEqual(upscale["models"]["unet"], "relayvsr_x4_int8_convrot.safetensors")
        response = await self.client.post(
            f"/api/jobs/{source['id']}/upscale",
            data={"upscaler": "relayvsr", "target": "1080p"},
        )
        self.assertEqual(response.status, 400)
        large = self.store.create(
            mode="t2v", prompt="large", video="large.mp4", width=1344, height=768
        )
        self.store.update(large["id"], status="done")
        response = await self.client.post(
            f"/api/jobs/{large['id']}/upscale",
            data={"upscaler": "relayvsr", "target": "4x"},
        )
        self.assertEqual(response.status, 400)
