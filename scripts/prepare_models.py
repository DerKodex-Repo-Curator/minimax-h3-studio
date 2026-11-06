#!/usr/bin/env python3
"""Prepare Studio ComfyUI checkpoints and the native audio.cpp CUDA runtime/models.

The file list comes from server/model_catalog.py (base DiTs, text encoder, VAEs and
every entry in LORAS, including FastH3 bases, plus the image_models.py catalog,
MiniMax Music 3, upscalers and Pixal3D), plus audio_catalog.py for native speech/ASR,
so adding a LoRA there is enough for it to be prepared.
Standard library only: run it with any Python 3.9+, no ComfyUI env needed.

    python scripts/prepare_models.py --comfyui /path/to/ComfyUI
    python scripts/prepare_models.py --list                 # show what would be fetched
    python scripts/prepare_models.py --skip ref2va fasth3   # leave out groups

Existing files are skipped; interrupted downloads resume from their .part file.
HF_ENDPOINT selects a mirror (e.g. https://hf-mirror.com), HF_TOKEN is sent if set.
ComfyUI itself and the larryvrh custom node are not installed by this script.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))

from model_catalog import (  # noqa: E402
    AUDIO_VAE,
    CLIP,
    PIXAL_FILES,
    FLASHVSR_DIR,
    FLASHVSR_FILES,
    FLASHVSR_NODE,
    FLASHVSR_REPO,
    LORAS,
    RELAYVSR_CHECKSUMS,
    RELAYVSR_MODELS,
    RELAYVSR_REPO,
    MOSS_CLIP,
    MOSS_REPO,
    MOSS_UNET,
    MOSS_VAE,
    MUSIC_CLIP,
    MUSIC_REPO,
    MUSIC_UNET,
    MUSIC_VAE,
    SEEDVR2_REPO,
    SEEDVR2_UNET,
    SEEDVR2_VAE,
    UNETS,
    VIDEO_VAE,
)

from image_models import IMAGE_FILES, IMAGE_CHECKSUMS
from audio_catalog import AUDIO_GROUPS, AUDIO_CHECKSUMS, audio_manifest
from model_download import download, verify_checksum
from prepare_audio_runtime import install_runtime

COMFY_REPO = "Comfy-Org/MiniMax-H3"
LARRYVRH_NODE = "custom_nodes/ComfyUI-MiniMax-H3-Turbo"
GROUPS = (
    "shared",
    "fl2va",
    "ref2va",
    "larryvrh",
    "pika",
    "fasth3",
    "image",
    "music",
    "upscale",
    "flashvsr",
    "relayvsr",
    "pixal3d",
    "klein",
    "qwen-image",
)

GROUPS = (*GROUPS, *AUDIO_GROUPS)
# Pinned hashes of the LoRAs that publish one (verified after download and on existing files).
LORA_CHECKSUMS = {v["file"]: v["sha256"] for v in LORAS.values() if v.get("sha256")}
CHECKSUMS = IMAGE_CHECKSUMS | AUDIO_CHECKSUMS | LORA_CHECKSUMS | RELAYVSR_CHECKSUMS


def manifest():
    """(group, repo, path in repo, models subfolder, file name) for every checkpoint."""
    items = [
        ("shared", COMFY_REPO, f"text_encoders/{CLIP}", "text_encoders", CLIP),
        ("shared", COMFY_REPO, f"vae/{VIDEO_VAE}", "vae", VIDEO_VAE),
        ("shared", COMFY_REPO, f"vae/{AUDIO_VAE}", "vae", AUDIO_VAE),
    ]
    items += [
        (family, COMFY_REPO, f"diffusion_models/{name}", "diffusion_models", name)
        for family, name in UNETS.items()
    ]
    for lora in LORAS.values():
        if lora.get("unet"):  # FastH3: its own base, no LoRA
            items.append(
                (
                    "fasth3",
                    lora["unet_source"],
                    f"diffusion_models/{lora['unet']}",
                    "diffusion_models",
                    lora["unet"],
                )
            )
        if lora["file"]:
            group = "larryvrh" if lora["loader"] == "larryvrh" else lora["group"]
            path = lora.get("source_path", lora["file"])
            items.append((group, lora["source"], path, "loras", lora["file"]))
    items += [
        (
            "image",
            MOSS_REPO,
            f"diffusion_models/{MOSS_UNET}",
            "diffusion_models",
            MOSS_UNET,
        ),
        ("image", MOSS_REPO, f"text_encoders/{MOSS_CLIP}", "text_encoders", MOSS_CLIP),
        ("image", MOSS_REPO, f"vae/{MOSS_VAE}", "vae", MOSS_VAE),
        (
            "music",
            MUSIC_REPO,
            f"diffusion_models/{MUSIC_UNET}",
            "diffusion_models",
            MUSIC_UNET,
        ),
        (
            "music",
            MUSIC_REPO,
            f"text_encoders/{MUSIC_CLIP}",
            "text_encoders",
            MUSIC_CLIP,
        ),
        ("music", MUSIC_REPO, f"vae/{MUSIC_VAE}", "vae", MUSIC_VAE),
        (
            "upscale",
            SEEDVR2_REPO,
            f"diffusion_models/{SEEDVR2_UNET}",
            "diffusion_models",
            SEEDVR2_UNET,
        ),
        ("upscale", SEEDVR2_REPO, f"vae/{SEEDVR2_VAE}", "vae", SEEDVR2_VAE),
    ]
    items += [
        ("flashvsr", FLASHVSR_REPO, name, FLASHVSR_DIR, name) for name in FLASHVSR_FILES
    ]
    for files in RELAYVSR_MODELS.values():
        items += [
            ("relayvsr", RELAYVSR_REPO, f"diffusion_models/{files['unet']}", "diffusion_models", files["unet"]),
            ("relayvsr", RELAYVSR_REPO, f"vae/{files['vae']}", "vae", files["vae"]),
        ]
    items += [
        ("pixal3d", repo, f"{sub}/{name}", sub, name) for repo, sub, name in PIXAL_FILES
    ]
    items += IMAGE_FILES
    # A LoRA or FastH3 base belongs to its family too, so skipping "ref2va" also skips the ref2v ones.
    families = {
        lora[k]: lora["family"]
        for lora in LORAS.values()
        for k in ("file", "unet")
        if lora.get(k)
    }
    seen = set()
    items = [it for it in items if (it[3], it[4]) not in seen and not seen.add((it[3], it[4]))]
    return [
        (
            g,
            repo,
            path,
            sub,
            name,
            ("image" if g in ("klein", "qwen-image") else families.get(name)),
        )
        for g, repo, path, sub, name in items
    ] + audio_manifest()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--comfyui",
        default=os.environ.get("H3_COMFYUI"),
        help="ComfyUI checkout (default: $H3_COMFYUI)",
    )
    p.add_argument("--audio-cpp", default=os.environ.get("H3_AUDIOCPP_DIR"), help="audio.cpp runtime/model directory (default: sibling of ComfyUI)")
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=GROUPS,
        metavar="GROUP",
        help=f"groups to leave out: {', '.join(GROUPS)}",
    )
    p.add_argument(
        "--connections",
        type=int,
        choices=range(1, 9),
        default=1,
        help="connections per file (2-8 requires aria2c)",
    )
    p.add_argument(
        "--workers",
        type=int,
        choices=range(1, 9),
        default=1,
        help="parallel downloads (1-8, default 1)",
    )
    p.add_argument(
        "--only", nargs="+", choices=GROUPS, help="download only these groups"
    )
    p.add_argument("--list", action="store_true", help="print the files and exit")
    args = p.parse_args()

    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    items = [
        it
        for it in manifest()
        if it[0] not in args.skip
        and it[5] not in args.skip
        and (not args.only or it[0] in args.only or it[5] in args.only)
    ]

    if args.list:
        for group, repo, path, sub, name, family in items:
            prefix = ("audio_cpp/" if group == "audio-runtime" else "audio_cpp/models/") if family == "audio" else "models/"
            print(f"{group:12} {prefix}{sub}/{name}  <-  {repo}{'/' + path if path else ''}")
        return
    needs_comfy = any(it[5] != "audio" for it in items)
    if needs_comfy and not args.comfyui:
        p.error("--comfyui (or $H3_COMFYUI) is required for ComfyUI model groups")
    if not args.audio_cpp and not args.comfyui:
        p.error("--audio-cpp is required without --comfyui")
    audio_root = Path(args.audio_cpp).expanduser() if args.audio_cpp else Path(args.comfyui).expanduser().parent / "audio_cpp"
    models = Path(args.comfyui).expanduser() / "models" if args.comfyui else None
    if needs_comfy and not models.is_dir():
        p.error(f"{models} does not exist; point --comfyui at a ComfyUI checkout")

    def fetch(item):
        i, (group, repo, path, sub, name, family) = item
        base = (audio_root if group == "audio-runtime" else audio_root / "models") if family == "audio" else models
        dest = base / sub / name
        try:
            if dest.exists():
                verify_checksum(dest, CHECKSUMS.get(name))
                if group == "audio-runtime":
                    install_runtime(audio_root)
                print(f"[{i}/{len(items)}] ok      {sub}/{name}")
                return None
            print(f"[{i}/{len(items)}] fetch   {sub}/{name}  ({repo})")
            dest.parent.mkdir(parents=True, exist_ok=True)
            download(
                (repo if repo.startswith("https://") else f"{endpoint}/{repo}/resolve/main/{path}"),
                dest,
                CHECKSUMS.get(name),
                args.connections,
            )
            if group == "audio-runtime":
                install_runtime(audio_root)
        except Exception as e:
            print(f"\n    failed {name}: {e}")
            return name
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failed = [name for name in pool.map(fetch, enumerate(items, 1)) if name]

    if (
        (not args.only or "larryvrh" in args.only)
        and "larryvrh" not in args.skip
        and not (Path(args.comfyui).expanduser() / LARRYVRH_NODE).is_dir()
    ):
        print(
            f"\nnote: the larryvrh LoRA also needs its ComfyUI node:\n"
            f"    git clone https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo {args.comfyui}/{LARRYVRH_NODE}"
        )
    if (
        (not args.only or "flashvsr" in args.only)
        and "flashvsr" not in args.skip
        and not (
            Path(args.comfyui).expanduser() / "custom_nodes" / FLASHVSR_NODE
        ).is_dir()
    ):
        print(
            f"\nnote: FlashVSR upscaling also needs its ComfyUI node:\n"
            f"    git clone https://github.com/T8mars/{FLASHVSR_NODE} {args.comfyui}/custom_nodes/{FLASHVSR_NODE}"
        )
    if failed:
        print(
            f"\n{len(failed)} download(s) failed: {', '.join(failed)}; rerun to resume."
        )
        sys.exit(1)
    if any(it[0] == "audio-runtime" for it in items):
        install_runtime(audio_root)
    print("\nall models ready")


if __name__ == "__main__":
    main()
