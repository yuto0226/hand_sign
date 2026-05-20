"""
Convert a directory of class-labelled images to NPZ feature files.

Input layout:
    src/<class>/*.png

Output layout:
    dst/<class>/*.npz

Usage:
    python scripts/convert_image.py --src naruto-hand-sign-dataset/png --dst naruto-hand-sign-dataset/npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import encode_hands
from utils import MODEL_PATH, download_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Directory with <class>/<images>")
    parser.add_argument("--dst", required=True, help="Output directory for .npz files")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"[!] Source not found: {src}")

    pairs: list[tuple[Path, str]] = [
        (img, cls_dir.name)
        for cls_dir in sorted(src.iterdir())
        if cls_dir.is_dir()
        for img in sorted(cls_dir.iterdir())
        if img.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    if not pairs:
        raise SystemExit(f"[!] No images found under {src}")

    download_model()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )

    skipped: list[Path] = []
    saved = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for img_path, cls_name in tqdm(pairs, desc="Converting"):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                skipped.append(img_path)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

            if not result.hand_landmarks:
                skipped.append(img_path)
                continue

            out_dir = dst / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / (img_path.stem + ".npz"),
                features=encode_hands(result.hand_landmarks, result.handedness),
            )
            saved += 1

    classes = sorted({c for _, c in pairs})
    print(f"\n[+] Saved {saved} .npz files across {len(classes)} classes → {dst}/")
    if skipped:
        print(f"[!] Skipped {len(skipped)} images (no hand detected)")
        for p in skipped[:10]:
            print(f"    {p}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
