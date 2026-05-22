"""Record hand-sign images for the CNN classifier.

Usage:
    python -m cnnclassifier.record --sign tora --session s1
    SPACE — toggle recording on/off
    q     — quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import MODEL_PATH, download_model

_SIGNS = [
    "ne",
    "ushi",
    "tora",
    "u",
    "tatsu",
    "mi",
    "uma",
    "hitsuji",
    "saru",
    "tori",
    "inu",
    "i",
]
_KANJI = dict(
    zip(
        _SIGNS,
        ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"],
    )
)

SAGE = (55, 145, 80)
WHITE = (245, 245, 240)


def _union_bbox(lm_lists, w: int, h: int, pad: float = 0.05):
    all_bboxes = []
    for lms in lm_lists:
        xs, ys = zip(*((lm.x, lm.y) for lm in lms))
        x1 = int(max(0.0, min(xs) - pad) * w)
        y1 = int(max(0.0, min(ys) - pad) * h)
        x2 = int(min(1.0, max(xs) + pad) * w)
        y2 = int(min(1.0, max(ys) + pad) * h)
        all_bboxes.append((x1, y1, x2, y2))
    ux1 = min(b[0] for b in all_bboxes)
    uy1 = min(b[1] for b in all_bboxes)
    ux2 = max(b[2] for b in all_bboxes)
    uy2 = max(b[3] for b in all_bboxes)
    return ux1, uy1, ux2, uy2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", required=True, choices=_SIGNS)
    parser.add_argument("--session", default="s1")
    parser.add_argument("--every", type=int, default=3, help="save every N frames")
    parser.add_argument("--out", default="data/cnn")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.sign
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = len(list(out_dir.glob(f"{args.session}_*.jpg")))

    download_model()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[!] cannot open camera")
        return

    t0 = time.monotonic()
    frame_n = 0
    recording = False

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                now = time.monotonic()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int((now - t0) * 1000),
                )

                roi = None
                if result.hand_landmarks:
                    ux1, uy1, ux2, uy2 = _union_bbox(result.hand_landmarks, w, h)
                    cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), SAGE, 2)
                    if ux2 > ux1 and uy2 > uy1:
                        roi = frame[uy1:uy2, ux1:ux2]

                label = f"{'[REC]' if recording else '[PAUSE]'}  {_KANJI[args.sign]}  n={counter}"
                cv2.putText(
                    frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2
                )
                cv2.imshow("record", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    recording = not recording

                if recording and roi is not None and frame_n % args.every == 0:
                    path = out_dir / f"{args.session}_{counter:04d}.jpg"
                    cv2.imwrite(str(path), roi)
                    counter += 1

                frame_n += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"[*] saved {counter} images → {out_dir}")


if __name__ == "__main__":
    main()
