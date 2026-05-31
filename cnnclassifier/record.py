"""Record hand-sign images for the CNN classifier.

Usage:
    python -m cnnclassifier.record --sign tora --session s1
    SPACE — save one photo now
    r     — toggle continuous recording on/off
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


_HINT_SIZE = 160  # px, square


def _load_hint(sign: str):
    img_path = Path(__file__).parent.parent / "images" / f"{sign}.jpg"
    if not img_path.exists():
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return cv2.resize(img, (_HINT_SIZE, _HINT_SIZE))


def _draw_hint(frame, hint) -> None:
    h, w = frame.shape[:2]
    pad = 8
    y1 = h - _HINT_SIZE - pad
    x1 = w - _HINT_SIZE - pad
    roi = frame[y1 : y1 + _HINT_SIZE, x1 : x1 + _HINT_SIZE]
    blended = cv2.addWeighted(roi, 0.3, hint, 0.7, 0)
    frame[y1 : y1 + _HINT_SIZE, x1 : x1 + _HINT_SIZE] = blended
    cv2.rectangle(frame, (x1, y1), (x1 + _HINT_SIZE, y1 + _HINT_SIZE), WHITE, 1)


def _draw_countdown(frame, n: int) -> None:
    h, w = frame.shape[:2]
    text = str(n)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5.0
    thickness = 10
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, WHITE, thickness, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Get ready",
        (x, y + th + baseline + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        WHITE,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign", required=True, choices=_SIGNS + ["unknown"])
    parser.add_argument("--session", default="s1")
    parser.add_argument("--every", type=int, default=10, help="save every N frames")
    parser.add_argument("--out", default="data/cnn")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.sign
    out_dir.mkdir(parents=True, exist_ok=True)

    hint = _load_hint(args.sign)
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

    countdown_active = False
    countdown_t0 = 0.0
    countdown_seconds = 3

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

                if recording:
                    cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)

                if countdown_active:
                    elapsed = now - countdown_t0
                    n = countdown_seconds - int(elapsed)
                    if n > 0:
                        _draw_countdown(frame, n)
                    else:
                        # Save the *current* ROI after the countdown finishes.
                        if roi is not None:
                            path = out_dir / f"{args.session}_{counter:04d}.jpg"
                            cv2.imwrite(str(path), roi)
                            counter += 1
                        countdown_active = False

                kanji = _KANJI.get(args.sign, "")
                label = f"{args.sign} {kanji}  n={counter}".strip()
                cv2.putText(
                    frame, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1
                )
                if hint is not None:
                    _draw_hint(frame, hint)
                cv2.imshow("record", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    recording = not recording
                elif key == ord(" "):
                    if not countdown_active:
                        countdown_active = True
                        countdown_t0 = now

                if (
                    (not countdown_active)
                    and recording
                    and roi is not None
                    and frame_n % args.every == 0
                ):
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
