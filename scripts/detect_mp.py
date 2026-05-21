"""MediaPipe hand detection experiment — shows bbox derived from landmarks.

Usage:
    python scripts/detect_mp.py
    q — quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import MODEL_PATH, download_model, draw_landmarks

SAGE = (55, 145, 80)  # deep green  — bbox / detected
CORAL = (60, 50, 170)  # deep red    — no detection
SILVER = (210, 210, 210)  # light gray  — FPS text
WHITE = (245, 245, 240)  # near-white  — landmarks


def _bbox_from_landmarks(lm_list, w: int, h: int) -> tuple[int, int, int, int]:
    pad = 0.05
    xs, ys = zip(*((lm.x, lm.y) for lm in lm_list))
    x1 = int(max(0.0, min(xs) - pad) * w)
    y1 = int(max(0.0, min(ys) - pad) * h)
    x2 = int(min(1.0, max(xs) + pad) * w)
    y2 = int(min(1.0, max(ys) + pad) * h)
    return x1, y1, x2, y2


def _draw_status(frame, n_hands: int, fps: float) -> None:
    color = SAGE if n_hands else CORAL
    label = f"hands: {n_hands}" if n_hands else "no hands"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(
        frame, f"{fps:.0f} fps", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SILVER, 1
    )


def _open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _capture_loop(landmarker, cap) -> None:
    t0 = time.monotonic()
    prev_t = t0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_buf = np.empty((h, w, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.monotonic()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        timestamp_ms = int((now - t0) * 1000)
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=rgb_buf)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_buf), timestamp_ms
        )

        for lms in result.hand_landmarks:
            draw_landmarks(frame, lms)
            x1, y1, x2, y2 = _bbox_from_landmarks(lms, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), SAGE, 2)

        _draw_status(frame, len(result.hand_landmarks), fps)
        cv2.imshow("MediaPipe — hand detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    download_model()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = None
    try:
        cap = _open_camera()
        if cap is None:
            print("[!] cannot open camera")
            return

        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            _capture_loop(landmarker, cap)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
