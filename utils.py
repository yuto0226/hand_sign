from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def download_model() -> None:
    if not Path(MODEL_PATH).exists():
        print("[*] downloading hand landmarker model ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] download complete")


WHITE = (255, 255, 255)
GRAY = (170, 170, 170)

# fmt:off
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (2, 5), (5, 9), (9, 13), (13, 17),      # Palm
]
# fmt:on

_LANDMARK_LABELS = tuple(str(i) for i in range(21))


def angle_at(a, b, c) -> float:
    a, b, c = (
        np.array([a.x, a.y, a.z]),
        np.array([b.x, b.y, b.z]),
        np.array([c.x, c.y, c.z]),
    )
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], GRAY, 2)

    for i, (x, y) in enumerate(points):
        cv2.circle(frame, (x, y), 4, WHITE, -1)
        cv2.putText(
            frame,
            _LANDMARK_LABELS[i],
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            WHITE,
            1,
        )
