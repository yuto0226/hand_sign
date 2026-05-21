"""YOLO pose hand-region experiment — uses wrist keypoints to locate hands.

Uses body pose to find wrist positions, then draws a bounding box around
the hand region. Wrists are robust to finger occlusion because they sit
at the base of hand complexity.

Usage:
    python scripts/detect_yolo.py [--model yolo11n-pose.pt] [--camera 0]
    q — quit

COCO-17 wrist indices: 9 = left_wrist, 10 = right_wrist
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

SAGE = (55, 145, 80)  # deep green  — bbox / detected
CORAL = (60, 50, 170)  # deep red    — no detection
SILVER = (210, 210, 210)  # light gray  — FPS text
WHITE = (245, 245, 240)  # near-white  — wrist dots

_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_WRIST_CONF_THRESH = 0.5
_BBOX_PAD = 0.15  # padding relative to wrist-to-wrist distance


def _hand_bbox(
    kpts: np.ndarray, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    lw, rw = kpts[_LEFT_WRIST], kpts[_RIGHT_WRIST]
    lw_ok, rw_ok = lw[2] >= _WRIST_CONF_THRESH, rw[2] >= _WRIST_CONF_THRESH
    if not lw_ok and not rw_ok:
        return None

    if lw_ok and rw_ok:
        min_x, max_x = min(lw[0], rw[0]), max(lw[0], rw[0])
        min_y, max_y = min(lw[1], rw[1]), max(lw[1], rw[1])
        span = max(max_x - min_x, max_y - min_y, 80)
    else:
        pt = lw if lw_ok else rw
        min_x = max_x = pt[0]
        min_y = max_y = pt[1]
        span = 80

    pad = span * _BBOX_PAD
    return (
        int(max(0, min_x - pad)),
        int(max(0, min_y - pad)),
        int(min(img_w, max_x + pad)),
        int(min(img_h, max_y + pad)),
    )


def _draw_status(frame, n_persons: int, n_hands_bbox: int, fps: float) -> None:
    cv2.putText(
        frame,
        f"persons: {n_persons}  wrists: {n_hands_bbox}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        SAGE if n_hands_bbox else CORAL,
        2,
    )
    cv2.putText(
        frame, f"{fps:.0f} fps", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SILVER, 1
    )


def _open_camera(idx: int):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _capture_loop(model: YOLO, cap) -> None:
    prev_t = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.monotonic()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        h, w = frame.shape[:2]
        results = model(frame, verbose=False)

        n_persons = 0
        n_hands_bbox = 0

        for r in results:
            if r.keypoints is None:
                continue
            kpts_all = r.keypoints.data.cpu().numpy()  # (N, 17, 3)
            n_persons += len(kpts_all)

            for kpts in kpts_all:
                for wrist_idx in (_LEFT_WRIST, _RIGHT_WRIST):
                    x, y, conf = kpts[wrist_idx]
                    if conf >= _WRIST_CONF_THRESH:
                        cv2.circle(frame, (int(x), int(y)), 8, WHITE, -1)

                bbox = _hand_bbox(kpts, w, h)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), SAGE, 2)
                    n_hands_bbox += 1

        _draw_status(frame, n_persons, n_hands_bbox, fps)
        cv2.imshow("YOLO pose — hand region", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="YOLO model weight (downloads automatically if not present)",
    )
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    print(f"[*] loading {args.model} ...")
    model = YOLO(args.model)

    cap = None
    try:
        cap = _open_camera(args.camera)
        if cap is None:
            print("[!] cannot open camera")
            return
        _capture_loop(model, cap)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
