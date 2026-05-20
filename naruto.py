"""
Real-time Naruto jutsu recognition demo.

Usage:
    python naruto.py --checkpoint best.pth [--camera 0] [--hold 500] [--gap 3000]
    q — quit
"""

from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

from evaluate import load_checkpoint
from jutsu import JutsuFSM, SignFilter, draw_jutsu
from model import HandSignTransformer, extract_features
from utils import MODEL_PATH, WHITE, download_model, draw_landmarks


def draw_overlay(
    frame, classes: list[str], probs: list[float], pred_idx: int, conf: float
) -> None:
    is_unknown = pred_idx == -1
    pred_cls = "unknown" if is_unknown else classes[pred_idx]
    h = frame.shape[0]

    roi = frame[8:58, 8:420]
    black = np.zeros_like(roi)
    frame[8:58, 8:420] = cv2.addWeighted(roi, 0.55, black, 0.45, 0)
    cv2.putText(
        frame,
        f"{pred_cls}  {conf:.1%}",
        (16, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        WHITE,
        2,
    )

    for i, (cls, p) in enumerate(zip(classes, probs)):
        y = h - 16 - i * 28
        bar_w = int(p * 170)
        color = (0, 220, 120) if (not is_unknown and i == pred_idx) else (100, 100, 100)
        cv2.rectangle(frame, (10, y - 16), (10 + bar_w, y), color, -1)
        cv2.putText(
            frame,
            f"{cls}: {p:.2f}",
            (10 + bar_w + 8, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (225, 225, 225),
            1,
        )


def main():
    parser = argparse.ArgumentParser(description="Naruto jutsu recognition demo")
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="max softmax prob below this → unknown",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=500,
        help="ms a sign must be held to confirm (default: 500)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=3000,
        help="ms allowed between consecutive signs before sequence resets (default: 3000)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    state_dict, ckpt_meta = load_checkpoint(args.checkpoint, device)
    classes = ckpt_meta["classes"]

    model = HandSignTransformer(
        num_classes=len(classes),
        d_model=ckpt_meta["d_model"],
        nhead=ckpt_meta["nhead"],
        num_layers=ckpt_meta["num_layers"],
        dim_feedforward=ckpt_meta["dim_feedforward"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    sign_filter = SignFilter(hold_ms=args.hold)
    last_fired: tuple[str, float] | None = None

    def on_jutsu(name: str) -> None:
        nonlocal last_fired
        last_fired = (name, time.monotonic())
        print(f"[!] {name}")

    fsm = JutsuFSM(on_jutsu=on_jutsu, gap_ms=args.gap)

    download_model()

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    t0 = time.monotonic()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[!] can't open camera")
        return

    print("q: quit")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                now = time.monotonic()
                timestamp_ms = int((now - t0) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    draw_landmarks(frame, landmarks)

                    features = extract_features(landmarks)
                    x = torch.from_numpy(features).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(x)
                        probs_t = F.softmax(logits, dim=1).squeeze(0)

                    raw_idx = int(probs_t.argmax())
                    conf = probs_t[raw_idx].item()
                    pred_idx = raw_idx if conf >= args.threshold else -1
                    probs = probs_t.tolist()
                    draw_overlay(frame, classes, probs, pred_idx, conf)

                    sign = sign_filter.update(pred_idx, classes, now)
                else:
                    sign = sign_filter.update(-1, classes, now)

                if sign is not None:
                    fsm.feed(sign, now)

                draw_jutsu(frame, fsm, last_fired, now)

                cv2.imshow("naruto", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
