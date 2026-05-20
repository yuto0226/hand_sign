"""
Capture hand gesture data using MediaPipe and save as NPZ files.

Usage:
    python record.py <category>
    r     — start / stop continuous recording
    space — single snapshot
    q     — quit
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from model import extract_features
from utils import MODEL_PATH, download_model, draw_landmarks


def main():
    parser = argparse.ArgumentParser(description="Record hand sign samples")
    parser.add_argument("category", type=str, help="Gesture label (e.g. rock, paper)")
    args = parser.parse_args()

    category = args.category
    out_dir = Path("data/npz") / category
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_id = 0
    recording = False

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

    def save_sample(landmarks) -> Path:
        nonlocal frame_id
        features = extract_features(landmarks)
        path = out_dir / f"{category}_{session_id}_{frame_id:06d}.npz"
        np.savez_compressed(path, features=features)
        frame_id += 1
        return path

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] can't open camera")
        return

    print("r: start/stop recording  space: snapshot  q: quit")

    t0 = time.monotonic()
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int((time.monotonic() - t0) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                landmarks = result.hand_landmarks[0] if result.hand_landmarks else None

                if landmarks:
                    draw_landmarks(frame, landmarks)

                if recording:
                    cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)
                    if landmarks:
                        save_sample(landmarks)

                cv2.imshow("record", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("r"):
                    recording = not recording
                    if recording:
                        print(f"[+] recording started → {out_dir}/")
                    else:
                        print(f"[!] recording stopped, saved {frame_id} frames total")
                elif key == ord(" "):
                    if landmarks:
                        path = save_sample(landmarks)
                        print(f"[+] snapshot → {path}")
                    else:
                        print("[!] no hand detected")
                elif key == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
