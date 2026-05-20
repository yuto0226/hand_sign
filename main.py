"""hand landmark visualization for debugging — press q: quit, space: log"""

import time

import cv2
import mediapipe as mp

from model import FINGER_JOINTS, FINGER_LANDMARK_IDXS, FINGER_NAMES
from utils import MODEL_PATH, angle_at, download_model, draw_landmarks


def snapshot(result):
    if not result.hand_landmarks:
        print("[!] no hands detected")
        return

    for hand_idx, lm in enumerate(result.hand_landmarks):
        wrist = lm[0]
        print(f"\n── hand {hand_idx} ──────────────")
        print(f"  {'wrist':<6}  ({wrist.x:.3f}, {wrist.y:.3f}, {wrist.z:.3f})")

        for name, idxs, joints in zip(
            FINGER_NAMES, FINGER_LANDMARK_IDXS, FINGER_JOINTS
        ):
            coords = " ".join(
                f"({lm[i].x:.3f},{lm[i].y:.3f},{lm[i].z:.3f})" for i in idxs
            )
            angles = " ".join(
                f"{angle_at(lm[a], lm[b], lm[c]):3.0f}°" for a, b, c in joints
            )
            print(f"  {name:<6}  {coords}   {angles}")


def _open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cap if cap.isOpened() else None


def _capture_loop(landmarker, cap):
    t0 = time.monotonic()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((time.monotonic() - t0) * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        for landmarks in result.hand_landmarks:
            draw_landmarks(frame, landmarks)

        cv2.imshow("Hands Sign", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            snapshot(result)


def main():
    download_model()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
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
