"""Real-time Naruto jutsu recognition — CNN edition.

Usage:
    python naruto.py --checkpoint best_cnn.pth [--camera 0] [--threshold 0.4]
    q — quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from cnnclassifier.evaluate import load_checkpoint
from cnnclassifier.model import build_model
from jutsu import JutsuFSM, SignFilter, draw_jutsu
from utils import MODEL_PATH, download_model

_PREPROCESS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

WHITE = (245, 245, 240)
SAGE = (55, 145, 80)

_HINT_SIZE = 160


def _load_hint(sign: str):
    img_path = Path(__file__).parent / "images" / f"{sign}.jpg"
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


def _union_bbox(lm_lists, w: int, h: int, pad: float = 0.05):
    all_bboxes = []
    for lms in lm_lists:
        xs, ys = zip(*((lm.x, lm.y) for lm in lms))
        x1 = int(max(0.0, min(xs) - pad) * w)
        y1 = int(max(0.0, min(ys) - pad) * h)
        x2 = int(min(1.0, max(xs) + pad) * w)
        y2 = int(min(1.0, max(ys) + pad) * h)
        all_bboxes.append((x1, y1, x2, y2))
    return (
        min(b[0] for b in all_bboxes),
        min(b[1] for b in all_bboxes),
        max(b[2] for b in all_bboxes),
        max(b[3] for b in all_bboxes),
    )


def draw_overlay(
    frame, classes: list[str], probs: list[float], pred_idx: int, conf: float
) -> None:
    is_unknown = pred_idx == -1
    pred_cls = "unknown" if is_unknown else classes[pred_idx]

    roi = frame[8:58, 8:210]
    black = np.zeros_like(roi)
    frame[8:58, 8:210] = cv2.addWeighted(roi, 0.55, black, 0.45, 0)
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
        y_top = 65 + i * 22
        bar_w = int(p * 170)
        color = (0, 220, 120) if (not is_unknown and i == pred_idx) else (100, 100, 100)
        cv2.rectangle(frame, (10, y_top), (10 + bar_w, y_top + 16), color, -1)
        cv2.putText(
            frame,
            f"{cls}: {p:.2f}",
            (10 + bar_w + 8, y_top + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (225, 225, 225),
            1,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_cnn.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--hold", type=float, default=100)
    parser.add_argument("--gap", type=float, default=5000)
    parser.add_argument("--cooldown", type=float, default=5000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}")

    state_dict, classes = load_checkpoint(args.checkpoint, device)
    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    hints = {cls: _load_hint(cls) for cls in classes}

    sign_filter = SignFilter(hold_ms=args.hold)
    last_fired: tuple[str, float] | None = None
    last_sign_at: float = 0.0

    def on_jutsu(name: str) -> None:
        nonlocal last_fired
        last_fired = (name, time.monotonic())
        print(f"[!] {name}")

    fsm = JutsuFSM(on_jutsu=on_jutsu, gap_ms=args.gap, cooldown_ms=args.cooldown)

    download_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[!] can't open camera")
        return

    t0 = time.monotonic()
    print("q: quit")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                now = time.monotonic()
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int((now - t0) * 1000),
                )

                sign = None
                pred_idx = -1
                if result.hand_landmarks:
                    for lms in result.hand_landmarks:
                        pass  # draw_landmarks(frame, lms)

                    ux1, uy1, ux2, uy2 = _union_bbox(result.hand_landmarks, w, h)
                    cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), SAGE, 1)
                    roi = frame[uy1:uy2, ux1:ux2]

                    if roi.size > 0:
                        x = torch.as_tensor(_PREPROCESS(roi)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            probs_t = F.softmax(model(x), dim=1).squeeze(0)
                        raw_idx = int(probs_t.argmax())
                        conf = probs_t[raw_idx].item()
                        pred_idx = raw_idx if conf >= args.threshold else -1
                        draw_overlay(frame, classes, probs_t.tolist(), pred_idx, conf)
                        sign = sign_filter.update(pred_idx, classes, now)
                else:
                    sign = sign_filter.update(-1, classes, now)

                if sign is not None:
                    fsm.feed(sign, now)
                    last_sign_at = now
                elif now - last_sign_at > args.gap / 1000:
                    fsm.reset()
                    sign_filter.reset()
                    last_sign_at = now

                if pred_idx != -1:
                    hint = hints.get(classes[pred_idx])
                    if hint is not None:
                        _draw_hint(frame, hint)

                draw_jutsu(frame, fsm, last_fired, now)
                cv2.imshow("naruto", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    fsm.reset()
                    sign_filter.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
