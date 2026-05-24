from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/msmincho.ttc",
]


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for p in _FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise RuntimeError("No CJK font found. Install one of: " + ", ".join(_FONT_PATHS))


SIGN_KANJI: dict[str, str] = {
    "ne": "子",
    "ushi": "丑",
    "tora": "寅",
    "u": "卯",
    "tatsu": "辰",
    "mi": "巳",
    "uma": "午",
    "hitsuji": "未",
    "saru": "申",
    "tori": "酉",
    "inu": "戌",
    "i": "亥",
}

JUTSU: dict[str, list[str]] = {
    "火遁・豪火球の術": ["亥", "寅"],
    # "影分身の術": ["寅"],
    "土遁・土流壁": ["亥", "寅", "戌"],
    "雷遁・千鳥": ["丑", "寅", "戌", "卯", "巳"],
    "風遁・螺旋手裏剣": ["巳", "午", "丑"],
    "水遁・水龍弾の術": ["寅", "丑", "午", "寅", "戌", "寅", "巳", "寅", "未"],
}


class SignFilter:
    """Debounces per-frame classifier output into confirmed sign events.

    Emits a sign (romaji str) when the same sign is held for hold_ms.
    Resets immediately on sign change or unknown prediction.
    """

    def __init__(self, hold_ms: float = 500) -> None:
        self.hold_ms = hold_ms
        self._current: str | None = None
        self._since: float = 0.0

    def update(
        self,
        pred_idx: int,
        classes: list[str],
        now: float,
    ) -> str | None:
        if pred_idx == -1:
            self._current = None
            return None
        sign = classes[pred_idx]
        if sign != self._current:
            self._current = sign
            self._since = now
            return None
        if (now - self._since) * 1000 >= self.hold_ms:
            self._current = None  # reset so it must be re-held
            return sign
        return None

    def reset(self) -> None:
        self._current = None


class JutsuFSM:
    """Matches a stream of confirmed signs against jutsu sequences.

    Tracks per-jutsu progress. Resets a jutsu's progress on wrong sign
    or when gap_ms elapses since the last confirmed sign for that jutsu.
    """

    def __init__(
        self,
        on_jutsu: Callable[[str], None],
        jutsu: dict[str, list[str]] = JUTSU,
        gap_ms: float = 3000,
        cooldown_ms: float = 5000,
    ) -> None:
        self.jutsu = jutsu
        self.gap_ms = gap_ms
        self.cooldown_ms = cooldown_ms
        self.on_jutsu = on_jutsu
        self._step: dict[str, int] = {name: 0 for name in jutsu}
        self._last_at: dict[str, float] = {name: 0.0 for name in jutsu}
        self._fired_at: float = -cooldown_ms  # allow immediate first fire
        self._last_sign: str | None = None

    def feed(self, sign: str, now: float) -> None:
        if sign == self._last_sign:
            return
        self._last_sign = sign
        kanji = SIGN_KANJI.get(sign)
        if kanji is None:
            return
        completed: list[str] = []
        for name, seq in self.jutsu.items():
            if (
                self._step[name] > 0
                and (now - self._last_at[name]) * 1000 > self.gap_ms
            ):
                self._step[name] = 0
            step = self._step[name]
            if kanji == seq[step]:
                self._step[name] += 1
                self._last_at[name] = now
                if self._step[name] == len(seq):
                    self._step[name] = 0
                    completed.append(name)
            else:
                self._step[name] = 0
        if completed and (now - self._fired_at) * 1000 >= self.cooldown_ms:
            winner = max(completed, key=lambda n: len(self.jutsu[n]))
            self._fired_at = now
            self.reset()
            self.on_jutsu(winner)

    def reset(self) -> None:
        for name in self._step:
            self._step[name] = 0
        self._last_sign = None

    def leading_jutsu(self) -> tuple[str, int, int] | None:
        """Return (name, step, total) for the jutsu with the most progress.

        Ties are broken by insertion order of the jutsu dict (first entry wins).
        Returns None if no jutsu has any progress.
        """
        if not self._step:
            return None
        best_name = max(self._step, key=self._step.__getitem__)
        if self._step[best_name] == 0:
            return None
        return (best_name, self._step[best_name], len(self.jutsu[best_name]))


def draw_jutsu(
    frame: np.ndarray,
    fsm: JutsuFSM,
    last_fired: tuple[str, float] | None,
    now: float,
) -> None:
    h, w = frame.shape[:2]
    font_fl = _font(54)
    font_pr = _font(24)

    # Flash: darken background strip first (cv2 blend), then draw text with PIL
    if last_fired is not None:
        name, fired_at = last_fired
        if now - fired_at < 1.0:
            bb = font_fl.getbbox(name)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            tx = (w - tw) // 2
            ty = (h - th) // 2
            x1, x2 = max(tx - 8, 0), min(tx + tw + 8, w)
            y1, y2 = max(ty - 8, 0), min(ty + th + 8, h)
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.addWeighted(
                roi, 0.45, np.zeros_like(roi), 0.55, 0
            )

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    if last_fired is not None:
        name, fired_at = last_fired
        if now - fired_at < 1.0:
            bb = font_fl.getbbox(name)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            tx = (w - tw) // 2
            ty = (h - th) // 2
            draw.text((tx, ty), name, font=font_fl, fill=(255, 255, 255))

    leading = fsm.leading_jutsu()
    if leading is not None:
        jutsu_name, step, total = leading
        seq = fsm.jutsu[jutsu_name]
        x, y = 10, h - 55

        for i, kanji in enumerate(seq):
            done = i < step
            current = i == step
            color = (
                (120, 220, 0) if done else (200, 200, 200) if current else (80, 80, 80)
            )
            label = f"[{kanji}]" if current else kanji
            draw.text((x, y), label, font=font_pr, fill=color)
            bb_label = font_pr.getbbox(label)
            x += (bb_label[2] - bb_label[0]) + 4
            if i < total - 1:
                draw.text((x, y), ">", font=font_pr, fill=(100, 100, 100))
                bb_gt = font_pr.getbbox(">")
                x += (bb_gt[2] - bb_gt[0]) + 4

    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
