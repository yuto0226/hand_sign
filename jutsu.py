from __future__ import annotations


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
    "影分身の術": ["寅"],
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
        conf: float,
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
