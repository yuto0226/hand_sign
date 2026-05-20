from jutsu import SignFilter


def test_sign_filter_emits_after_hold():
    sf = SignFilter(hold_ms=500)
    classes = ["tora", "i"]
    assert sf.update(0, 0.9, classes, now=0.0) is None
    assert sf.update(0, 0.9, classes, now=0.3) is None
    assert sf.update(0, 0.9, classes, now=0.5) == "tora"


def test_sign_filter_no_emit_before_hold():
    sf = SignFilter(hold_ms=500)
    classes = ["tora"]
    sf.update(0, 0.9, classes, now=0.0)
    assert sf.update(0, 0.9, classes, now=0.499) is None


def test_sign_filter_resets_on_sign_change():
    sf = SignFilter(hold_ms=500)
    classes = ["tora", "i"]
    sf.update(0, 0.9, classes, now=0.0)
    sf.update(0, 0.9, classes, now=0.4)
    sf.update(1, 0.9, classes, now=0.41)
    assert sf.update(1, 0.9, classes, now=0.8) is None
    assert sf.update(1, 0.9, classes, now=0.91) == "i"


def test_sign_filter_resets_on_unknown():
    sf = SignFilter(hold_ms=500)
    classes = ["tora"]
    sf.update(0, 0.9, classes, now=0.0)
    sf.update(-1, 0.0, classes, now=0.4)
    assert sf.update(0, 0.9, classes, now=0.5) is None
    assert sf.update(0, 0.9, classes, now=1.0) == "tora"


def test_sign_filter_emits_only_once():
    sf = SignFilter(hold_ms=500)
    classes = ["tora"]
    sf.update(0, 0.9, classes, now=0.0)
    assert sf.update(0, 0.9, classes, now=0.5) == "tora"
    assert sf.update(0, 0.9, classes, now=0.6) is None
