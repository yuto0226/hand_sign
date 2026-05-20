from jutsu import JutsuFSM, SignFilter  # type: ignore[reportAttributeAccessIssue]


def test_sign_filter_emits_after_hold():
    sf = SignFilter(hold_ms=500)
    classes = ["tora", "i"]
    assert sf.update(0, classes, now=0.0) is None
    assert sf.update(0, classes, now=0.3) is None
    assert sf.update(0, classes, now=0.5) == "tora"


def test_sign_filter_no_emit_before_hold():
    sf = SignFilter(hold_ms=500)
    classes = ["tora"]
    sf.update(0, classes, now=0.0)
    assert sf.update(0, classes, now=0.499) is None


def test_sign_filter_resets_on_sign_change():
    sf = SignFilter(hold_ms=500)
    classes = ["tora", "i"]
    sf.update(0, classes, now=0.0)
    sf.update(0, classes, now=0.4)
    sf.update(1, classes, now=0.41)
    assert sf.update(1, classes, now=0.8) is None
    assert sf.update(1, classes, now=0.91) == "i"


def test_sign_filter_resets_on_unknown():
    sf = SignFilter(hold_ms=500)
    classes = ["tora"]
    sf.update(0, classes, now=0.0)
    sf.update(-1, classes, now=0.4)
    assert sf.update(0, classes, now=0.5) is None
    assert sf.update(0, classes, now=1.0) == "tora"


def test_sign_filter_emits_only_once():
    sf = SignFilter(hold_ms=500)
    classes = ["tora"]
    sf.update(0, classes, now=0.0)
    assert sf.update(0, classes, now=0.5) == "tora"
    assert sf.update(0, classes, now=0.6) is None


FIRE = {"火遁・豪火球の術": ["亥", "寅"]}
CLONE = {"影分身の術": ["寅"]}
WALL = {"土遁・土流壁": ["亥", "寅", "戌"]}


def test_fsm_fires_single_sign_jutsu():
    fired = []
    fsm = JutsuFSM(on_jutsu=fired.append, jutsu=CLONE, gap_ms=3000)
    fsm.feed("tora", now=0.0)
    assert fired == ["影分身の術"]


def test_fsm_fires_two_sign_jutsu():
    fired = []
    fsm = JutsuFSM(on_jutsu=fired.append, jutsu=FIRE, gap_ms=3000)
    fsm.feed("i", now=0.0)
    assert fired == []
    fsm.feed("tora", now=0.5)
    assert fired == ["火遁・豪火球の術"]


def test_fsm_resets_on_wrong_sign():
    fired = []
    fsm = JutsuFSM(on_jutsu=fired.append, jutsu=FIRE, gap_ms=3000)
    fsm.feed("tora", now=0.0)
    fsm.feed("i", now=0.5)
    fsm.feed("inu", now=1.0)
    fsm.feed("tora", now=1.5)
    assert fired == []


def test_fsm_gap_timeout_resets_progress():
    fired = []
    fsm = JutsuFSM(on_jutsu=fired.append, jutsu=FIRE, gap_ms=3000)
    fsm.feed("i", now=0.0)
    fsm.feed("tora", now=4.0)
    assert fired == []


def test_fsm_gap_timeout_within_limit_does_not_reset():
    fired = []
    fsm = JutsuFSM(on_jutsu=fired.append, jutsu=FIRE, gap_ms=3000)
    fsm.feed("i", now=0.0)
    fsm.feed("tora", now=2.9)
    assert fired == ["火遁・豪火球の術"]


def test_fsm_resets_step_after_fire():
    fired = []
    fsm = JutsuFSM(on_jutsu=fired.append, jutsu=CLONE, gap_ms=3000)
    fsm.feed("tora", now=0.0)
    assert fired == ["影分身の術"]
    fsm.feed("tora", now=1.0)
    assert fired == ["影分身の術", "影分身の術"]


def test_fsm_leading_jutsu_none_when_no_progress():
    fsm = JutsuFSM(on_jutsu=lambda _: None, jutsu=FIRE, gap_ms=3000)
    assert fsm.leading_jutsu() is None


def test_fsm_leading_jutsu_returns_best():
    fsm = JutsuFSM(on_jutsu=lambda _: None, jutsu={**FIRE, **WALL}, gap_ms=3000)
    fsm.feed("i", now=0.0)
    result = fsm.leading_jutsu()
    assert result is not None
    name, step, total = result
    assert step == 1
    assert name in ("火遁・豪火球の術", "土遁・土流壁")
    assert total == len({**FIRE, **WALL}[name])


def test_fsm_unknown_sign_is_ignored():
    fired = []
    fsm = JutsuFSM(on_jutsu=fired.append, jutsu=FIRE, gap_ms=3000)
    fsm.feed("i", now=0.0)
    fsm.feed("unknown_sign", now=0.5)
    fsm.feed("tora", now=1.0)
    assert fired == ["火遁・豪火球の術"]
