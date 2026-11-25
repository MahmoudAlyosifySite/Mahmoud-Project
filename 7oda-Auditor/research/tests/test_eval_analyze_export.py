"""evaluation.analyze_export on a small simulated export written as JSON Lines."""
import json

import numpy as np

from evaluation import analyze_export
from sim import make_phone, make_population, session_clock, simulate_batch

ROUNDS = (("sitting", 1), ("standing", 2), ("freshGrip", 3))


def _export(path):
    people = make_population(4, seed=5)                  # P01-P03 enroll, P04 is a stranger
    phone = make_phone(3, "lpf")
    persons, hands, postures, sessions, grips, purposes, conditions, starts = [], [], [], [], [], [], [], []

    def add(person, hand, posture, session, grip, purpose, start, condition="handheld"):
        persons.append(person)
        hands.append(hand)
        postures.append(posture)
        sessions.append(session)
        grips.append(grip)
        purposes.append(purpose)
        conditions.append(condition)
        starts.append(start)

    up1, _ = session_clock(phone, 1)
    for k in range(20):                                # the app's 20 on-table patterns (SPEC 8)
        add(None, "right", "unspecified", 1, None, "calibration", up1 + 10 + 2.5 * k, "onTable")
    t = up1 + 100
    for p in people[:3]:
        for hand in ("right", "left"):
            for posture, grip in ROUNDS:
                for k in range(3):
                    add(p, hand, posture, 1, grip, "enrollment", t)
                    t += 2.5
                t += 20
    for s in (2, 3):
        t, _ = session_clock(phone, s)
        for p in people:
            for hand in ("right", "left"):
                for c in range(2):                     # two checks of three patterns each, 30 s apart
                    for k in range(3):
                        add(p, hand, "sitting", s, 10 * c + (hand == "left"), "test", t + 2.4 * k)
                    t += 30
    recs = simulate_batch(persons, hands, list(range(500, 500 + len(persons))), phone, None, postures=postures,
                          sessions=sessions, grips=grips, purposes=purposes, conditions=conditions,
                          seeds=list(range(len(persons))), start_times=starts, include_meta=False)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.write("not json\n")
    return len(recs)


def test_analyze_export_end_to_end(tmp_path):
    src = tmp_path / "recordings.jsonl"
    n = _export(src)
    analyze_export.main([str(src), "--out", str(tmp_path / "out")])
    rep = json.loads((tmp_path / "out" / "recordings_analysis.json").read_text())
    md = (tmp_path / "out" / "recordings_analysis.md").read_text()
    assert rep["n_recordings"] == n and rep["unreadable_lines"] == 1
    assert rep["calibration"] is not None
    assert rep["enrolled"] == ["P01", "P02", "P03"]
    cs = rep["cross_session"]
    assert cs["n_enrolled_checks"] >= 12 and cs["n_stranger_checks"] >= 4     # 2 sessions x 2 hands x 2 checks each
    for k in ("top1_name_acc", "stranger_accept", "false_unknown", "name_and_hand_acc", "misidentified"):
        assert cs[k] is None or 0.0 <= cs[k] <= 1.0
    assert 1.0 <= cs["patterns_per_answer"] <= 3.0
    cm = np.array(rep["confusion"]["counts"])
    assert cm.shape == (4, 5) and cm.sum() == cs["n_enrolled_checks"] + cs["n_stranger_checks"]
    assert "tuned" in rep and rep["tuned"]["test_sessions"]
    assert set(rep["per_session"]) == {"1", "2"}         # per-person test-session numbers (0 = enrollment)
    assert "Confusion matrix" in md and "Cross-session identification" in md


def test_unlabelled_export_reports_no_enrollment(tmp_path):
    phone = make_phone(3, "none")
    recs = simulate_batch(make_population(1, seed=2)[0], "right", [1, 2], phone, None, purposes="check",
                          seeds=[1, 2], include_meta=False)
    for r in recs:
        r["label"] = None
    rep = analyze_export.analyze(recs)
    assert "error" in rep and rep["n_enrolled_people"] == 0
