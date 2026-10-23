from inspect_robots_pai.domains.nero.motion import DIM_LABELS, interpolate, reached


def test_labels_and_step():
    assert DIM_LABELS[0] == "left_x"
    assert DIM_LABELS[9] == "left_gripper"
    assert len(DIM_LABELS) == 20
    current = dict.fromkeys(DIM_LABELS, 0.0)
    rows, steps = interpolate(current, {"left_x": 0.02, "left_r1": 0.1})
    assert steps == 2
    assert abs(rows[-1][0] - 0.02) < 1e-9
    assert reached(rows[-1], {"left_x": 0.02})
