from anpr.training.metrics import character_accuracy, full_plate_accuracy


def test_metrics() -> None:
    pred = ["AB12", "XYZ9"]
    truth = ["AB12", "XY99"]
    assert full_plate_accuracy(pred, truth) == 0.5
    assert 0.0 <= character_accuracy(pred, truth) <= 1.0
