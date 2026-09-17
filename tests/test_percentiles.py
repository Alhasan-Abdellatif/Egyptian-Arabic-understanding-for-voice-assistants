from lahja.eval.metrics import percentiles


def test_empty():
    assert percentiles([]) == {}


def test_sorts_input_and_picks_expected_positions():
    xs = [float(i) for i in range(1, 101)]
    p = percentiles(list(reversed(xs)))  # unsorted on purpose
    assert p["n"] == 100
    assert (p["p50"], p["p90"], p["p95"], p["p99"]) == (51.0, 91.0, 96.0, 100.0)
    assert p["mean"] == 50.5
    assert p["max"] == 100.0


def test_small_sample_p99_is_just_the_max():
    """With a handful of samples the tail percentiles carry no information."""
    p = percentiles([0.3, 0.1, 0.2])
    assert p["p99"] == p["max"] == 0.3
    assert p["p50"] == 0.2
