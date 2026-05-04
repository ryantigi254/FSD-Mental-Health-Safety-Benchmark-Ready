from reliable_clinical_benchmark.utils.stats import (
    bootstrap_confidence_interval,
    cluster_bootstrap_confidence_interval,
)


def test_cluster_bootstrap_confidence_interval_returns_ordered_ci():
    values = [0.8, 0.7, 0.4, 0.5, 0.9, 0.85]
    clusters = ["p1", "p1", "p2", "p2", "p3", "p3"]

    point, low, high = cluster_bootstrap_confidence_interval(
        values, clusters, n_iterations=200
    )

    assert 0.0 <= point <= 1.0
    assert low <= point <= high


def test_cluster_bootstrap_handles_invalid_input():
    point, low, high = cluster_bootstrap_confidence_interval([], [])
    assert (point, low, high) == (0.0, 0.0, 0.0)


def test_bootstrap_confidence_interval_is_deterministic_by_default():
    data = [1.0, 0.0, 1.0, -1.0, 0.0, 1.0]

    first = bootstrap_confidence_interval(data, n_iterations=200)
    second = bootstrap_confidence_interval(data, n_iterations=200)

    assert first == second
