"""Statistical utilities for confidence intervals."""

import numpy as np
from typing import List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    data: List[float],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    statistic_fn: Callable = np.mean,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Non-parametric bootstrap CI (Efron & Tibshirani, 1993) to provide
    error bars on metrics for publication-quality results. This technique
    resamples the data with replacement to estimate the sampling distribution
    without assuming a specific parametric form.

    Reference: Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
    """
    if not data:
        return 0.0, 0.0, 0.0

    data = np.array(data)
    n = len(data)

    point_estimate = statistic_fn(data)

    rng = np.random.default_rng(seed)
    bootstrap_stats = []
    for _ in range(n_iterations):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)

    logger.info(
        f"Bootstrap CI ({confidence_level*100}%): "
        f"{point_estimate:.3f} [{lower_bound:.3f}, {upper_bound:.3f}]"
    )

    return point_estimate, lower_bound, upper_bound


def cluster_bootstrap_confidence_interval(
    values: List[float],
    clusters: List[str],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    statistic_fn: Callable = np.mean,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Compute cluster bootstrap CI by resampling cluster IDs with replacement.

    Args:
        values: Per-observation metric values.
        clusters: Cluster IDs aligned to `values` (e.g., persona_id).
    """
    if not values or not clusters or len(values) != len(clusters):
        return 0.0, 0.0, 0.0

    values_np = np.array(values, dtype=float)
    clusters_np = np.array(clusters, dtype=object)
    unique_clusters = np.unique(clusters_np)
    if len(unique_clusters) == 0:
        return 0.0, 0.0, 0.0

    point_estimate = float(statistic_fn(values_np))
    bootstrap_stats = []

    rng = np.random.default_rng(seed)
    for _ in range(n_iterations):
        sampled_clusters = rng.choice(
            unique_clusters, size=len(unique_clusters), replace=True
        )
        sampled_indices = np.concatenate(
            [np.where(clusters_np == cluster_id)[0] for cluster_id in sampled_clusters]
        )
        bootstrap_stats.append(float(statistic_fn(values_np[sampled_indices])))

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    lower_bound = float(np.percentile(bootstrap_stats, lower_percentile))
    upper_bound = float(np.percentile(bootstrap_stats, upper_percentile))

    return point_estimate, lower_bound, upper_bound
