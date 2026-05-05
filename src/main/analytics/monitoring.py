from __future__ import annotations

from collections import Counter
import math


class LatencyTracker:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, seconds: float) -> None:
        self.values.append(seconds)

    def summary(self) -> dict[str, float]:
        if not self.values:
            return {"count": 0.0, "mean_ms": 0.0, "p95_ms": 0.0}

        sorted_vals = sorted(self.values)
        count = len(sorted_vals)
        mean_ms = (sum(sorted_vals) / count) * 1000.0
        p95_index = min(count - 1, int(0.95 * count))
        p95_ms = sorted_vals[p95_index] * 1000.0
        return {"count": float(count), "mean_ms": mean_ms, "p95_ms": p95_ms}


def _to_distribution(items: list[str]) -> dict[str, float]:
    if not items:
        return {}
    counts = Counter(items)
    n = sum(counts.values())
    return {k: v / n for k, v in counts.items()}


def js_divergence(reference: list[str], current: list[str]) -> float:
    p = _to_distribution(reference)
    q = _to_distribution(current)
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    m = {k: 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0) for k in keys}

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for k in keys:
            if a.get(k, 0.0) > 0 and b.get(k, 0.0) > 0:
                total += a[k] * math.log(a[k] / b[k], 2)
        return total

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
