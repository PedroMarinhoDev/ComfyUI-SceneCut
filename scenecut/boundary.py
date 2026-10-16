"""Where exactly does one shot end and the next begin?

Given frames around a cut and how much each looks like the shot before (`to_before`) and the shot
after (`to_after`), the cut goes where the frames switch from one group to the other: the position
that leaves the fewest frames on the wrong side, weighted by how wrong they look. Used by the
validator to place cuts beside flashes and by the viewer's "fix cut" button.
"""
from __future__ import annotations

from typing import Optional, Sequence


def split_costs(to_before: Sequence[float], to_after: Sequence[float]) -> list[float]:
    """cost[j]: the cut before frame j of the span (j = 0..n, n meaning after the last frame);
    frames before the cut pay (1 - to_before), frames from it on pay (1 - to_after)."""
    n = len(to_before)
    tail = [0.0] * (n + 1)
    for j in range(n - 1, -1, -1):
        tail[j] = tail[j + 1] + (1.0 - float(to_after[j]))
    costs, head = [], 0.0
    for j in range(n + 1):
        costs.append(head + tail[j])
        if j < n:
            head += 1.0 - float(to_before[j])
    return costs


def best_split(first: int, to_before: Sequence[float], to_after: Sequence[float],
               prefer: Optional[int] = None, margin: float = 0.0) -> tuple[int, list[float]]:
    """The cut frame for a span of frames starting at frame `first`, and the cost of every
    position (index j is frame first + j). `prefer` (usually the current cut) is kept unless
    another position is better by more than `margin`; among equal costs the one nearest `prefer`
    wins."""
    costs = split_costs(to_before, to_after)
    ref = prefer if prefer is not None else first + len(costs) // 2
    best = min(range(len(costs)), key=lambda j: (round(costs[j], 9), abs(first + j - ref)))
    if prefer is not None and first <= prefer <= first + len(costs) - 1:
        if costs[prefer - first] <= costs[best] + margin:
            return prefer, costs
    return first + best, costs
