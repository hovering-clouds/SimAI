"""
Link compatibility optimization between jobs sharing a link.

For each shared link, Cassini finds the optimal time-shift (rotation) for
each job to minimise link contention.  The compatibility score measures
how well a set of time-shifted demands fit within the link's capacity:

    Excess(α) = max(0, Σ_j bw_j(α − ∆_j) − C_link)
    score     = 1 − Σ_α Excess(α) / (|A| · C_link)

Search strategy:
    - 2 jobs: fix job[0] at 0°, grid-search job[1]'s rotation.
    - 3+ jobs: iterative greedy — fix all but one, search the free axis,
      repeat until convergence.
"""

from dataclasses import dataclass, field

from .circle_abstraction import CircleAbstraction


@dataclass
class CompatibilityResult:
    """Result of link compatibility optimisation.

    Attributes:
        score: Compatibility score in [−inf, 1].  1 = perfect (no excess).
        time_shifts_us: Mapping from circle index (key in the input dict)
            to the optimal time-shift in microseconds.
    """

    score: float
    time_shifts_us: dict[int, int] = field(default_factory=dict)


def _search_angles(step_deg: int) -> list[int]:
    """Return evenly-spaced search angles covering [0, 360)."""
    num_steps = 360 // step_deg
    return [i * step_deg for i in range(num_steps)]


def _deg_to_us(deg: int, perimeter: int) -> int:
    """Convert degree shift to microseconds, rounding to nearest integer."""
    return round(deg * perimeter / 360)


def _pre_sample(circles: list[CircleAbstraction]) -> list[list[float]]:
    """Pre-sample all circles' demand curves into flat lookup arrays.

    Returns samples[i][angle] = demand at that angle for circle i.
    """
    return [[c.demand_at(a) for a in range(360)] for c in circles]


def compute_score(
    circles: list[CircleAbstraction],
    shifts_deg: list[int],
    link_capacity: float,
    num_angles: int = 360,
    samples: list[list[float]] | None = None,
) -> float:
    """Evaluate the compatibility score for a set of circles and shifts.

    Args:
        circles: List of circles.
        shifts_deg: Rotation in degrees for each circle (same order).
        link_capacity: Link bandwidth capacity in Gbps.
        num_angles: Angular resolution (default 360).
        samples: Pre-sampled demand arrays (samples[i][a]).  If None,
            computed from circles on the fly.

    Returns:
        Score in [−inf, 1]; higher is better.
    """
    total_excess = 0.0

    if samples is None:
        samples = [[c.demand_at(a) for a in range(360)] for c in circles]

    for a in range(num_angles):
        total_demand = 0.0
        for i, shift in enumerate(shifts_deg):
            total_demand += samples[i][(a - shift) % 360]

        excess = max(0.0, total_demand - link_capacity)
        total_excess += excess

    denominator = num_angles * link_capacity
    if denominator <= 0:
        return 0.0

    return 1.0 - total_excess / denominator


def optimize_link_compatibility(
    circles: dict[int, CircleAbstraction],
    link_capacity: float,
    step_deg: int = 5,
    fixed_shifts_deg: dict[int, int] | None = None,
) -> CompatibilityResult:
    """Find near-optimal time-shifts for a group of jobs sharing a link.

    When *fixed_shifts_deg* is None or empty all circles are optimised freely
    (existing behaviour).  Otherwise the listed circles keep their pre-assigned
    rotation while the remaining circles are grid-searched.

    Args:
        circles: Mapping of circle_index → CircleAbstraction to optimise.
        link_capacity: Link bandwidth capacity in Gbps.
        step_deg: Angular step for grid search (default 5°).
        fixed_shifts_deg: Pre-assigned shifts in degrees for a subset of
            circles.  Circles NOT in this dict are optimised.

    Returns:
        CompatibilityResult with the best score and corresponding
        time-shifts (in microseconds).
    """
    if not circles:
        return CompatibilityResult(score=1.0, time_shifts_us={})

    indices = sorted(circles.keys())
    circle_list = [circles[i] for i in indices]
    n = len(circle_list)

    # ── All-free path (original behaviour) ──
    if not fixed_shifts_deg:
        if n == 1:
            return CompatibilityResult(score=1.0, time_shifts_us={indices[0]: 0})
        if n == 2:
            return _optimize_two(circle_list, indices, link_capacity, step_deg)
        return _optimize_multi(circle_list, indices, link_capacity, step_deg)

    # ── Partial-fix path ──
    unfixed = [i for i in range(n) if indices[i] not in fixed_shifts_deg]

    if not unfixed:
        # All fixed — evaluate only
        shifts_deg = [fixed_shifts_deg[indices[i]] for i in range(n)]
        score = compute_score(circle_list, shifts_deg, link_capacity)
        return CompatibilityResult(
            score=score,
            time_shifts_us={
                indices[i]: _deg_to_us(shifts_deg[i], circle_list[i].perimeter)
                for i in range(n)
            },
        )

    if len(unfixed) == 1:
        return _search_one_fixed(
            circle_list, indices, unfixed[0], fixed_shifts_deg,
            link_capacity, step_deg,
        )

    return _search_multi_fixed(
        circle_list, indices, unfixed, fixed_shifts_deg,
        link_capacity, step_deg,
    )


def _optimize_two(
    circles: list[CircleAbstraction],
    indices: list[int],
    link_capacity: float,
    step_deg: int,
) -> CompatibilityResult:
    """Exhaustive grid search for exactly two jobs.

    Fixes job 0 at 0° and searches the full rotation space of job 1.
    When the best non-zero shift provides negligible improvement over no
    shift (compatibility delta < 0.02), prefers shift 0 — the delay cost
    of a useless shift always hurts makespan.
    """
    samples = _pre_sample(circles)
    angles = _search_angles(step_deg)
    score_zero = compute_score(circles, [0, 0], link_capacity, samples=samples)
    best_score = score_zero
    best_s1 = 0

    for s1 in angles:
        score = compute_score(circles, [0, s1], link_capacity, samples=samples)
        if score > best_score + 1e-9:
            best_score = score
            best_s1 = s1

    # When the improvement is negligible the delay is pure waste.
    if best_score - score_zero < 0.02:
        best_s1 = 0
        best_score = score_zero

    return CompatibilityResult(
        score=best_score,
        time_shifts_us={
            indices[0]: 0,
            indices[1]: _deg_to_us(best_s1, circles[1].perimeter),
        },
    )


def _optimize_multi(
    circles: list[CircleAbstraction],
    indices: list[int],
    link_capacity: float,
    step_deg: int,
    max_iter: int = 10,
) -> CompatibilityResult:
    """Iterative greedy optimisation for 3+ jobs.

    At each round, fix all shifts except one and grid-search the free
    axis.  Repeat until convergence or max_iter.

    When the best non-zero shifts provide negligible improvement over all
    zeros (compatibility delta < 0.02), resets to zero — the delay cost
    of useless shifts always hurts makespan.
    """
    n = len(circles)
    samples = _pre_sample(circles)
    angles = _search_angles(step_deg)
    score_zero = compute_score(circles, [0] * n, link_capacity, samples=samples)
    best_shifts = [0] * n

    for _ in range(max_iter):
        improved = False
        for idx in range(n):
            # Per-dimension local best — compare within this coordinate only.
            # A global best_score would stall: once a high score is reached,
            # no single-coordinate change can beat it even when a better
            # value exists for that coordinate in the current context.
            best_s = best_shifts[idx]
            best_local = compute_score(
                circles, best_shifts, link_capacity, samples=samples,
            )

            for s in angles:
                candidate = list(best_shifts)
                candidate[idx] = s
                score = compute_score(
                    circles, candidate, link_capacity, samples=samples
                )
                if score > best_local + 1e-9:
                    best_local = score
                    best_s = s

            if best_s != best_shifts[idx]:
                improved = True
            best_shifts[idx] = best_s

        if not improved:
            break

    final_score = compute_score(circles, best_shifts, link_capacity, samples=samples)
    if final_score - score_zero < 0.02:
        best_shifts = [0] * n
        final_score = score_zero

    return CompatibilityResult(
        score=final_score,
        time_shifts_us={
            indices[i]: _deg_to_us(best_shifts[i], circles[i].perimeter)
            for i in range(n)
        },
    )


# ---------------------------------------------------------------------------
# Partial-fix search helpers (one or more dimensions pre-locked)
# ---------------------------------------------------------------------------


def _search_one_fixed(
    circle_list: list[CircleAbstraction],
    indices: list[int],
    free_pos: int,
    fixed_shifts_deg: dict[int, int],
    link_capacity: float,
    step_deg: int,
) -> CompatibilityResult:
    """1D grid search: one free circle, the rest pre-fixed."""
    n = len(circle_list)
    samples = _pre_sample(circle_list)
    angles = _search_angles(step_deg)

    base = [fixed_shifts_deg.get(indices[i], 0) for i in range(n)]
    best_score = -float("inf")
    best_deg = 0

    for s in angles:
        candidate = list(base)
        candidate[free_pos] = s
        score = compute_score(circle_list, candidate, link_capacity, samples=samples)
        if score > best_score:
            best_score = score
            best_deg = s

    base[free_pos] = best_deg
    return CompatibilityResult(
        score=best_score,
        time_shifts_us={
            indices[i]: _deg_to_us(base[i], circle_list[i].perimeter)
            for i in range(n)
        },
    )


def _search_multi_fixed(
    circle_list: list[CircleAbstraction],
    indices: list[int],
    unfixed: list[int],
    fixed_shifts_deg: dict[int, int],
    link_capacity: float,
    step_deg: int,
    max_iter: int = 10,
) -> CompatibilityResult:
    """Iterative greedy search with a subset of dimensions pre-locked.

    Only *unfixed* positions participate in the grid search; pre-assigned
    dimensions never change.
    """
    n = len(circle_list)
    samples = _pre_sample(circle_list)
    angles = _search_angles(step_deg)

    best_shifts = [fixed_shifts_deg.get(indices[i], 0) for i in range(n)]

    for _ in range(max_iter):
        improved = False
        for pos in unfixed:
            best_s = best_shifts[pos]
            best_local = compute_score(
                circle_list, best_shifts, link_capacity, samples=samples,
            )
            for s in angles:
                candidate = list(best_shifts)
                candidate[pos] = s
                score = compute_score(
                    circle_list, candidate, link_capacity, samples=samples,
                )
                if score > best_local:
                    best_local = score
                    best_s = s
            if best_s != best_shifts[pos]:
                improved = True
            best_shifts[pos] = best_s
        if not improved:
            break

    final_score = compute_score(circle_list, best_shifts, link_capacity, samples=samples)
    return CompatibilityResult(
        score=final_score,
        time_shifts_us={
            indices[i]: _deg_to_us(best_shifts[i], circle_list[i].perimeter)
            for i in range(n)
        },
    )
