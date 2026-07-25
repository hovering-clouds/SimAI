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

from .cassini_circle_abstraction import CircleAbstraction

# Minimum compatibility improvement threshold.  If rotating a job improves the
# score by less than this, the shift is discarded — the scheduling delay it
# introduces always hurts makespan more than the contention it avoids.
_MIN_COMPAT_IMPROVEMENT = 0.02


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

    When *fixed_shifts_deg* is None or empty all circles are optimised freely.
    Otherwise the listed circles keep their pre-assigned rotation while the
    remaining circles are grid-searched.

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

    # N=1 is always contention-free
    if n == 1:
        return CompatibilityResult(score=1.0, time_shifts_us={indices[0]: 0})

    fixed = fixed_shifts_deg or {}

    # All fixed — evaluate only
    if len(fixed) == n:
        shifts_deg = [fixed[i] for i in indices]
        score = compute_score(circle_list, shifts_deg, link_capacity)
        return CompatibilityResult(
            score=score,
            time_shifts_us={
                indices[i]: _deg_to_us(shifts_deg[i], circle_list[i].perimeter)
                for i in range(n)
            },
        )

    # 2 jobs, nothing pre-fixed → fix one at 0°, search the other
    if n == 2 and not fixed_shifts_deg:
        return _optimize_two(
            circle_list, indices, link_capacity, step_deg,
            {indices[0]: 0},
        )

    # General case: 1 unfixed → _optimize_two, 2+ → _optimize_multi
    unfixed_count = n - len(fixed)
    if unfixed_count == 1:
        return _optimize_two(circle_list, indices, link_capacity, step_deg, fixed)
    return _optimize_multi(circle_list, indices, link_capacity, step_deg, fixed_shifts_deg=fixed)


def _optimize_two(
    circles: list[CircleAbstraction],
    indices: list[int],
    link_capacity: float,
    step_deg: int,
    fixed_shifts_deg: dict[int, int] | None = None,
) -> CompatibilityResult:
    """1D grid search: one free circle, the rest fixed.

    When *fixed_shifts_deg* is None (the all-free 2-job case), fixes
    circle at indices[0] at 0° and searches indices[1].
    Otherwise searches the one dimension NOT in fixed_shifts_deg.

    When the best non-zero shift provides negligible improvement over no
    shift (compatibility delta < 0.02), prefers shift 0 — the delay cost
    of a useless shift always hurts makespan.
    """
    n = len(circles)
    samples = _pre_sample(circles)
    angles = _search_angles(step_deg)

    if fixed_shifts_deg is None:
        fixed_shifts_deg = {}
    free_pos = next(i for i in range(n) if indices[i] not in fixed_shifts_deg)

    base = [fixed_shifts_deg.get(indices[i], 0) for i in range(n)]
    score_zero = compute_score(circles, base, link_capacity, samples=samples)
    best_score = score_zero
    best_deg = 0

    for s in angles:
        candidate = list(base)
        candidate[free_pos] = s
        score = compute_score(circles, candidate, link_capacity, samples=samples)
        if score > best_score + 1e-9:
            best_score = score
            best_deg = s

    if best_score - score_zero < _MIN_COMPAT_IMPROVEMENT:
        best_deg = 0
        best_score = score_zero

    base[free_pos] = best_deg
    return CompatibilityResult(
        score=best_score,
        time_shifts_us={
            indices[i]: _deg_to_us(base[i], circles[i].perimeter)
            for i in range(n)
        },
    )


def _optimize_multi(
    circles: list[CircleAbstraction],
    indices: list[int],
    link_capacity: float,
    step_deg: int,
    max_iter: int = 10,
    fixed_shifts_deg: dict[int, int] | None = None,
) -> CompatibilityResult:
    """Iterative greedy optimisation for 3+ jobs.

    At each round, fix all shifts except one and grid-search the free
    axis.  Repeat until convergence or max_iter.

    When *fixed_shifts_deg* is provided, only dimensions NOT in the dict
    participate in the search; pre-assigned dimensions stay locked.

    When the improvement over the no-shift baseline is negligible
    (compatibility delta < 0.02), resets unfixed dimensions to 0 — the
    delay cost of useless shifts always hurts makespan.
    """
    n = len(circles)
    samples = _pre_sample(circles)
    angles = _search_angles(step_deg)

    if fixed_shifts_deg is None:
        fixed_shifts_deg = {}
    unfixed = [i for i in range(n) if indices[i] not in fixed_shifts_deg]

    best_shifts = [fixed_shifts_deg.get(indices[i], 0) for i in range(n)]

    # Baseline: keep fixed positions, set unfixed to 0
    baseline = list(best_shifts)
    for pos in unfixed:
        baseline[pos] = 0
    score_zero = compute_score(circles, baseline, link_capacity, samples=samples)

    for _ in range(max_iter):
        improved = False
        for pos in unfixed:
            best_s = best_shifts[pos]
            best_local = compute_score(
                circles, best_shifts, link_capacity, samples=samples,
            )

            for s in angles:
                candidate = list(best_shifts)
                candidate[pos] = s
                score = compute_score(
                    circles, candidate, link_capacity, samples=samples
                )
                if score > best_local + 1e-9:
                    best_local = score
                    best_s = s

            if best_s != best_shifts[pos]:
                improved = True
            best_shifts[pos] = best_s

        if not improved:
            break

    final_score = compute_score(circles, best_shifts, link_capacity, samples=samples)
    if final_score - score_zero < _MIN_COMPAT_IMPROVEMENT:
        for pos in unfixed:
            best_shifts[pos] = 0
        final_score = score_zero

    return CompatibilityResult(
        score=final_score,
        time_shifts_us={
            indices[i]: _deg_to_us(best_shifts[i], circles[i].perimeter)
            for i in range(n)
        },
    )







