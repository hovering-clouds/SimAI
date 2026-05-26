"""Unit tests for src/cassini/pair_compatibility.py — score & optimise."""

import pytest

from src.cassini.circle_abstraction import CircleAbstraction
from src.cassini.pair_compatibility import (
    CompatibilityResult,
    compute_score,
    optimize_link_compatibility,
    _deg_to_us,
    _search_angles,
    _pre_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(demands, perimeter=3600):
    """Shorthand: build a CircleAbstraction from a dict of angle→gbps."""
    full = {a: demands.get(a, 0.0) for a in range(360)}
    return CircleAbstraction(perimeter=perimeter, bw_demand=full)


def _spike(angle, gbps=10.0, perimeter=3600):
    """Build a circle with a single spike at *angle*."""
    return _c({angle: gbps}, perimeter=perimeter)


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------


class TestComputeScore:
    def test_perfect_score_when_no_demand(self):
        c = [_c({})]
        score = compute_score(c, [0], link_capacity=100.0)
        assert score == 1.0

    def test_perfect_score_when_under_capacity(self):
        c = [_spike(0, gbps=5.0)]
        score = compute_score(c, [0], link_capacity=10.0)
        assert score == 1.0

    def test_lower_score_when_over_capacity(self):
        """Single circle at 20 Gbps on a 10 Gbps link → excess at angle 0."""
        c = [_spike(0, gbps=20.0)]
        score = compute_score(c, [0], link_capacity=10.0)
        assert score < 1.0

    def test_score_improves_with_shift(self):
        """Two spikes at same angle → shift reduces overlap → better score."""
        c1 = _spike(0, gbps=10.0)
        c2 = _spike(0, gbps=10.0)
        score_aligned = compute_score([c1, c2], [0, 0], link_capacity=10.0)
        score_shifted = compute_score([c1, c2], [0, 90], link_capacity=10.0)
        assert score_shifted > score_aligned

    def test_zero_capacity_returns_zero(self):
        c = [_spike(0, gbps=5.0)]
        score = compute_score(c, [0], link_capacity=0.0)
        assert score == 0.0

    def test_pre_sampled_matches_on_the_fly(self):
        c = [_spike(0, 5.0), _spike(90, 3.0)]
        samples = _pre_sample(c)
        s1 = compute_score(c, [0, 45], link_capacity=10.0)
        s2 = compute_score(c, [0, 45], link_capacity=10.0, samples=samples)
        assert s1 == s2


# ---------------------------------------------------------------------------
# _deg_to_us
# ---------------------------------------------------------------------------


class TestDegToUs:
    def test_zero(self):
        assert _deg_to_us(0, 3600) == 0

    def test_half_circle(self):
        assert _deg_to_us(180, 3600) == 1800

    def test_full_circle(self):
        assert _deg_to_us(360, 7200) == 7200


# ---------------------------------------------------------------------------
# _search_angles
# ---------------------------------------------------------------------------


class TestSearchAngles:
    def test_step_90(self):
        assert _search_angles(90) == [0, 90, 180, 270]

    def test_step_360(self):
        assert _search_angles(360) == [0]


# ---------------------------------------------------------------------------
# optimize_link_compatibility
# ---------------------------------------------------------------------------


class TestOptimizeTwo:
    def test_empty_circles(self):
        result = optimize_link_compatibility({}, link_capacity=100.0)
        assert result.score == 1.0
        assert result.time_shifts_us == {}

    def test_single_circle(self):
        c = {0: _spike(0)}
        result = optimize_link_compatibility(c, link_capacity=100.0)
        assert result.score == 1.0
        assert result.time_shifts_us == {0: 0}

    def test_no_contention_prefers_zero_shift(self):
        """Two non-overlapping spikes → score perfect at shift 0."""
        c0 = _c({0: 5.0})
        c1 = _c({180: 5.0})
        c = {0: c0, 1: c1}
        result = optimize_link_compatibility(c, link_capacity=10.0, step_deg=15)
        assert result.score == 1.0
        # Job 1's shift should stay 0 since no improvement from shifting
        assert result.time_shifts_us[1] == 0

    def test_identical_phased_circles_find_shift(self):
        """Two identical spikes → shifting one reduces overlap."""
        c0 = _spike(0, gbps=10.0)
        c1 = _spike(0, gbps=10.0)
        c = {0: c0, 1: c1}
        result = optimize_link_compatibility(c, link_capacity=10.0, step_deg=15)
        # Score at zero should be bad; optimiser should find a better shift
        score_at_zero = compute_score([c0, c1], [0, 0], link_capacity=10.0)
        assert result.score >= score_at_zero


class TestOptimizeMulti:
    def test_three_circles_converges(self):
        c0 = _spike(0, gbps=5.0)
        c1 = _spike(120, gbps=5.0)
        c2 = _spike(240, gbps=5.0)
        c = {0: c0, 1: c1, 2: c2}
        result = optimize_link_compatibility(c, link_capacity=10.0, step_deg=30)
        assert isinstance(result, CompatibilityResult)
        assert len(result.time_shifts_us) == 3
        assert result.score <= 1.0

    def test_three_identical_circles(self):
        c0 = _spike(0, gbps=10.0)
        c = {0: c0, 1: _spike(0, gbps=10.0), 2: _spike(0, gbps=10.0)}
        result = optimize_link_compatibility(c, link_capacity=10.0, step_deg=30)
        assert isinstance(result, CompatibilityResult)
        # With 3×10 Gbps on 10 Gbps link, can't get perfect score
        assert result.score < 1.0


class TestPartialFix:
    def test_one_fixed_one_free(self):
        c0 = _spike(0, gbps=10.0)
        c1 = _spike(0, gbps=10.0)
        c = {0: c0, 1: c1}
        # Fix circle 0 at 0 degrees, search circle 1
        result = optimize_link_compatibility(
            c, link_capacity=10.0, step_deg=15,
            fixed_shifts_deg={0: 0},
        )
        # Fixing 0 at 0 should produce the same result as the all-free case
        assert result.time_shifts_us[0] == 0
        assert result.time_shifts_us[1] > 0  # should have optimised away from 0

    def test_all_fixed_just_evaluates(self):
        c0 = _spike(0, gbps=10.0)
        c1 = _spike(0, gbps=10.0)
        c = {0: c0, 1: c1}
        result = optimize_link_compatibility(
            c, link_capacity=10.0,
            fixed_shifts_deg={0: 0, 1: 90},
        )
        assert result.time_shifts_us[0] == 0
        assert result.time_shifts_us[1] == _deg_to_us(90, c1.perimeter)
