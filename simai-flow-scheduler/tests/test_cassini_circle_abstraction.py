"""Unit tests for src/cassini/circle_abstraction.py — CircleAbstraction."""

from src.static_analysis.passes.cassini_circle_abstraction import CircleAbstraction
from src.static_analysis.passes.cassini_communication_pattern import CommunicationPattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern(job_id=0, iteration_time_us=3600, link_demands=None):
    return CommunicationPattern(
        job_id=job_id,
        iteration_time_us=iteration_time_us,
        link_demands=link_demands or {},
    )


def _make_circle(perimeter=3600, bw_demand=None):
    if bw_demand is None:
        bw_demand = {a: 1.0 for a in range(0, 360, 10)}
    return CircleAbstraction(perimeter=perimeter, bw_demand=bw_demand)


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------


class TestRotate:
    def test_rotate_zero_is_identity(self):
        c = _make_circle()
        r = c.rotate(0)
        assert r.perimeter == c.perimeter
        for a in c.bw_demand:
            assert r.demand_at(a) == c.demand_at(a)

    def test_rotate_90_shifts_values(self):
        c = CircleAbstraction(perimeter=3600, bw_demand={0: 5.0, 90: 10.0})
        r = c.rotate(90)
        assert r.demand_at(90) == 5.0   # was at 0
        assert r.demand_at(180) == 10.0  # was at 90
        assert r.demand_at(0) == 0.0

    def test_rotate_360_is_identity(self):
        c = _make_circle()
        r = c.rotate(360)
        for a in c.bw_demand:
            assert r.demand_at(a) == c.demand_at(a)


# ---------------------------------------------------------------------------
# demand_at
# ---------------------------------------------------------------------------


class TestDemandAt:
    def test_exact_angle(self):
        c = CircleAbstraction(perimeter=3600, bw_demand={42: 7.5})
        assert c.demand_at(42) == 7.5

    def test_missing_angle_returns_zero(self):
        c = CircleAbstraction(perimeter=3600, bw_demand={0: 1.0})
        assert c.demand_at(100) == 0.0

    def test_out_of_range_wraps(self):
        c = CircleAbstraction(perimeter=3600, bw_demand={5: 3.0})
        assert c.demand_at(365) == 3.0  # 365 % 360 = 5

    def test_negative_angle_wraps(self):
        c = CircleAbstraction(perimeter=3600, bw_demand={355: 8.0})
        assert c.demand_at(-5) == 8.0  # -5 % 360 = 355


# ---------------------------------------------------------------------------
# from_pattern
# ---------------------------------------------------------------------------


class TestFromPattern:
    def test_basic(self):
        p = CommunicationPattern(
            job_id=0,
            iteration_time_us=7200,
            link_demands={(0, 1): {100: 50.0, 200: 25.0}},
        )
        c = CircleAbstraction.from_pattern(p, (0, 1))
        assert c.perimeter == 7200
        assert c.demand_at(100) == 50.0
        assert c.demand_at(200) == 25.0
        assert c.demand_at(0) == 0.0

    def test_missing_link_returns_all_zeros(self):
        p = _make_pattern(link_demands={(0, 1): {10: 5.0}})
        c = CircleAbstraction.from_pattern(p, (2, 3))
        assert c.perimeter == 3600
        for a in range(360):
            assert c.demand_at(a) == 0.0

    def test_all_360_angles_populated(self):
        """from_pattern fills all 360 buckets explicitly (zeros for missing)."""
        p = _make_pattern(link_demands={(0, 1): {0: 1.0}})
        c = CircleAbstraction.from_pattern(p, (0, 1))
        assert len(c.bw_demand) == 360
        assert c.demand_at(0) == 1.0
        assert c.demand_at(1) == 0.0


# ---------------------------------------------------------------------------
# _lcm
# ---------------------------------------------------------------------------


class TestLcm:
    def test_same_value(self):
        assert CircleAbstraction._lcm(12, 12) == 12

    def test_one_multiple_of_other(self):
        assert CircleAbstraction._lcm(6, 12) == 12

    def test_coprime(self):
        assert CircleAbstraction._lcm(7, 11) == 77

    def test_with_one(self):
        assert CircleAbstraction._lcm(1, 100) == 100

    def test_large_values(self):
        assert CircleAbstraction._lcm(3600, 4800) == 14400


# ---------------------------------------------------------------------------
# build_unified
# ---------------------------------------------------------------------------


class TestBuildUnified:
    def test_single_job(self):
        p = _make_pattern(iteration_time_us=1000,
                          link_demands={(0, 1): {0: 10.0}})
        unified = CircleAbstraction.build_unified([p], (0, 1))
        assert len(unified) == 1
        assert unified[0].perimeter == 1000
        assert unified[0].demand_at(0) == 10.0

    def test_same_perimeter(self):
        p1 = _make_pattern(iteration_time_us=3600,
                           link_demands={(0, 1): {0: 5.0}})
        p2 = _make_pattern(job_id=1, iteration_time_us=3600,
                           link_demands={(0, 1): {90: 3.0}})
        unified = CircleAbstraction.build_unified([p1, p2], (0, 1))
        assert len(unified) == 2
        assert unified[0].demand_at(0) == 5.0
        assert unified[1].demand_at(90) == 3.0

    def test_different_perimeter_tiling(self):
        """T1=1800, T2=3600 → LCM=3600, p1 tiled 2× (max-pool)."""
        p1 = _make_pattern(iteration_time_us=1800,
                           link_demands={(0, 1): {0: 10.0}})
        p2 = _make_pattern(job_id=1, iteration_time_us=3600,
                           link_demands={(0, 1): {100: 5.0}})
        unified = CircleAbstraction.build_unified([p1, p2], (0, 1))
        assert len(unified) == 2
        # p1 tiled 2×: r = 3600//1800 = 2, max-pool over 2 consecutive buckets
        # original p1 has demand at angle 0, tiled to angles 0 & 1
        assert unified[0].perimeter == 3600
        assert unified[0].demand_at(0) == 10.0  # max-pool of [0°, 1°] → 10
        assert unified[1].demand_at(100) == 5.0

    def test_no_job_uses_link_returns_empty(self):
        p = _make_pattern()
        unified = CircleAbstraction.build_unified([p], (99, 100))
        assert unified == []
