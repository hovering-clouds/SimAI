"""
Geometric circle abstraction for periodic communication patterns.

Cassini "rolls" time onto a circle:
    - Circumference = job iteration time (perimeter)
    - Each angle 0-359 maps to a bandwidth demand bw(α)
    - Rotating the circle by ∆ applies a time-shift to the job.

Two jobs with different iteration times cannot share the same circle
directly.  The build_unified method computes an LCM perimeter and tiles
each job's pattern r = LCM / T times around the unified circle.
"""

from dataclasses import dataclass

from .communication_pattern import CommunicationPattern

# Upper bound on unified perimeter (50 seconds in microseconds) to prevent
# unbounded LCM growth when jobs have coprime iteration times.
_MAX_UNIFIED_PERIMETER_US = 50_000_000


@dataclass
class CircleAbstraction:
    """A geometric circle encoding a job's periodic bandwidth demand.

    Attributes:
        perimeter: Circumference in microseconds (= iteration time).
        bw_demand: Dict mapping angle 0-359 to bandwidth demand in Gbps.
    """

    perimeter: int
    bw_demand: dict[int, float]

    def rotate(self, shift_deg: int) -> "CircleAbstraction":
        """Return a new circle rotated clockwise by shift_deg degrees.

        Rotating the circle is equivalent to delaying the job's iteration
        start by (shift_deg / 360) * perimeter microseconds.
        """
        return CircleAbstraction(
            perimeter=self.perimeter,
            bw_demand={(a + shift_deg) % 360: bw for a, bw in self.bw_demand.items()},
        )

    def demand_at(self, angle: int) -> float:
        """Return the bandwidth demand at the given angle (0-359)."""
        return self.bw_demand.get(angle % 360, 0.0)

    @staticmethod
    def from_pattern(
        pattern: CommunicationPattern,
        link_id: tuple[int, int],
    ) -> "CircleAbstraction":
        """Build a circle for a specific link from a CommunicationPattern.

        Args:
            pattern: The job's communication pattern.
            link_id: The link (src, dst) to extract.

        Returns:
            A CircleAbstraction with all 360 angles populated
            (missing positions default to 0 Gbps).
        """
        raw = pattern.link_demands.get(link_id, {})
        return CircleAbstraction(
            perimeter=pattern.iteration_time_us,
            bw_demand={a: raw.get(a, 0.0) for a in range(360)},
        )

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        """Least common multiple of two integers."""
        import math

        return a // math.gcd(a, b) * b

    @staticmethod
    def build_unified(
        patterns: list[CommunicationPattern],
        link_id: tuple[int, int],
        max_perimeter: int = _MAX_UNIFIED_PERIMETER_US,
    ) -> list["CircleAbstraction"]:
        """Build unified (LCM) circles for multiple jobs on a shared link.

        When jobs have different iteration times, they can't be compared
        on the same circle directly.  This method:
            1. Builds individual circles for each job on the given link.
            2. Computes the LCM of all iteration times as the unified perimeter.
            3. Tiles each job's pattern r = LCM / T times around the circle.

        Args:
            patterns: Communication patterns for the jobs of interest.
            link_id: The shared link to analyze.
            max_perimeter: Upper bound on the unified perimeter to prevent
                unbounded growth (default 50 seconds in microseconds).

        Returns:
            List of unified CircleAbstraction objects, one per job that uses
            the link.  Empty list if no pattern uses the given link.
        """
        circles: list[CircleAbstraction] = []
        for p in patterns:
            if link_id in p.link_demands and p.iteration_time_us > 0:
                circles.append(CircleAbstraction.from_pattern(p, link_id))

        if not circles:
            return []

        lcm = 1
        for c in circles:
            lcm = CircleAbstraction._lcm(lcm, c.perimeter)

        overflowed = lcm <= 0 or lcm > max_perimeter

        if overflowed:
            lcm = max(c.perimeter for c in circles)

        unified: list[CircleAbstraction] = []
        for c in circles:
            tile_bw: dict[int, float] = {}

            if overflowed:
                for a_uni in range(360):
                    t_us = a_uni * lcm // 360
                    a_orig = (t_us * 360 // c.perimeter) % 360
                    tile_bw[a_uni] = c.demand_at(a_orig)
            else:
                r = lcm // c.perimeter
                for a_uni in range(360):
                    a_start = (a_uni * r) % 360
                    tile_bw[a_uni] = max(
                        c.demand_at((a_start + i) % 360) for i in range(r)
                    )

            unified.append(
                CircleAbstraction(perimeter=lcm, bw_demand=tile_bw)
            )

        return unified
