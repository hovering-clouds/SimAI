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

import math
from dataclasses import dataclass

from .communication_pattern import CommunicationPattern


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
        rotated: dict[int, float] = {}
        for a, bw in self.bw_demand.items():
            rotated[(a + shift_deg) % 360] = bw
        return CircleAbstraction(
            perimeter=self.perimeter,
            bw_demand=rotated,
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
        raw_demands = pattern.link_demands.get(link_id, {})
        full: dict[int, float] = {}
        for a in range(360):
            full[a] = raw_demands.get(a, 0.0)
        return CircleAbstraction(
            perimeter=pattern.iteration_time_us,
            bw_demand=full,
        )

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        """Least common multiple of two integers."""
        return a // math.gcd(a, b) * b

    @staticmethod
    def build_unified(
        patterns: list[CommunicationPattern],
        link_id: tuple[int, int],
        max_perimeter: int = 10_000_000,
    ) -> tuple[int, list["CircleAbstraction"]]:
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
                unbounded growth (default 10 seconds in microseconds).

        Returns:
            Tuple of (lcm_perimeter, list of unified CircleAbstraction objects).
            Returns (0, []) if no pattern uses the given link.
        """
        circles: list[CircleAbstraction] = []
        for p in patterns:
            if link_id in p.link_demands and p.iteration_time_us > 0:
                circles.append(CircleAbstraction.from_pattern(p, link_id))

        if not circles:
            return 0, []

        lcm = 1
        for c in circles:
            lcm = CircleAbstraction._lcm(lcm, c.perimeter)

        overflow = lcm <= 0 or lcm > max_perimeter

        if overflow:
            # LCM overflow: use max perimeter with time-proportional mapping.
            # Each job's original angles are mapped onto the unified circle
            # through the time domain so that every angle represents the same
            # absolute wall-clock instant across all jobs.
            #
            #   angle_u  →  time = angle_u × P_max / 360
            #   time     →  angle_orig = (time % P_i) × 360 / P_i
            #
            lcm = max(c.perimeter for c in circles)

        unified: list[CircleAbstraction] = []
        for c in circles:
            tile_bw: dict[int, float] = {}

            if overflow:
                # Time-proportional mapping — no integer-tiling assumption.
                # Each unified-degree bucket maps to a single closest
                # original-degree bucket.  No max-pooling: at 360 samples
                # the granularity is fine enough (~0.3 ms/° worst case).
                for a_uni in range(360):
                    t_us = a_uni * lcm // 360
                    a_orig = (t_us * 360 // c.perimeter) % 360
                    tile_bw[a_uni] = c.demand_at(a_orig)
            else:
                r = lcm // c.perimeter
                for a_uni in range(360):
                    # Each unified bucket spans r consecutive original buckets.
                    # Max-pool over all r to avoid understating peak demand.
                    a_start = (a_uni * r) % 360
                    peak = max(
                        c.demand_at((a_start + i) % 360) for i in range(r)
                    )
                    tile_bw[a_uni] = peak

            unified.append(
                CircleAbstraction(
                    perimeter=lcm,
                    bw_demand=tile_bw,
                )
            )

        return lcm, unified
