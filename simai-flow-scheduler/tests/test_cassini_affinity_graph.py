"""Unit tests for src/cassini/affinity_graph.py — BFS traversal & global shifts."""

from src.static_analysis.passes.cassini_affinity_graph import (
    AffinityGraph,
    build_affinity_graph,
    compute_cluster_time_shifts,
    _find_connected_components,
)
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


# ---------------------------------------------------------------------------
# build_affinity_graph
# ---------------------------------------------------------------------------


class TestBuildAffinityGraph:
    def test_filters_uncontended_links(self):
        """Links with only 1 job are excluded from link_jobs."""
        job_links = {0: {(0, 1)}, 1: {(2, 3)}}
        patterns = {
            0: _make_pattern(job_id=0, link_demands={(0, 1): {0: 5.0}}),
            1: _make_pattern(job_id=1, link_demands={(2, 3): {0: 5.0}}),
        }
        link_caps = {(0, 1): 100.0, (2, 3): 100.0}

        graph = build_affinity_graph(patterns, job_links, link_caps)
        assert len(graph.link_jobs) == 0  # neither link is contended

    def test_keeps_contended_links(self):
        """Links traversed by 2+ jobs are kept."""
        job_links = {0: {(0, 1)}, 1: {(0, 1)}}
        patterns = {
            0: _make_pattern(job_id=0, link_demands={(0, 1): {0: 5.0}}),
            1: _make_pattern(job_id=1, link_demands={(0, 1): {0: 3.0}}),
        }
        link_caps = {(0, 1): 100.0}

        graph = build_affinity_graph(patterns, job_links, link_caps)
        assert (0, 1) in graph.link_jobs
        assert graph.link_jobs[(0, 1)] == {0, 1}

    def test_mixed_contended_and_uncontended(self):
        """Only contended links survive; uncontended are dropped."""
        job_links = {
            0: {(0, 1), (1, 2)},
            1: {(0, 1), (3, 4)},
        }
        patterns = {
            0: _make_pattern(job_id=0, link_demands={
                (0, 1): {0: 5.0}, (1, 2): {0: 2.0}}),
            1: _make_pattern(job_id=1, link_demands={
                (0, 1): {0: 3.0}, (3, 4): {0: 1.0}}),
        }
        link_caps = {(0, 1): 100.0, (1, 2): 100.0, (3, 4): 100.0}

        graph = build_affinity_graph(patterns, job_links, link_caps)
        # (0,1) shared by 2 jobs → kept; (1,2) and (3,4) single-job → dropped
        assert (0, 1) in graph.link_jobs
        assert (1, 2) not in graph.link_jobs
        assert (3, 4) not in graph.link_jobs


# ---------------------------------------------------------------------------
# _find_connected_components
# ---------------------------------------------------------------------------


class TestFindConnectedComponents:
    def test_single_component(self):
        """Two jobs sharing one link form one component."""
        graph = AffinityGraph(
            job_links={0: {(0, 1)}, 1: {(0, 1)}},
            link_jobs={(0, 1): {0, 1}},
            link_capacities={(0, 1): 100.0},
            patterns={
                0: _make_pattern(link_demands={(0, 1): {0: 5.0}}),
                1: _make_pattern(job_id=1, link_demands={(0, 1): {0: 3.0}}),
            },
        )
        components = _find_connected_components(graph)
        assert len(components) == 1
        assert components[0] == {0, 1}

    def test_disjoint_components(self):
        """Jobs on disjoint link sets form separate components."""
        graph = AffinityGraph(
            job_links={0: {(0, 1)}, 1: {(2, 3)}, 2: {(4, 5)}, 3: {(4, 5)}},
            link_jobs={(0, 1): {0}, (2, 3): {1}, (4, 5): {2, 3}},
            link_capacities={(0, 1): 100.0, (2, 3): 100.0, (4, 5): 100.0},
            patterns={
                0: _make_pattern(link_demands={(0, 1): {0: 5.0}}),
                1: _make_pattern(job_id=1, link_demands={(2, 3): {0: 3.0}}),
                2: _make_pattern(job_id=2, link_demands={(4, 5): {0: 2.0}}),
                3: _make_pattern(job_id=3, link_demands={(4, 5): {0: 1.0}}),
            },
        )
        components = _find_connected_components(graph)
        assert len(components) == 3  # {0}, {1}, {2,3}

    def test_chain_connectivity(self):
        """Job0→link0→Job1→link1→Job2 forms one chain component."""
        graph = AffinityGraph(
            job_links={
                0: {(0, 1)},
                1: {(0, 1), (1, 2)},
                2: {(1, 2)},
            },
            link_jobs={
                (0, 1): {0, 1},
                (1, 2): {1, 2},
            },
            link_capacities={(0, 1): 100.0, (1, 2): 100.0},
            patterns={
                0: _make_pattern(link_demands={(0, 1): {0: 5.0}}),
                1: _make_pattern(job_id=1, link_demands={(0, 1): {0: 3.0}, (1, 2): {0: 2.0}}),
                2: _make_pattern(job_id=2, link_demands={(1, 2): {0: 1.0}}),
            },
        )
        components = _find_connected_components(graph)
        assert len(components) == 1
        assert components[0] == {0, 1, 2}


# ---------------------------------------------------------------------------
# compute_cluster_time_shifts
# ---------------------------------------------------------------------------


class TestComputeClusterTimeShifts:
    def test_no_contended_links_all_zero(self):
        """All shifts are 0 when no contended links exist."""
        graph = AffinityGraph(
            job_links={0: {(0, 1)}, 1: {(2, 3)}},
            link_jobs={},  # empty → no contended links
            link_capacities={(0, 1): 100.0, (2, 3): 100.0},
            patterns={
                0: _make_pattern(link_demands={(0, 1): {0: 5.0}}),
                1: _make_pattern(job_id=1, link_demands={(2, 3): {0: 3.0}}),
            },
        )
        shifts = compute_cluster_time_shifts(graph, step_deg=15)
        assert shifts[0] == 0
        assert shifts[1] == 0

    def test_single_contended_link(self):
        """Two jobs on one contended link → shifts computed via optimisation."""
        # Two identical patterns on the same link at high demand → conflict
        graph = AffinityGraph(
            job_links={0: {(0, 1)}, 1: {(0, 1)}},
            link_jobs={(0, 1): {0, 1}},
            link_capacities={(0, 1): 10.0},  # small capacity to force conflict
            patterns={
                0: _make_pattern(link_demands={(0, 1): {0: 10.0, 90: 10.0}}),
                1: _make_pattern(job_id=1, link_demands={(0, 1): {0: 10.0, 90: 10.0}}),
            },
        )
        shifts = compute_cluster_time_shifts(graph, step_deg=30)
        assert 0 in shifts
        assert 1 in shifts
        assert isinstance(shifts[0], int)
        assert isinstance(shifts[1], int)
        # root job (highest demand) gets shift 0
        # At least one job has a shift assigned

    def test_empty_graph(self):
        """Empty graph yields empty shifts dict."""
        graph = AffinityGraph()
        shifts = compute_cluster_time_shifts(graph)
        assert shifts == {}
