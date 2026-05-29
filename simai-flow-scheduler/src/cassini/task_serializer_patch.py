"""Cassini compatibility patch for compute serialization.

Replicated Cassini workloads use iteration labels grouped as full training
windows: ``[-1, 0, 1]``, ``[2, 3, 4]``, ... .  The generic C++ reference
serializer sorts all forward compute before all backward compute across the
entire workload.  That ordering conflicts with Cassini's cross-iteration
dependency ``copy_k -> copy_{k+1}`` and creates a validation cycle.

This module keeps the generic serializer file unchanged and installs a narrow
runtime patch that only changes ordering when such replicated Cassini labels
are present.
"""

from __future__ import annotations

from ..static_analysis.passes.task_serializer import CppReferenceSerializer
from ..workload_format.schema import Phase


_ORIGINAL_SERIALIZE = CppReferenceSerializer.serialize
_ORIGINAL_SORT_KEY = CppReferenceSerializer._compute_sort_key
_PATCHED = False


def _is_replicated_cassini_workload(workload) -> bool:
    iterations = {task.iteration for task in workload.tasks}
    return -1 in iterations and max(iterations, default=0) > 1


def _sort_key_with_iteration(task, iteration: int) -> tuple:
    if task.phase == Phase.FORWARD:
        direction = 0
    elif task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT):
        direction = 1
    else:
        direction = 2

    if task.phase in (Phase.PREFILL, Phase.DECODE):
        return (direction, task.task_id)

    is_backward = task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT)
    ga_sort = -iteration if is_backward else iteration
    layer_sort = -task.layer_id if is_backward else task.layer_id

    if task.phase == Phase.BACKWARD_INPUT:
        sub_phase = 0
    elif task.phase == Phase.BACKWARD_WEIGHT:
        sub_phase = 1
    else:
        sub_phase = 0

    return (
        direction,
        ga_sort,
        layer_sort,
        sub_phase,
        task.item_id,
    )


def _cassini_sort_key(task) -> tuple:
    logical_iter = (task.iteration + 1) // 3
    local_iter = task.iteration - logical_iter * 3
    return (
        logical_iter,
        *_sort_key_with_iteration(task, local_iter),
    )


def _serialize_with_cassini_replicas(self, workload):
    if not _is_replicated_cassini_workload(workload):
        return _ORIGINAL_SERIALIZE(self, workload)

    old_sort_key = CppReferenceSerializer._compute_sort_key
    CppReferenceSerializer._compute_sort_key = staticmethod(_cassini_sort_key)
    try:
        return _ORIGINAL_SERIALIZE(self, workload)
    finally:
        CppReferenceSerializer._compute_sort_key = old_sort_key


def install_cassini_task_serializer_patch() -> None:
    """Install the Cassini replicated-workload serializer compatibility patch."""
    global _PATCHED
    if _PATCHED:
        return
    CppReferenceSerializer.serialize = _serialize_with_cassini_replicas
    _PATCHED = True

