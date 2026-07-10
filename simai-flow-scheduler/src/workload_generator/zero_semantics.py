"""Name classification helpers for AICB DeepSpeed ZeRO SimAI rows."""

import re
from enum import Enum

from .aicb_parser import AicbWorkItem


class ZeroItemKind(str, Enum):
    NONE = "none"
    FWD_PARAM_ALLGATHER = "fwd_param_allgather"
    BWD_PARAM_ALLGATHER = "bwd_param_allgather"
    FWD_COMPUTE = "fwd_compute"
    BWD_COMPUTE = "bwd_compute"
    BWD_WEIGHT_COMPUTE = "bwd_weight_compute"
    GRAD_REDUCESCATTER = "grad_reducescatter"
    STEP_GRAD_REDUCESCATTER = "step_grad_reducescatter"
    GRAD_SYNC = "grad_sync"
    STEP = "step"
    INIT = "init"
    GA_BOUNDARY = "ga_boundary"


_ZERO_PREFIX_RE = re.compile(r"^zero[123]_")
_ZERO_FORWARD_PARAM_RE = re.compile(r"^zero[123]_forward_param_\d+$")
_ZERO_BACKWARD_PARAM_RE = re.compile(r"^zero[123]_backward_param_\d+$")
_ZERO_BACKWARD_WEIGHT_PARAM_RE = re.compile(
    r"^zero[123]_backward_param_\d+_weight_grad$"
)


def classify_zero_item(name: str) -> ZeroItemKind:
    """Classify a SimAI row name produced by the DeepSpeed ZeRO generator."""
    if not _ZERO_PREFIX_RE.match(name):
        return ZeroItemKind.NONE

    if name in {"zero1_ga_boundary", "zero2_ga_boundary", "zero3_ga_boundary"}:
        return ZeroItemKind.GA_BOUNDARY

    if name.endswith("_init_broadcast_model") or name == "zero3_init_param_allgather":
        return ZeroItemKind.INIT

    if name in {
        "zero1_has_overflow",
        "zero2_has_overflow",
        "zero3_has_overflow",
        "zero2_grad_norm",
        "zero3_grad_norm",
        "zero1_param_allgather",
        "zero2_param_allgather",
        "zero3_step_persistent_param_allgather",
    }:
        return ZeroItemKind.STEP

    if name in {"zero3_forward_param_allgather"}:
        return ZeroItemKind.FWD_PARAM_ALLGATHER
    if name in {"zero3_backward_param_allgather"}:
        return ZeroItemKind.BWD_PARAM_ALLGATHER
    if name.startswith("zero3_forward_allgather_"):
        return ZeroItemKind.FWD_PARAM_ALLGATHER
    if name.startswith("zero3_backward_allgather_"):
        return ZeroItemKind.BWD_PARAM_ALLGATHER

    if name == "zero3_step_grad_reduce_scatter":
        return ZeroItemKind.STEP_GRAD_REDUCESCATTER

    if name == "zero3_grad_reduce_scatter" or name.startswith(
        "zero3_grad_reducescatter_"
    ):
        return ZeroItemKind.GRAD_REDUCESCATTER

    if name in {"zero1_grad_sync", "zero2_grad_sync"}:
        return ZeroItemKind.GRAD_SYNC
    if name.startswith("zero1_grad_sync_") or name.startswith("zero2_grad_sync_"):
        return ZeroItemKind.GRAD_SYNC

    if _ZERO_BACKWARD_WEIGHT_PARAM_RE.match(name):
        return ZeroItemKind.BWD_WEIGHT_COMPUTE
    if _ZERO_BACKWARD_PARAM_RE.match(name):
        return ZeroItemKind.BWD_COMPUTE
    if _ZERO_FORWARD_PARAM_RE.match(name):
        return ZeroItemKind.FWD_COMPUTE

    return ZeroItemKind.NONE


def is_zero_workload(items: list[AicbWorkItem]) -> bool:
    """Return True if any item is a recognized DeepSpeed ZeRO row."""
    return any(classify_zero_item(item.name) is not ZeroItemKind.NONE for item in items)


def is_zero_pre_item(name: str) -> bool:
    return classify_zero_item(name) is ZeroItemKind.INIT


def is_zero_post_item(name: str) -> bool:
    return classify_zero_item(name) in {
        ZeroItemKind.STEP,
        ZeroItemKind.STEP_GRAD_REDUCESCATTER,
    }
