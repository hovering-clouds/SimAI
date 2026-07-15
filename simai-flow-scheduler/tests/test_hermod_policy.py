import pytest

from src.executor.policies.hermod_policy import HermodSchedulingPolicy


def test_dynamic_analysis_update_is_explicitly_rejected():
    """DynamicExecutor must not silently bypass Hermod priority metadata."""
    policy = object.__new__(HermodSchedulingPolicy)
    with pytest.raises(NotImplementedError, match="Dynamic Hermod scheduling"):
        policy.update_analysis(None, None)
