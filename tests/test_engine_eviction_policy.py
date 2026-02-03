import pytest

from faiss.engine.engine import _EvictionTracker, EvictionPolicyError


def test_eviction_tracker_lru_order():
    tracker = _EvictionTracker("lru")
    tracker.record_loaded(1)
    tracker.record_loaded(2)
    tracker.record_loaded(3)
    tracker.record_access([1])

    victim = tracker.pop_victim(protected=set())
    assert victim == 2


def test_eviction_tracker_fifo_order():
    tracker = _EvictionTracker("fifo")
    tracker.record_loaded(1)
    tracker.record_loaded(2)
    tracker.record_loaded(3)
    tracker.record_access([1])

    victim = tracker.pop_victim(protected=set())
    assert victim == 1


def test_eviction_tracker_respects_protected():
    tracker = _EvictionTracker("lru")
    tracker.record_loaded(10)
    tracker.record_loaded(20)

    victim = tracker.pop_victim(protected={10})
    assert victim == 20


def test_eviction_tracker_invalid_policy():
    with pytest.raises(EvictionPolicyError):
        _EvictionTracker("clock")
