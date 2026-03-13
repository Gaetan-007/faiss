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


def test_eviction_tracker_lfu_order():
    tracker = _EvictionTracker("lfu")
    tracker.record_loaded(1)
    tracker.record_loaded(2)
    tracker.record_loaded(3)

    # Access list 1 twice, list 2 once, list 3 never
    tracker.record_access([1])
    tracker.record_access([1, 2])

    # Least frequently used (3 with freq 0) should be evicted first
    victim1 = tracker.pop_victim(protected=set())
    assert victim1 == 3

    # Next LFU between 1 (freq 2) and 2 (freq 1) should be 2
    victim2 = tracker.pop_victim(protected=set())
    assert victim2 == 2

    # Remaining victim must be 1
    victim3 = tracker.pop_victim(protected=set())
    assert victim3 == 1


def test_eviction_tracker_lfu_respects_protected():
    tracker = _EvictionTracker("lfu")
    tracker.record_loaded(10)
    tracker.record_loaded(20)

    tracker.record_access([10])
    # 10 has higher frequency, but we protect 20 so 10 must be skipped
    victim = tracker.pop_victim(protected={20})
    assert victim == 10
