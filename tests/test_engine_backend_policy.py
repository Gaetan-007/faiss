import types
from types import SimpleNamespace
from typing import Dict

import pytest

import faiss
from faiss.engine.engine import FaissEngine


class _ConcreteFaissEngine(FaissEngine):
    """Concrete subclass for tests; FaissEngine is abstract (_add/_batch_add not implemented)."""

    def _add(self, query: str, return_centroid: bool, retrain: bool):
        raise NotImplementedError("test double")

    def _batch_add(self, query_list, return_centroid: bool, retrain: bool):
        raise NotImplementedError("test double")


class _DummyIndex:
    def __init__(self, nlist: int):
        self.nlist = nlist


def test_refresh_cluster_stats_populates_snapshot(monkeypatch):
    """_refresh_cluster_stats should populate internal snapshot from backend."""

    calls: Dict[str, int] = {"count": 0}

    def fake_get_list_activation_stats(index):
        calls["count"] += 1
        return [
            {
                "list_id": 0,
                "probe_count": 10,
                "load_count": 1,
                "last_probe_ts": 123,
            },
            {
                "list_id": 1,
                "probe_count": 5,
                "load_count": 0,
                "last_probe_ts": 124,
            },
        ]

    monkeypatch.setattr(
        faiss, "get_list_activation_stats", fake_get_list_activation_stats, raising=False
    )

    engine = _ConcreteFaissEngine.__new__(_ConcreteFaissEngine)
    engine.index = _DummyIndex(nlist=2)
    engine.config = SimpleNamespace(
        cluster_stats_refresh_interval_ms=0, cluster_policy_type="backend_lfu"
    )
    engine._cluster_stats_last_snapshot = {}
    engine._cluster_stats_last_update_ts = 0.0

    engine._cluster_policy = engine._build_cluster_policy(engine.config)

    engine._refresh_cluster_stats(force=True)
    snapshot = engine.get_cluster_stats()

    assert calls["count"] >= 1
    assert 0 in snapshot and 1 in snapshot
    assert snapshot[0]["probe_count"] == 10
    assert snapshot[1]["probe_count"] == 5


def test_backend_lfu_policy_drives_eviction(monkeypatch):
    """_evict_one_list should use backend LFU policy when configured."""

    def fake_get_list_activation_stats(index):
        # list 2 is coldest (0 probes), then list 1 (1 probe), list 0 hottest.
        return [
            {"list_id": 0, "probe_count": 100, "load_count": 0, "last_probe_ts": 300},
            {"list_id": 1, "probe_count": 1, "load_count": 0, "last_probe_ts": 200},
            {"list_id": 2, "probe_count": 0, "load_count": 0, "last_probe_ts": 100},
        ]

    evicted = []

    def fake_is_list_on_gpu(index, list_id: int) -> bool:
        return True

    def fake_evict_ivf_lists(index, list_id: int):
        evicted.append(int(list_id))
        return 0

    monkeypatch.setattr(
        faiss, "get_list_activation_stats", fake_get_list_activation_stats, raising=False
    )
    monkeypatch.setattr(faiss, "is_list_on_gpu", fake_is_list_on_gpu, raising=False)
    monkeypatch.setattr(faiss, "evict_ivf_lists", fake_evict_ivf_lists, raising=False)

    engine = _ConcreteFaissEngine.__new__(_ConcreteFaissEngine)
    engine.index = _DummyIndex(nlist=3)
    engine.config = SimpleNamespace(
        cluster_stats_refresh_interval_ms=0, cluster_policy_type="backend_lfu"
    )
    engine._cluster_stats_last_snapshot = {}
    engine._cluster_stats_last_update_ts = 0.0
    engine._cluster_policy = engine._build_cluster_policy(engine.config)
    engine._eviction_tracker = None

    victim = engine._evict_one_list(protected=set())
    assert victim == 2
    assert evicted and evicted[0] == 2

