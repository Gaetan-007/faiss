import time
from concurrent.futures import Future

import pytest

from faiss.engine.engine import BaseEngine
from faiss.engine.scheduler import FifoScheduler, RoundRobinScheduler, SchedulerRequest


class DummyEngine(BaseEngine):
    def __init__(self):
        super().__init__(config=None)
        self.batch_calls = []

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = 1
        results = [{"text": query} for _ in range(num)]
        if return_score:
            return results, [0.0 for _ in range(num)]
        return results

    def _batch_search(
        self,
        query_list,
        query_id_list,
        num,
        return_score,
        eval_cache,
    ):
        if num is None:
            num = 1
        self.batch_calls.append((list(query_list), num, return_score))
        results = [[{"text": query} for _ in range(num)] for query in query_list]
        if return_score:
            scores = [[0.0 for _ in range(num)] for _ in query_list]
            return results, scores
        return results

    def _add(self, query: str, return_centroid: bool, retrain: bool):
        raise NotImplementedError("DummyEngine does not support add")

    def _batch_add(self, query_list, return_centroid: bool, retrain: bool):
        raise NotImplementedError("DummyEngine does not support batch add")


def _make_request(req_id: str, query: str, source_id: str) -> SchedulerRequest:
    return SchedulerRequest(
        id=req_id,
        query=query,
        num=1,
        return_score=False,
        enqueue_time=time.monotonic(),
        source_id=source_id,
        future=Future(),
    )


def test_fifo_scheduler_batches_in_order():
    scheduler = FifoScheduler(max_batch_size=2, max_wait_ms=0, step_interval_ms=0)
    scheduler.add_request(_make_request("r1", "q1", "s1"))
    scheduler.add_request(_make_request("r2", "q2", "s1"))

    assert scheduler.should_step() is True
    batch = scheduler.step()
    assert [req.id for req in batch] == ["r1", "r2"]


def test_round_robin_scheduler_rotates_sources():
    scheduler = RoundRobinScheduler(max_batch_size=3, max_wait_ms=0, step_interval_ms=0)
    scheduler.add_request(_make_request("a1", "q1", "A"))
    scheduler.add_request(_make_request("a2", "q2", "A"))
    scheduler.add_request(_make_request("b1", "q3", "B"))

    batch = scheduler.step()
    assert [req.id for req in batch] == ["a1", "b1", "a2"]


def test_async_batching_engine_returns_results():
    scheduler = FifoScheduler(max_batch_size=8, max_wait_ms=0, step_interval_ms=0)
    engine = DummyEngine()
    engine.enable_async(scheduler, max_queue_size=16, idle_sleep_s=0.001)

    try:
        futures = [
            engine.search_async("alpha", num=2),
            engine.search_async("beta", num=2),
            engine.search_async("gamma", num=2),
        ]
        results = [future.result(timeout=2) for future in futures]
        assert len(results) == 3
        assert results[0][0]["text"] == "alpha"
        assert results[1][0]["text"] == "beta"
        assert results[2][0]["text"] == "gamma"
        assert len(engine.batch_calls) == 1
        assert engine.batch_calls[0][0] == ["alpha", "beta", "gamma"]
    finally:
        engine.shutdown_async(drain=True)
