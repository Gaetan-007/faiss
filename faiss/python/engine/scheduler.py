from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import threading
import time
from typing import Deque, Dict, Iterable, List, Optional


class SchedulerError(RuntimeError):
    pass


@dataclass(frozen=True)
class SchedulerRequest:
    id: str
    query: str
    num: Optional[int]
    return_score: bool
    enqueue_time: float
    source_id: str
    future: Future


class BaseScheduler(ABC):
    def __init__(self, max_batch_size: int, max_wait_ms: int, step_interval_ms: int):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if step_interval_ms < 0:
            raise ValueError("step_interval_ms must be >= 0")

        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._step_interval_s = step_interval_ms / 1000.0
        self._next_step_ts = time.monotonic()
        self._lock = threading.Lock()

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def max_wait_s(self) -> float:
        return self._max_wait_s

    @property
    def step_interval_s(self) -> float:
        return self._step_interval_s

    @abstractmethod
    def add_request(self, request: SchedulerRequest) -> None:
        raise NotImplementedError("add_request must be implemented")

    @abstractmethod
    def pending_count(self) -> int:
        raise NotImplementedError("pending_count must be implemented")

    @abstractmethod
    def _peek_oldest_enqueue_time(self) -> Optional[float]:
        raise NotImplementedError("_peek_oldest_enqueue_time must be implemented")

    def should_step(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.monotonic()
        # When step_interval_ms=0, skip the timestamp check to allow immediate stepping
        if self._step_interval_s > 0:
            if now < self._next_step_ts:
                return False
        if self.pending_count() == 0:
            return False
        oldest = self._peek_oldest_enqueue_time()
        if oldest is None:
            return False
        if self.pending_count() >= self._max_batch_size:
            return True
        # When max_wait_ms=0, we still need to respect step_interval_ms to allow batching.
        # If step_interval_ms=0, we require at least one step interval to have passed
        # since the oldest request was enqueued to allow multiple requests to accumulate.
        if self._max_wait_s == 0:
            # Use step_interval_ms as minimum wait time when max_wait_ms=0
            # This allows multiple requests submitted quickly to be batched together
            min_wait_s = max(self._step_interval_s, 0.001)  # At least 1ms
            return (now - oldest) >= min_wait_s
        return (now - oldest) >= self._max_wait_s

    def step(self, now: Optional[float] = None) -> List[SchedulerRequest]:
        if now is None:
            now = time.monotonic()
        if not self.should_step(now):
            return []
        batch = self._pop_batch(now, self._max_batch_size)
        # Only update _next_step_ts if step_interval_ms > 0 to avoid timing issues
        if self._step_interval_s > 0:
            self._next_step_ts = now + self._step_interval_s
        else:
            # When step_interval_ms=0, set to current time to allow immediate next step
            self._next_step_ts = now
        return batch

    def flush(self) -> List[SchedulerRequest]:
        return self._pop_batch(time.monotonic(), self.pending_count())

    @abstractmethod
    def _pop_batch(self, now: float, max_count: int) -> List[SchedulerRequest]:
        raise NotImplementedError("_pop_batch must be implemented")


class FifoScheduler(BaseScheduler):
    def __init__(self, max_batch_size: int, max_wait_ms: int, step_interval_ms: int):
        super().__init__(max_batch_size, max_wait_ms, step_interval_ms)
        self._queue: Deque[SchedulerRequest] = deque()

    def add_request(self, request: SchedulerRequest) -> None:
        if request is None:
            raise ValueError("request must not be None")
        with self._lock:
            self._queue.append(request)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def _peek_oldest_enqueue_time(self) -> Optional[float]:
        with self._lock:
            return self._queue[0].enqueue_time if self._queue else None

    def _pop_batch(self, now: float, max_count: int) -> List[SchedulerRequest]:
        if max_count <= 0:
            return []
        batch: List[SchedulerRequest] = []
        with self._lock:
            while self._queue and len(batch) < max_count:
                batch.append(self._queue.popleft())
        return batch


class RoundRobinScheduler(BaseScheduler):
    def __init__(self, max_batch_size: int, max_wait_ms: int, step_interval_ms: int):
        super().__init__(max_batch_size, max_wait_ms, step_interval_ms)
        self._queues: Dict[str, Deque[SchedulerRequest]] = {}
        self._source_order: Deque[str] = deque()

    def add_request(self, request: SchedulerRequest) -> None:
        if request is None:
            raise ValueError("request must not be None")
        with self._lock:
            if request.source_id not in self._queues:
                self._queues[request.source_id] = deque()
                self._source_order.append(request.source_id)
            self._queues[request.source_id].append(request)

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(queue) for queue in self._queues.values())

    def _peek_oldest_enqueue_time(self) -> Optional[float]:
        with self._lock:
            oldest = None
            for queue in self._queues.values():
                if not queue:
                    continue
                ts = queue[0].enqueue_time
                if oldest is None or ts < oldest:
                    oldest = ts
            return oldest

    def _pop_batch(self, now: float, max_count: int) -> List[SchedulerRequest]:
        if max_count <= 0:
            return []
        batch: List[SchedulerRequest] = []
        with self._lock:
            while self._source_order and len(batch) < max_count:
                source_id = self._source_order.popleft()
                queue = self._queues.get(source_id)
                if not queue:
                    continue
                batch.append(queue.popleft())
                if queue:
                    self._source_order.append(source_id)
                else:
                    self._queues.pop(source_id, None)
        return batch
