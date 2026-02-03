import gc
import multiprocessing as mp

import numpy as np
import pytest

import faiss

try:
    from faiss.gpu_pool_controller import (  # type: ignore[reportMissingImports]
        GpuPoolController,
        ResizeStatus,
        get_pool_controller,
    )
except ImportError as exc:
    try:
        from faiss.python.gpu_pool_controller import (  # type: ignore[reportMissingImports]
            GpuPoolController,
            ResizeStatus,
            get_pool_controller,
        )
    except ImportError:
        raise exc


def _create_gpu_pool(reserved_bytes: int, device_id: int = 0, dim: int = 64):
    res = faiss.StandardGpuResources()
    res.setTempMemory(0)
    res.setDeviceMemoryReservation(reserved_bytes)

    config = faiss.GpuIndexFlatConfig()
    config.device = device_id
    if hasattr(config, "use_cuvs"):
        config.use_cuvs = False

    index = faiss.GpuIndexFlatL2(res, dim, config)
    xb = np.random.RandomState(123).rand(1024, dim).astype("float32")
    index.add(xb)
    return res, index


def _serialize_result(result):
    return {
        "status": int(result["status"]),
        "actual_size": int(result["actual_size"]),
        "available": int(result["available"]),
        "error": result["error"],
    }


def _ipc_child_roundtrip(
    device_id: int,
    initial_size: int,
    expand_delta: int,
    timeout_ms: int,
    queue: mp.Queue,
):
    try:
        with GpuPoolController(device_id) as ctrl:
            query = _serialize_result(ctrl.query(timeout_ms))
            noop_expand = _serialize_result(ctrl.expand(initial_size, timeout_ms))
            expanded = _serialize_result(ctrl.expand_by(expand_delta, timeout_ms))
            shrunk = _serialize_result(ctrl.shrink(initial_size, timeout_ms))
        queue.put(
            {
                "query": query,
                "noop_expand": noop_expand,
                "expanded": expanded,
                "shrunk": shrunk,
            }
        )
    except Exception as exc:  # pragma: no cover - debug path
        queue.put({"error": repr(exc)})


@pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="gpu only test")
def test_ipc_controller_basic_query_and_stats():
    reserved_bytes = 32 * 1024 * 1024
    res, index = _create_gpu_pool(reserved_bytes)
    try:
        with GpuPoolController(0) as ctrl:
            result = ctrl.query()
            assert result["status"] == ResizeStatus.SUCCESS
            assert result["actual_size"] >= reserved_bytes
            assert 0 <= result["available"] <= result["actual_size"]
            assert result["error"] == ""

            stats = ctrl.get_stats()
            assert stats["total_bytes"] == result["actual_size"]
            assert stats["available_bytes"] == result["available"]
            assert stats["used_bytes"] == (
                result["actual_size"] - result["available"]
            )
            assert 0.0 <= stats["utilization"] <= 1.0

            noop = ctrl.expand(result["actual_size"] - 1)
            assert noop["status"] == ResizeStatus.SUCCESS
            assert noop["actual_size"] == result["actual_size"]

            with pytest.raises(ValueError):
                ctrl.expand(-1)
            with pytest.raises(ValueError):
                ctrl.expand_by(-1)
            with pytest.raises(ValueError):
                ctrl.shrink(-1)
    finally:
        del index
        del res
        gc.collect()


@pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="gpu only test")
def test_ipc_expand_shrink_across_process():
    reserved_bytes = 32 * 1024 * 1024
    res, index = _create_gpu_pool(reserved_bytes)
    try:
        with GpuPoolController(0) as ctrl:
            initial = ctrl.query()
        initial_size = initial["actual_size"]

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_ipc_child_roundtrip,
            args=(0, initial_size, 1, 5000, queue),
        )
        proc.start()
        proc.join(timeout=15)
        assert proc.exitcode == 0

        result = queue.get(timeout=5)
        if "error" in result:
            pytest.fail(result["error"])

        expanded = result["expanded"]
        if (
            expanded["status"] == int(ResizeStatus.FAILED)
            and "cudaMalloc failed" in expanded["error"]
        ):
            pytest.skip("GPU memory too constrained for IPC expansion")

        assert result["query"]["status"] == int(ResizeStatus.SUCCESS)
        assert result["noop_expand"]["status"] == int(ResizeStatus.SUCCESS)
        assert expanded["status"] in {
            int(ResizeStatus.SUCCESS),
            int(ResizeStatus.PARTIAL),
        }
        assert expanded["actual_size"] >= initial_size
        assert expanded["available"] <= expanded["actual_size"]

        shrunk = result["shrunk"]
        assert shrunk["status"] in {
            int(ResizeStatus.SUCCESS),
            int(ResizeStatus.PARTIAL),
        }
        assert shrunk["actual_size"] <= expanded["actual_size"]
        assert shrunk["actual_size"] >= initial_size

        with GpuPoolController(0) as ctrl:
            after = ctrl.query()
        assert after["actual_size"] == shrunk["actual_size"]
    finally:
        del index
        del res
        gc.collect()


def test_get_pool_controller_invalid_device_returns_none():
    invalid_device = faiss.get_num_gpus() + 100
    assert get_pool_controller(invalid_device) is None
