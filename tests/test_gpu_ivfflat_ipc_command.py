import ctypes
import mmap
import os
import struct
import uuid

import numpy as np
import pytest

import faiss
from faiss.contrib.datasets import SyntheticDataset


def _shm_open(name: str, size: int) -> int:
    if not isinstance(name, str) or not name.startswith("/"):
        raise ValueError("shm name must be a string starting with '/'")
    if size <= 0:
        raise ValueError("size must be > 0")

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    shm_open = libc.shm_open
    shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    shm_open.restype = ctypes.c_int

    fd = shm_open(name.encode("utf-8"), os.O_CREAT | os.O_RDWR, 0o666)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"shm_open failed for {name!r}")

    ftruncate = libc.ftruncate
    ftruncate.argtypes = [ctypes.c_int, ctypes.c_long]
    ftruncate.restype = ctypes.c_int
    if ftruncate(fd, size) != 0:
        err = ctypes.get_errno()
        os.close(fd)
        raise OSError(err, f"ftruncate failed for {name!r}")

    return fd


def _shm_unlink(name: str) -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    shm_unlink = libc.shm_unlink
    shm_unlink.argtypes = [ctypes.c_char_p]
    shm_unlink.restype = ctypes.c_int
    shm_unlink(name.encode("utf-8"))


def _ipc_struct_pack(magic, version, state, opcode, list_id, result) -> bytes:
    # Matches GpuIndexIVFFlat::IpcCommand layout:
    # uint32 magic, uint32 version, uint32 state, uint32 opcode, int64 listId, int64 result
    return struct.pack("=IIIIqq", magic, version, state, opcode, int(list_id), int(result))


def _ipc_struct_unpack(buf: bytes):
    magic, version, state, opcode, list_id, result = struct.unpack("=IIIIqq", buf)
    return {
        "magic": magic,
        "version": version,
        "state": state,
        "opcode": opcode,
        "list_id": list_id,
        "result": result,
    }


def _send_ipc_command(index, opcode: int, list_id: int):
    # Constants must match faiss/gpu/GpuIndexIVFFlat.h
    kIpcMagic = 0x4956464C  # "IVFL"
    kIpcVersion = 1
    kIpcStatePending = 1

    shm_name = f"/faiss_ivfflat_ipc_{uuid.uuid4().hex}"
    size = struct.calcsize("=IIIIqq")
    fd = _shm_open(shm_name, size=size)
    try:
        mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        try:
            mm.seek(0)
            mm.write(_ipc_struct_pack(kIpcMagic, kIpcVersion, kIpcStatePending, opcode, list_id, 0))
            mm.flush()

            handled = bool(index.processSharedMemoryCommand(shm_name))

            mm.seek(0)
            raw = mm.read(size)
            decoded = _ipc_struct_unpack(raw)
            return handled, decoded
        finally:
            mm.close()
    finally:
        os.close(fd)
        _shm_unlink(shm_name)


def _build_cpu_ivfflat(d=32, nb=8000, nq=50, nlist=128, nprobe=8):
    ds = SyntheticDataset(d, nb, nq, 100)
    index = faiss.index_factory(ds.d, f"IVF{nlist},Flat")
    index.train(ds.get_train())
    index.add(ds.get_database())
    index.nprobe = nprobe
    return index, ds


def _pick_non_empty_list_id(cpu_index, xq, nprobe: int) -> int:
    _, list_ids = cpu_index.quantizer.search(xq, nprobe)
    flat = np.unique(list_ids.reshape(-1))
    for lid in flat:
        if int(lid) >= 0 and cpu_index.invlists.list_size(int(lid)) > 0:
            return int(lid)
    raise RuntimeError("no non-empty IVF list id found for this workload")


@pytest.mark.skipif(faiss.get_num_gpus() < 1, reason="gpu only test")
def test_ivfflat_ipc_command_evict_load_single_gpu():
    cpu_index, ds = _build_cpu_ivfflat()
    xq = ds.get_queries()

    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = cpu_index.nprobe

    if not hasattr(gpu_index, "processSharedMemoryCommand"):
        pytest.skip("processSharedMemoryCommand not available in this build")

    list_id = _pick_non_empty_list_id(cpu_index, xq, cpu_index.nprobe)
    assert faiss.is_list_on_gpu(gpu_index, list_id)

    # EVict
    handled, out = _send_ipc_command(gpu_index, opcode=1, list_id=list_id)
    assert handled
    assert out["state"] in {2, 3}  # Done or Error
    assert out["opcode"] == 1
    assert out["list_id"] == list_id
    assert not faiss.is_list_on_gpu(gpu_index, list_id)

    # Load
    handled2, out2 = _send_ipc_command(gpu_index, opcode=2, list_id=list_id)
    assert handled2
    assert out2["state"] in {2, 3}
    assert out2["opcode"] == 2
    assert out2["list_id"] == list_id
    assert faiss.is_list_on_gpu(gpu_index, list_id)


@pytest.mark.skipif(faiss.get_num_gpus() < 2, reason="requires >= 2 GPUs")
def test_ivfflat_ipc_command_multi_gpu_shards_independent():
    cpu_index, ds = _build_cpu_ivfflat(d=32, nb=6000, nq=30, nlist=128, nprobe=8)
    xq = ds.get_queries()

    ngpu = 2
    res = [faiss.StandardGpuResources() for _ in range(ngpu)]
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.shard_type = 4
    co.common_ivf_quantizer = True
    co.use_cuvs = False
    index_gpu = faiss.index_cpu_to_gpu_multiple_py(res, cpu_index, co, list(range(ngpu)))

    # Ensure we have a sharded IndexShardsIVF to exercise per-shard IPC.
    # (If not available, skip.)
    if not (hasattr(index_gpu, "count") and hasattr(index_gpu, "at")):
        pytest.skip("multi-index container not available")

    # Pick one list and find which shard owns it.
    list_id = _pick_non_empty_list_id(cpu_index, xq, cpu_index.nprobe)

    owner = None
    for s in range(index_gpu.count()):
        shard = faiss.downcast_index(index_gpu.at(s))
        if hasattr(shard, "processSharedMemoryCommand") and shard.isListOnGpu(list_id):
            owner = shard
            break
    if owner is None:
        pytest.skip("could not locate owning shard for chosen list_id")

    handled, _out = _send_ipc_command(owner, opcode=1, list_id=list_id)
    assert handled
    assert not owner.isListOnGpu(list_id)

    handled2, _out2 = _send_ipc_command(owner, opcode=2, list_id=list_id)
    assert handled2
    assert owner.isListOnGpu(list_id)

