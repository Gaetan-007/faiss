# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# @nolint

# not linting this file because it imports * from swigfaiss, which
# causes a ton of useless warnings.

import numpy as np

from faiss.loader import *
from faiss.array_conversions import vector_to_array


###########################################
# GPU functions
###########################################


def index_cpu_to_gpu_multiple_py(resources, index, co=None, gpus=None):
    """ builds the C++ vectors for the GPU indices and the
    resources. Handles the case where the resources are assigned to
    the list of GPUs """
    if gpus is None:
        gpus = range(len(resources))
    vres = GpuResourcesVector()
    vdev = Int32Vector()
    for i, res in zip(gpus, resources):
        vdev.push_back(i)
        vres.push_back(res)
    if isinstance(index, IndexBinary):
        return index_binary_cpu_to_gpu_multiple(vres, vdev, index, co)
    else:
        return index_cpu_to_gpu_multiple(vres, vdev, index, co)


def index_cpu_to_all_gpus(index, co=None, ngpu=-1):
    index_gpu = index_cpu_to_gpus_list(index, co=co, gpus=None, ngpu=ngpu)
    return index_gpu


def index_cpu_to_gpus_list(index, co=None, gpus=None, ngpu=-1):
    """ Here we can pass list of GPU ids as a parameter or ngpu to
    use first n GPU's. gpus mut be a list or None.
    co is a GpuMultipleClonerOptions
    """
    if (gpus is None) and (ngpu == -1):  # All blank
        gpus = range(get_num_gpus())
    elif (gpus is None) and (ngpu != -1):  # Get number of GPU's only
        gpus = range(ngpu)
    res = [StandardGpuResources() for _ in gpus]
    index_gpu = index_cpu_to_gpu_multiple_py(res, index, co, gpus)
    return index_gpu

# NOTE: (wangzehao)
def _compute_ivf_list_assignment(cpu_index, ngpu: int, shard_type: int):
    """
    Compute list_ids_per_shard and list_to_shard for IVF nlist-based sharding.

    shard_type:
      - 4: contiguous nlist ranges
      - 5: greedy load-balanced bin packing by IVFLists bytes
    """
    if cpu_index is None:
        raise ValueError("cpu_index must not be None")
    if ngpu <= 0:
        raise ValueError("ngpu must be > 0")
    if not hasattr(cpu_index, "nlist") or not hasattr(cpu_index, "invlists"):
        raise TypeError("cpu_index must be an IVF index exposing nlist/invlists")

    nlist = int(getattr(cpu_index, "nlist", 0))
    if nlist <= 0:
        raise ValueError("cpu_index.nlist must be > 0")

    if shard_type == 4:
        list_ids_per_shard = []
        list_to_shard = [0] * nlist
        for s in range(ngpu):
            i0 = s * nlist // ngpu
            i1 = (s + 1) * nlist // ngpu
            ids = list(range(i0, i1))
            list_ids_per_shard.append(ids)
            for lid in ids:
                list_to_shard[int(lid)] = int(s)
        return list_ids_per_shard, list_to_shard

    if shard_type == 5:
        invlists = cpu_index.invlists
        code_size = int(getattr(invlists, "code_size", 0))
        if code_size <= 0:
            raise ValueError("cpu_index.invlists.code_size must be > 0")

        # idx_t is int64 in Python bindings; use 8 bytes for ID storage.
        id_bytes = 8

        list_bytes = []
        for lid in range(nlist):
            # NOTE: (wangzehao)
            # Do not silently mask binding/runtime failures here; incorrect sizes
            # lead to incorrect sharding decisions that are hard to debug.
            sz = int(invlists.list_size(int(lid)))
            list_bytes.append(sz * (code_size + id_bytes))

        sorted_by_size = list(range(nlist))
        sorted_by_size.sort(key=lambda lid: list_bytes[lid], reverse=True)

        shard_bytes = [0] * ngpu
        list_ids_per_shard = [[] for _ in range(ngpu)]
        list_to_shard = [0] * nlist

        for lid in sorted_by_size:
            # Assign to shard with minimal current bytes.
            best = 0
            for s in range(1, ngpu):
                if shard_bytes[s] < shard_bytes[best]:
                    best = s
            list_ids_per_shard[best].append(int(lid))
            list_to_shard[int(lid)] = int(best)
            shard_bytes[best] += int(list_bytes[lid])

        return list_ids_per_shard, list_to_shard

    raise ValueError(f"unsupported shard_type: {shard_type!r} (expected 4 or 5)")


def build_sharded_ivfflat_cpu_offload_index(
    cpu_index,
    *,
    ngpu: int = -1,
    gpus=None,
    resources=None,
    shard_type: int = 5,
    preload_list_ids_per_shard=None,
    use_cuvs: bool = False,
):
    """
    Build a multi-GPU sharded IndexShardsIVF where each shard is a GpuIndexIVFFlat
    initialized via copyFromSelective from a per-shard CPU IndexIVFFlat backing.

    This enables IvfListMissPolicy::CpuOffload to work correctly under
    IndexShardsIVF::search (which calls search_preassigned on each shard),
    because each shard has a valid external CPU backing for its owned lists.

    Returns
    -------
    index_shards : IndexShardsIVF
        Multi-GPU sharded index
    cpu_backings : list[IndexIVFFlat]
        Per-shard CPU backing indices (kept alive via referenced_objects)
    list_ids_per_shard : list[list[int]]
        Owned list IDs per shard
    list_to_shard : list[int]
        Mapping list_id -> shard_id
    """
    if cpu_index is None:
        raise ValueError("cpu_index must not be None")

    if ngpu == -1 and gpus is None:
        ngpu = get_num_gpus()
    if gpus is None:
        if ngpu <= 0:
            raise ValueError("ngpu must be > 0 when gpus is None")
        gpus = list(range(int(ngpu)))
    else:
        gpus = list(gpus)
        if not gpus:
            raise ValueError("gpus must not be empty")
        ngpu = len(gpus)

    if get_num_gpus() < ngpu:
        raise RuntimeError(
            f"requested ngpu={ngpu}, but only {get_num_gpus()} GPUs are available"
        )

    list_ids_per_shard, list_to_shard = _compute_ivf_list_assignment(
        cpu_index, ngpu=ngpu, shard_type=int(shard_type)
    )

    if preload_list_ids_per_shard is None:
        preload_list_ids_per_shard = [[] for _ in range(ngpu)]
    if len(preload_list_ids_per_shard) != ngpu:
        raise ValueError(
            "preload_list_ids_per_shard length must match number of shards"
        )

    if resources is None:
        resources = [StandardGpuResources() for _ in range(ngpu)]
    if len(resources) != ngpu:
        raise ValueError("resources length must match number of shards")

    if not hasattr(cpu_index, "d") or not hasattr(cpu_index, "nlist"):
        raise TypeError("cpu_index must expose d and nlist")

    d = int(getattr(cpu_index, "d", 0))
    nlist = int(getattr(cpu_index, "nlist", 0))
    metric = int(getattr(cpu_index, "metric_type", 1))

    if not hasattr(cpu_index, "quantizer"):
        raise TypeError("cpu_index must expose quantizer")

    quantizer = cpu_index.quantizer
    index_shards = IndexShardsIVF(quantizer, nlist, True, False)

    cpu_backings = []
    for shard_id, dev in enumerate(gpus):
        # Per-shard CPU backing index (contains only owned IVF lists).
        cpu_shard = IndexIVFFlat(quantizer, d, nlist, metric)
        cpu_shard.nprobe = int(getattr(cpu_index, "nprobe", 1))
        cpu_shard.is_trained = bool(getattr(cpu_index, "is_trained", True))

        owned = list_ids_per_shard[shard_id]
        if owned:
            v_owned = Int64Vector()
            for lid in owned:
                v_owned.push_back(int(lid))
            cpu_index.copy_lists_to(cpu_shard, v_owned)

        cpu_backings.append(cpu_shard)

        cfg = GpuIndexIVFFlatConfig()
        cfg.device = int(dev)
        cfg.use_cuvs = bool(use_cuvs)

        gpu_shard = GpuIndexIVFFlat(resources[shard_id], d, nlist, metric, cfg)

        preload_ids = preload_list_ids_per_shard[shard_id] or []
        init_ivf_lists_from_cpu(gpu_shard, cpu_shard, preload_ids)
        gpu_shard.nprobe = int(getattr(cpu_index, "nprobe", 1))

        index_shards.add_shard(gpu_shard)

    # Enable evict/load forwarding on the sharded index.
    if hasattr(index_shards, "setListToShardMapping"):
        v_map = Int32Vector()
        for owner in list_to_shard:
            v_map.push_back(int(owner))
        index_shards.setListToShardMapping(v_map)

    # Keep CPU backings alive for the lifetime of the sharded index.
    if not hasattr(index_shards, "referenced_objects"):
        index_shards.referenced_objects = []
    index_shards.referenced_objects.extend(cpu_backings)

    return index_shards, cpu_backings, list_ids_per_shard, list_to_shard

# NOTE: (wangzehao) This function is used to convert a list of list IDs to a Int64Vector
def _to_int64_vector(list_ids):
    v = Int64Vector()
    if np.isscalar(list_ids):
        v.push_back(int(list_ids))
        return v
    for val in np.asarray(list_ids, dtype=np.int64).ravel():
        v.push_back(int(val))
    return v

# NOTE: (wangzehao) This function is used to convert a UInt64Vector to a numpy array using efficient memory copy
def _uint64_vector_to_numpy(vec):
    """Convert UInt64Vector to numpy array using efficient memory copy."""
    return vector_to_array(vec)

# NOTE: (wangzehao) This function is used to evict a single IVF list (centroid) to CPU memory and free GPU memory
def evict_ivf_lists(index, list_ids):
    """Evict IVF lists (centroids) to CPU memory and return reclaimed bytes."""
    if np.isscalar(list_ids):
        return index.evictCentroidToCpu(int(list_ids))
    v = _to_int64_vector(list_ids)
    out = index.evictCentroidsToCpu(v)
    return _uint64_vector_to_numpy(out)

# NOTE: (wangzehao) This function is used to load a single IVF list (centroid) from CPU memory back to GPU
def load_ivf_lists(index, list_ids):
    """Load IVF lists (centroids) back to GPU and return loaded bytes."""
    if np.isscalar(list_ids):
        return index.loadCentroidToGpu(int(list_ids))
    v = _to_int64_vector(list_ids)
    out = index.loadCentroidsToGpu(v)
    return _uint64_vector_to_numpy(out)


def init_ivf_lists_from_cpu(index, cpu_index, list_ids):
    """
    Initialize a GPU IVF-Flat index by loading only selected IVF lists.

    Non-loaded lists with data are cached in CPU memory for on-demand loading.
    """
    if not hasattr(index, "copyFromSelective"):
        raise RuntimeError(
            "copyFromSelective not available; rebuild Python bindings with GPU support"
        )
    v = _to_int64_vector(list_ids)
    index.copyFromSelective(cpu_index, v)


###########################################
# Page-fault style auto-fetch management
# NOTE: (wangzehao) Below functions implement automatic load-on-demand
###########################################

def set_auto_fetch(index, enable):
    """
    Enable or disable automatic fetching of evicted lists during search.
    
    When enabled, search operations will automatically load any IVF lists
    that are needed but currently evicted (in CPU cache) - similar to
    a page fault handler.
    
    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index
    enable : bool
        True to enable auto-fetch, False to disable
    """
    index.setAutoFetch(enable)


def is_auto_fetch_enabled(index):
    """
    Check if auto-fetch is currently enabled.
    
    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index
        
    Returns
    -------
    bool
        True if auto-fetch is enabled
    """
    return index.isAutoFetchEnabled()


def set_no_copy_evict(index, enable):
    """
    Enable or disable no-copy eviction for IVF lists that have valid
    external CPU backing. When enabled, eligible lists can be evicted
    without issuing a GPU->CPU copy; eviction will rely on metadata
    recorded inside the GpuIndexIVFFlat instance (for example when
    constructed via copyFromSelective).

    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index
    enable : bool
        True to enable no-copy eviction, False to disable
    """
    index.setNoCopyEvictEnabled(bool(enable))


def is_no_copy_evict_enabled(index):
    """
    Check if no-copy eviction is currently enabled.

    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index

    Returns
    -------
    bool
        True if no-copy eviction is enabled
    """
    return index.isNoCopyEvictEnabled()


def is_list_on_gpu(index, list_id):
    """
    Check if a single IVF list (centroid) is currently on GPU.
    
    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index
    list_id : int
        The list ID to check
        
    Returns
    -------
    bool
        True if the list data is on GPU, False if evicted
    """
    return index.isListOnGpu(int(list_id))


def get_evicted_lists(index):
    """
    Get the set of lists that are currently evicted (in CPU cache).
    
    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index
        
    Returns
    -------
    numpy.ndarray
        Array of list IDs that are currently evicted
    """
    vec = index.getEvictedLists()
    return vector_to_array(vec)


def get_auto_fetch_stats(index):
    """
    Get statistics about auto-fetch operations (for debugging/profiling).
    
    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'total_fetches': Total number of auto-fetch operations triggered
        - 'total_lists_fetched': Total number of lists fetched
        - 'total_bytes_fetched': Total bytes fetched from CPU
    """
    try:
        stats_vec = GpuIndexIVFFlat_getAutoFetchStatsVector(index)
    except NameError as exc:
        raise RuntimeError(
            "GpuIndexIVFFlat_getAutoFetchStatsVector is not available; "
            "rebuild Python bindings with GPU support"
        ) from exc
    if stats_vec.size() != 3:
        raise RuntimeError(
            f"Unexpected auto-fetch stats length: {stats_vec.size()}")
    return {
        'total_fetches': int(stats_vec.at(0)),
        'total_lists_fetched': int(stats_vec.at(1)),
        'total_bytes_fetched': int(stats_vec.at(2)),
    }


def reset_auto_fetch_stats(index):
    """
    Reset auto-fetch statistics.
    
    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index
    """
    index.resetAutoFetchStats()


def get_list_activation_stats(index):
    """
    Get per-list IVF activation statistics from a GPU IVF-Flat index.

    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index

    Returns
    -------
    list of dict
        Each element has keys:
        - 'list_id': int
        - 'probe_count': int
        - 'load_count': int
        - 'last_probe_ts': int (monotonic timestamp)
    """
    def _single_ivfflat_stats(ivfflat):
        try:
            flat_vec = GpuIndexIVFFlat_getActivationStatsVector(ivfflat)
        except NameError as exc:
            raise RuntimeError(
                "GpuIndexIVFFlat_getActivationStatsVector is not available; "
                "rebuild Python bindings with GPU support"
            ) from exc

        arr = vector_to_array(flat_vec)
        if arr.size % 4 != 0:
            raise RuntimeError(
                f"Unexpected activation stats vector length: {arr.size}"
            )

        stats = []
        for i in range(0, arr.size, 4):
            stats.append(
                {
                    "list_id": int(arr[i]),
                    "probe_count": int(arr[i + 1]),
                    "load_count": int(arr[i + 2]),
                    "last_probe_ts": int(arr[i + 3]),
                }
            )
        return stats

    # IndexShardsIVF / multi-index case: aggregate per-shard stats.
    if hasattr(index, "count") and hasattr(index, "at"):
        merged = {}
        for s in range(int(index.count())):
            shard = downcast_index(index.at(s))
            if shard is None:
                continue
            if not hasattr(shard, "getActivationStatsFlatVector"):
                # Not a GpuIndexIVFFlat shard; skip.
                continue
            for st in _single_ivfflat_stats(shard):
                lid = int(st.get("list_id", -1))
                if lid < 0:
                    continue
                prev = merged.get(lid)
                if prev is None:
                    merged[lid] = dict(st)
                else:
                    prev["probe_count"] = int(prev.get("probe_count", 0)) + int(
                        st.get("probe_count", 0)
                    )
                    prev["load_count"] = int(prev.get("load_count", 0)) + int(
                        st.get("load_count", 0)
                    )
                    prev["last_probe_ts"] = max(
                        int(prev.get("last_probe_ts", 0)),
                        int(st.get("last_probe_ts", 0)),
                    )
        return [merged[k] for k in sorted(merged.keys())]

    return _single_ivfflat_stats(index)


def reset_list_activation_stats(index):
    """
    Reset all per-list IVF activation statistics on a GPU IVF-Flat index.

    Parameters
    ----------
    index : GpuIndexIVFFlat
        The GPU IVF Flat index
    """
    def _single_reset(ivfflat):
        # Newer builds expose resetActivationStats as a member function.
        if hasattr(ivfflat, "resetActivationStats"):
            ivfflat.resetActivationStats()
        else:
            raise RuntimeError(
                "resetActivationStats is not available on this index; "
                "rebuild Python bindings with GPU support"
            )

    # IndexShardsIVF / multi-index case: reset per-shard.
    if hasattr(index, "count") and hasattr(index, "at"):
        any_shard = False
        for s in range(int(index.count())):
            shard = downcast_index(index.at(s))
            if shard is None:
                continue
            if not hasattr(shard, "resetActivationStats"):
                continue
            any_shard = True
            _single_reset(shard)
        if any_shard:
            return

    _single_reset(index)


# allows numpy ndarray usage with bfKnn


def knn_gpu(res, xq, xb, k, D=None, I=None, metric=METRIC_L2, device=-1, use_cuvs=False, vectorsMemoryLimit=0, queriesMemoryLimit=0):
    """
    Compute the k nearest neighbors of a vector on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    xq : array_like
        Query vectors, shape (nq, d) where d is appropriate for the index.
        `dtype` must be float32.
    xb : array_like
        Database vectors, shape (nb, d) where d is appropriate for the index.
        `dtype` must be float32.
    k : int
        Number of nearest neighbors.
    D : array_like, optional
        Output array for distances of the nearest neighbors, shape (nq, k)
    I : array_like, optional
        Output array for the nearest neighbors, shape (nq, k)
    metric : MetricType, optional
        Distance measure to use (either METRIC_L2 or METRIC_INNER_PRODUCT)
    device: int, optional
        Which CUDA device in the system to run the search on. -1 indicates that
        the current thread-local device state (via cudaGetDevice) should be used
        (can also be set via torch.cuda.set_device in PyTorch)
        Otherwise, an integer 0 <= device < numDevices indicates the GPU on which
        the computation should be run
    vectorsMemoryLimit: int, optional
    queriesMemoryLimit: int, optional
        Memory limits for vectors and queries.
        If not 0, the GPU will use at most this amount of memory
        for vectors and queries respectively.
        Vectors are broken up into chunks of size vectorsMemoryLimit,
        and queries are broken up into chunks of size queriesMemoryLimit,
        including the memory required for the results.

    Returns
    -------
    D : array_like
        Distances of the nearest neighbors, shape (nq, k)
    I : array_like
        Labels of the nearest neighbors, shape (nq, k)
    """
    nq, d = xq.shape
    if xq.flags.c_contiguous:
        xq_row_major = True
    elif xq.flags.f_contiguous:
        xq = xq.T
        xq_row_major = False
    else:
        xq = np.ascontiguousarray(xq, dtype='float32')
        xq_row_major = True

    xq_ptr = swig_ptr(xq)

    if xq.dtype == np.float32:
        xq_type = DistanceDataType_F32
    elif xq.dtype == np.float16:
        xq_type = DistanceDataType_F16
    else:
        raise TypeError('xq must be f32 or f16')

    nb, d2 = xb.shape
    assert d2 == d
    if xb.flags.c_contiguous:
        xb_row_major = True
    elif xb.flags.f_contiguous:
        xb = xb.T
        xb_row_major = False
    else:
        xb = np.ascontiguousarray(xb, dtype='float32')
        xb_row_major = True

    xb_ptr = swig_ptr(xb)

    if xb.dtype == np.float32:
        xb_type = DistanceDataType_F32
    elif xb.dtype == np.float16:
        xb_type = DistanceDataType_F16
    else:
        raise TypeError('xb must be float32 or float16')

    if D is None:
        D = np.empty((nq, k), dtype=np.float32)
    else:
        assert D.shape == (nq, k)
        # interface takes void*, we need to check this
        assert D.dtype == np.float32

    D_ptr = swig_ptr(D)

    if I is None:
        I = np.empty((nq, k), dtype=np.int64)
    else:
        assert I.shape == (nq, k)

    I_ptr = swig_ptr(I)

    if I.dtype == np.int64:
        I_type = IndicesDataType_I64
    elif I.dtype == I.dtype == np.int32:
        I_type = IndicesDataType_I32
    else:
        raise TypeError('I must be i64 or i32')

    args = GpuDistanceParams()
    args.metric = metric
    args.k = k
    args.dims = d
    args.vectors = xb_ptr
    args.vectorsRowMajor = xb_row_major
    args.vectorType = xb_type
    args.numVectors = nb
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    args.outIndicesType = I_type
    args.device = device
    args.use_cuvs = use_cuvs

    # no stream synchronization needed, inputs and outputs are guaranteed to
    # be on the CPU (numpy arrays)
    if vectorsMemoryLimit > 0 or queriesMemoryLimit > 0:
        bfKnn_tiling(res, args, vectorsMemoryLimit, queriesMemoryLimit)
    else:
        bfKnn(res, args)

    return D, I

# allows numpy ndarray usage with bfKnn for all pairwise distances


def pairwise_distance_gpu(res, xq, xb, D=None, metric=METRIC_L2, device=-1):
    """
    Compute all pairwise distances between xq and xb on one GPU without constructing an index

    Parameters
    ----------
    res : StandardGpuResources
        GPU resources to use during computation
    xq : array_like
        Query vectors, shape (nq, d) where d is appropriate for the index.
        `dtype` must be float32.
    xb : array_like
        Database vectors, shape (nb, d) where d is appropriate for the index.
        `dtype` must be float32.
    D : array_like, optional
        Output array for all pairwise distances, shape (nq, nb)
    metric : MetricType, optional
        Distance measure to use (either METRIC_L2 or METRIC_INNER_PRODUCT)
    device: int, optional
        Which CUDA device in the system to run the search on. -1 indicates that
        the current thread-local device state (via cudaGetDevice) should be used
        (can also be set via torch.cuda.set_device in PyTorch)
        Otherwise, an integer 0 <= device < numDevices indicates the GPU on which
        the computation should be run

    Returns
    -------
    D : array_like
        All pairwise distances, shape (nq, nb)
    """
    nq, d = xq.shape
    if xq.flags.c_contiguous:
        xq_row_major = True
    elif xq.flags.f_contiguous:
        xq = xq.T
        xq_row_major = False
    else:
        raise TypeError(
            'xq matrix should be row (C) or column-major (Fortran)')

    xq_ptr = swig_ptr(xq)

    if xq.dtype == np.float32:
        xq_type = DistanceDataType_F32
    elif xq.dtype == np.float16:
        xq_type = DistanceDataType_F16
    else:
        xq = np.ascontiguousarray(xb, dtype='float32')
        xq_row_major = True

    nb, d2 = xb.shape
    assert d2 == d
    if xb.flags.c_contiguous:
        xb_row_major = True
    elif xb.flags.f_contiguous:
        xb = xb.T
        xb_row_major = False
    else:
        xb = np.ascontiguousarray(xb, dtype='float32')
        xb_row_major = True

    xb_ptr = swig_ptr(xb)

    if xb.dtype == np.float32:
        xb_type = DistanceDataType_F32
    elif xb.dtype == np.float16:
        xb_type = DistanceDataType_F16
    else:
        raise TypeError('xb must be float32 or float16')

    if D is None:
        D = np.empty((nq, nb), dtype=np.float32)
    else:
        assert D.shape == (nq, nb)
        # interface takes void*, we need to check this
        assert D.dtype == np.float32

    D_ptr = swig_ptr(D)

    args = GpuDistanceParams()
    args.metric = metric
    args.k = -1  # selects all pairwise distances
    args.dims = d
    args.vectors = xb_ptr
    args.vectorsRowMajor = xb_row_major
    args.vectorType = xb_type
    args.numVectors = nb
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.queryType = xq_type
    args.numQueries = nq
    args.outDistances = D_ptr
    args.device = device

    # no stream synchronization needed, inputs and outputs are guaranteed to
    # be on the CPU (numpy arrays)
    bfKnn(res, args)

    return D
