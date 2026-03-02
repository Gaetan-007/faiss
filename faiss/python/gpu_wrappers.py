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
