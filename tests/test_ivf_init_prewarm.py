import json
from pathlib import Path

import numpy as np
import pytest

import faiss

try:
    # In the built my_faiss package, engine is available under faiss.engine.engine.
    from faiss.engine.engine import FaissEngine, FaissEnginConfig  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import FaissEngine from faiss.engine.engine. "
        "Make sure you're running tests with the built my_faiss Python package."
    ) from exc


pytestmark = pytest.mark.skipif(
    faiss.get_num_gpus() < 1,
    reason="gpu-only test for IVF prewarm",
)


def _build_toy_ivf_index(tmp_path: Path, d: int = 8, nlist: int = 8, nb: int = 256):
    rng = np.random.RandomState(123)
    xb = rng.random((nb, d)).astype("float32")

    quantizer = faiss.IndexFlatL2(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    cpu_index.train(xb)
    cpu_index.add(xb)

    index_path = tmp_path / "toy_ivf.index"
    faiss.write_index(cpu_index, str(index_path))
    return str(index_path), cpu_index


def _basic_engine_config(index_path: str, corpus_path: str, **overrides):
    base = {
        "index_type": "IVFFlat",
        "index_path": index_path,
        "corpus_path": corpus_path,
        "retrieval_method": "multilingual-e5-large",
        "retrieval_topk": 3,
        "retrieval_batch_size": 2,
        "retrieval_query_max_length": 16,
        "retrieval_use_fp16": False,
        "retrieval_pooling_method": "mean",
        "return_embedding": False,
        "larger_topk": 5,
        "use_sentence_transformer": False,
        "gpu_memory_utilization": 0.5,
        "eviction_policy": "lru",
        "eviction_max_attempts": 8,
        "nprobe": 2,
        "embedder_model_path": "/share_data/public_models/multilingual-e5-large",
    }
    # Only pass fields that the dataclass is guaranteed to know about.
    cfg = dict(base)
    for k, v in overrides.items():
        if k not in {"ivf_init_strategy", "ivf_init_profile_path", "ivf_init_n", "mode"}:
            cfg[k] = v

    config = FaissEnginConfig(**cfg)

    # Attach optional IVF init attributes dynamically so that tests remain
    # compatible even if the underlying FaissEnginConfig definition has not
    # been regenerated yet.
    for opt_key in ("ivf_init_strategy", "ivf_init_profile_path", "ivf_init_n", "mode"):
        if opt_key in overrides:
            setattr(config, opt_key, overrides[opt_key])

    return config


def test_ivf_init_random_prewarm(tmp_path, monkeypatch):
    # Make numpy random choice deterministic.
    rng = np.random.RandomState(7)

    # Patch np.random.choice used inside FaissEngine._init_ivf_lists_random
    def _fixed_choice(a, size=None, replace=True):
        # a is expected to be a 1D array of list ids.
        a = np.asarray(a)
        assert a.ndim == 1
        if size is None:
            size = 1
        # Use a dedicated RNG to make the test deterministic.
        idx = rng.choice(len(a), size=size, replace=replace)
        return a[idx]

    monkeypatch.setattr(np.random, "choice", _fixed_choice)

    index_path, cpu_index = _build_toy_ivf_index(tmp_path)

    # Ensure transformers doesn't try to reach the network during tests.
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    # Avoid hitting the HuggingFace datasets loader in load_corpus; we only
    # care about IVF list initialization here.
    def _fake_load_corpus(_path):
        return []

    monkeypatch.setattr(
        "faiss.engine.engine.load_corpus",
        _fake_load_corpus,
        raising=False,
    )

    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("{}", encoding="utf-8")

    # We want to prewarm exactly 3 lists out of nlist=8.
    config = _basic_engine_config(
        index_path=index_path,
        corpus_path=str(corpus_path),
        # First initialize the engine on CPU without eviction/ivf_init so that
        # we can then move the index to GPU and re-run reinit() to exercise the
        # GPU list-level APIs.
        eviction_policy="none",
        ivf_init_strategy="none",
    )

    engine = FaissEngine(config)

    # Move the IVF index to GPU, then enable eviction and run IVF prewarm.
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, engine.index)
    engine.index = gpu_index

    config.eviction_policy = "lru"
    setattr(config, "ivf_init_strategy", "random")
    setattr(config, "ivf_init_n", 3)
    setattr(config, "mode", "baseline")

    engine.reinit(config)

    # The engine should record which lists were initialized.
    ivf_lists = getattr(engine, "_ivf_init_lists", None)
    assert ivf_lists is not None
    assert len(ivf_lists) == 3
    assert all(0 <= lid < cpu_index.nlist for lid in ivf_lists)
    assert all(faiss.is_list_on_gpu(engine.index, int(lid)) for lid in ivf_lists)


def test_ivf_init_topk_prewarm(tmp_path, monkeypatch):
    index_path, cpu_index = _build_toy_ivf_index(tmp_path)

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def _fake_load_corpus(_path):
        return []

    monkeypatch.setattr(
        "faiss.engine.engine.load_corpus",
        _fake_load_corpus,
        raising=False,
    )

    # Minimal dummy corpus; actual content is ignored thanks to the
    # load_corpus monkeypatch applied below.
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("{}", encoding="utf-8")

    # Construct a fake activation profile where a few specific IVF list ids
    # are clearly the most frequent.
    profile_path = tmp_path / "activate-toy.json"
    # Use string keys to match the real offline profile format.
    profile = {
        "0": 1,
        "1": 10,
        "2": 5,
        "3": 20,
        "4": 15,
        # Add an out-of-range id to ensure it is ignored.
        str(cpu_index.nlist + 5): 100,
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    # We expect the top-2 lists by frequency to be ids 3 and 4.
    config = _basic_engine_config(
        index_path=index_path,
        corpus_path=str(corpus_path),
        eviction_policy="none",
        ivf_init_strategy="none",
    )

    engine = FaissEngine(config)

    # Move the IVF index to GPU, then enable eviction and run IVF prewarm.
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, engine.index)
    engine.index = gpu_index

    config.eviction_policy = "lru"
    setattr(config, "ivf_init_strategy", "topk")
    setattr(config, "ivf_init_profile_path", str(profile_path))
    setattr(config, "ivf_init_n", 2)
    setattr(config, "mode", "baseline")

    engine.reinit(config)

    ivf_lists = getattr(engine, "_ivf_init_lists", None)
    assert ivf_lists is not None
    assert len(ivf_lists) == 2
    assert set(ivf_lists) == {3, 4}
    assert all(faiss.is_list_on_gpu(engine.index, int(lid)) for lid in ivf_lists)


def test_gpu_reservation_capacity_guard(tmp_path, monkeypatch):
    index_path, _ = _build_toy_ivf_index(tmp_path)

    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def _fake_load_corpus(_path):
        return []

    monkeypatch.setattr(
        "faiss.engine.engine.load_corpus",
        _fake_load_corpus,
        raising=False,
    )

    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("{}", encoding="utf-8")

    config = _basic_engine_config(
        index_path=index_path,
        corpus_path=str(corpus_path),
        eviction_policy="none",
        ivf_init_strategy="none",
        gpu_memory_utilization=0.8,
    )
    engine = FaissEngine(config)

    # Simulate a device with insufficient free memory for target reservation.
    monkeypatch.setattr(engine, "_get_cuda_mem_info", lambda _device_id: (10 * (1024 ** 3), 80 * (1024 ** 3)))
    monkeypatch.setattr(engine, "_get_reservation_safety_bytes", lambda: int(2 * (1024 ** 3)))

    with pytest.raises(RuntimeError, match="exceeds available free memory"):
        engine.ensure_gpu_resources(config, device_id=0)

