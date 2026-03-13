#!/usr/bin/env python3
"""
Demo: IVF Search Server

This demo demonstrates how to:
1. Create and train an IVF index on sample data
2. Start a FastAPI server with the IVF index
3. Query the server via HTTP API
4. (Optional) Manage GPU IVF lists (evict/load) if GPU is available

Prerequisites:
    - Faiss must be installed (e.g., via pip install -e faiss/python)
    - Run this script with the Python interpreter that has faiss installed
    - If using a virtual environment, activate it first or use its Python

Usage:
    # Stage 0: Create and train index
    python demos/demo_ivf_server.py 0

    # Stage 1: Start server (CPU mode)
    python demos/demo_ivf_server.py 1

    # Stage 2: Start server (GPU mode with eviction policy)
    python demos/demo_ivf_server.py 2

    # Stage 3: Query the server (in another terminal)
    python demos/demo_ivf_server.py 3

    # Stage 4: Manage IVF lists (evict/load) - GPU only
    python demos/demo_ivf_server.py 4
"""

import sys
import os
import time
import json
import numpy as np
import faiss
import argparse
from pathlib import Path

# Import server modules from installed faiss package
# Note: Do NOT modify sys.path here - use the installed faiss package
try:
    from faiss.server import create_app
    from faiss.server.serve_config import ServerConfig
except ImportError as e:
    print("=" * 60)
    print("ERROR: Failed to import faiss.server module")
    print("=" * 60)
    print(f"\nImport error: {e}")
    print("\nPossible causes:")
    print("  1. Faiss is not installed. Install with:")
    print("     cd faiss/python && pip install -e .")
    print("  2. You're using the wrong Python interpreter.")
    print("     Current Python:", sys.executable)
    print("  3. If using a virtual environment, activate it first:")
    print("     source .venv/bin/activate")
    print("\nTo verify installation:")
    print("  python -c \"from faiss.server import create_app; print('OK')\"")
    print("=" * 60)
    sys.exit(1)

# Configuration
TMP_DIR = "/share_data/public_data"
DIM = 1024
NUM_VECTORS = 10000
NCLUSTERS = 64
NUM_QUERIES = 5
TOP_K = 10


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_random_data(n, d):
    """Generate random vectors."""
    return np.random.randn(n, d).astype('float32')


def generate_corpus(n):
    """Generate sample corpus documents."""
    return [
        {
            "id": i,
            "text": f"Sample document {i} with random content for testing IVF search",
            "title": f"Doc {i}"
        }
        for i in range(n)
    ]


def stage_0_create_index():
    """
    Stage 0: Create and train an IVF index.

    This creates:
    - A trained IVF index file
    - Sample corpus data
    - Query vectors for testing
    """
    print("=" * 60)
    print("Stage 0: Creating and Training IVF Index")
    print("=" * 60)

    ensure_dir(TMP_DIR)

    # Generate training data
    print(f"Generating {NUM_VECTORS} random vectors of dimension {DIM}...")
    train_data = generate_random_data(NUM_VECTORS, DIM)

    # Create IVF index
    print(f"Creating IVF index with {NCLUSTERS} clusters...")
    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, NCLUSTERS)

    # Train index
    print("Training index...")
    index.train(train_data)

    # Add vectors
    print("Adding vectors to index...")
    index.add(train_data)

    # Save index
    index_path = os.path.join(TMP_DIR, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"Index saved to: {index_path}")

    # Generate and save corpus
    print("Generating sample corpus...")
    corpus = generate_corpus(NUM_VECTORS)
    corpus_path = os.path.join(TMP_DIR, "corpus.jsonl")
    with open(corpus_path, 'w') as f:
        for doc in corpus:
            f.write(json.dumps(doc) + '\n')
    print(f"Corpus saved to: {corpus_path}")

    # Generate and save query vectors
    print("Generating sample queries...")
    queries = generate_random_data(NUM_QUERIES, DIM)
    query_path = os.path.join(TMP_DIR, "queries.npy")
    np.save(query_path, queries)
    print(f"Queries saved to: {query_path}")

    # Save text queries for HTTP API testing
    text_queries = [
        "machine learning",
        "deep learning",
        "neural networks",
        "natural language processing",
        "computer vision"
    ]
    text_query_path = os.path.join(TMP_DIR, "text_queries.txt")
    with open(text_query_path, 'w') as f:
        for q in text_queries:
            f.write(q + '\n')
    print(f"Text queries saved to: {text_query_path}")

    print("\nStage 0 complete!")
    print(f"Files created in {TMP_DIR}:")
    for f in os.listdir(TMP_DIR):
        print(f"  - {f}")


def stage_1_start_server_cpu():
    """
    Stage 1: Start the server in CPU mode.
    """
    print("=" * 60)
    print("Stage 1: Starting IVF Server (CPU Mode)")
    print("=" * 60)

    # Create server configuration
    config_dict = {
        "host": "0.0.0.0",
        "port": 9090,
        "workers": 1,
        "log_level": "info",
        "enable_request_logging": True,
        "log_dir": "/share_data/wangzehao/.faiss_log",
        "log_format": "json",
        "api_prefix": "/api/v1",
        "engine_config": {
            "index_type": "IVFFlat",
            "index_path": os.path.join(TMP_DIR, "browsecomp-plus-ivf-rebuilt", "e5_IVF2048,Flat.index"),
            "corpus_path": os.path.join(TMP_DIR, "browsecomp-plus-corpus", "browsecomp-plus-chunk-256.jsonl"),
            "retrieval_method": "e5",
            "retrieval_topk": TOP_K,
            "retrieval_batch_size": 32,
            "retrieval_model_path": "/share_data/public_model/multilingual-e5-large",
            "retrieval_query_max_length": 512,
            "retrieval_use_fp16": False,
            "retrieval_pooling_method": "mean",
            "use_sentence_transformer": False,
            "return_embedding": False,
            "gpu_memory_utilization": 0.8,
            "eviction_policy": None,  # No eviction in CPU mode
            "nprobe": 16,
            "gpu_enabled": False,  # CPU mode
        }
    }

    config = ServerConfig.from_dict(config_dict, validate=False)
    app = create_app(config)

    import uvicorn
    print("\nServer starting on http://0.0.0.0:9090")
    print("Available endpoints:")
    print("  GET  /api/v1/health       - Health check")
    print("  GET  /api/v1/ready        - Readiness check")
    print("  POST /api/v1/search       - Search with text query")
    print("  POST /api/v1/batch_search - Batch search")
    print("  GET  /api/v1/stats        - Server statistics")
    print("\nPress Ctrl+C to stop the server\n")

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


def stage_2_start_server_gpu():
    """
    Stage 2: Start the server with GPU mode and eviction policy.
    """
    print("=" * 60)
    print("Stage 2: Starting IVF Server (GPU Mode with Eviction)")
    print("=" * 60)

    if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() <= 0:
        print("ERROR: No GPU available. Falling back to CPU mode.")
        stage_1_start_server_cpu()
        return

    print(f"GPUs available: {faiss.get_num_gpus()}")

    # Create server configuration with GPU enabled
    config_dict = {
        "host": "0.0.0.0",
        "port": 9090,
        "workers": 1,
        "log_level": "info",
        "enable_request_logging": True,
        "log_dir": "/share_data/wangzehao/.faiss_log",
        "log_format": "json",
        "api_prefix": "/api/v1",
        "engine_config": {
            "index_type": "IVFFlat",
            "index_path": os.path.join(TMP_DIR, "browsecomp-plus-ivf-rebuilt", "e5_IVF2048,Flat.index"),
            "corpus_path": os.path.join(TMP_DIR, "browsecomp-plus-corpus", "browsecomp-plus-chunk-256.jsonl"),
            "retrieval_method": "e5",
            "retrieval_topk": TOP_K,
            "retrieval_batch_size": 32,
            "retrieval_model_path": "/share_data/public_model/multilingual-e5-large",
            "retrieval_query_max_length": 512,
            "retrieval_use_fp16": False,
            "retrieval_pooling_method": "mean",
            "use_sentence_transformer": True,
            "return_embedding": False,
            "larger_topk": 20,  # Required parameter for FaissEnginConfig
            "gpu_memory_utilization": 0.8,
            "eviction_policy": "lru",  # LRU eviction policy
            "eviction_max_attempts": 64,
            "nprobe": 128,
            "gpu_enabled": True,
            "gpu_ngpu": 1,
            "gpu_devices": [3],
            "gpu_ivf_init_strategy": "largest",  # Load largest clusters first
            "gpu_miss_policy": "cpu_offload",  # Auto-fetch missing lists
            "gpu_no_copy_evict": True,
        }
    }

    config = ServerConfig.from_dict(config_dict, validate=False)
    app = create_app(config)

    import uvicorn
    print("\nServer starting on http://0.0.0.0:9090")
    print("GPU Features enabled:")
    print("  - GPU acceleration on device 3")
    print("  - LRU eviction policy for IVF lists")
    print("  - CPU offload missing IVF lists")
    print("  - No copy eviction for IVF lists")
    print("\nAvailable endpoints:")
    print("  GET  /api/v1/health              - Health check")
    print("  GET  /api/v1/ready               - Readiness check")
    print("  POST /api/v1/search              - Search with text query")
    print("  POST /api/v1/batch_search        - Batch search")
    print("  GET  /api/v1/stats               - Server statistics")
    print("  GET  /api/v1/admin/evicted_lists - Get evicted IVF lists (GPU only)")
    print("  POST /api/v1/admin/evict         - Evict IVF lists (GPU only)")
    print("  POST /api/v1/admin/load          - Load IVF lists (GPU only)")
    print("\nPress Ctrl+C to stop the server\n")

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


def stage_3_query_server():
    """
    Stage 3: Query the server via HTTP API.
    """
    print("=" * 60)
    print("Stage 3: Querying the IVF Server")
    print("=" * 60)

    try:
        import requests
    except ImportError:
        print("ERROR: requests library not found. Install with: pip install requests")
        return

    base_url = "http://localhost:9090/api/v1"

    # Health check
    print("\n1. Health Check:")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {resp.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Is the server running? Start it with: python demo_ivf_server.py 1")
        return

    # Readiness check
    print("\n2. Readiness Check:")
    try:
        resp = requests.get(f"{base_url}/ready", timeout=5)
        print(f"   Status: {resp.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Search
    print("\n3. Single Search:")
    try:
        query = "machine learning"
        resp = requests.post(
            f"{base_url}/search",
            json={"query": query, "num": 3, "return_score": True},
            timeout=10
        )
        result = resp.json()
        print(f"   Query: '{query}'")
        print(f"   Latency: {result.get('latency_ms', 'N/A')} ms")
        print(f"   Results: {len(result.get('results', []))} documents")
        for i, doc in enumerate(result.get('results', [])[:3]):
            score = result.get('scores', [])[i] if result.get('scores') else 'N/A'
            print(f"     [{i+1}] (score: {score}) {str(doc)[:80]}...")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Batch search
    print("\n4. Batch Search:")
    try:
        queries = [
            "machine learning",
            "deep learning",
            "neural networks"
        ]
        resp = requests.post(
            f"{base_url}/batch_search",
            json={"queries": queries, "num": 3, "return_score": True},
            timeout=10
        )
        result = resp.json()
        print(f"   Queries: {queries}")
        print(f"   Latency: {result.get('latency_ms', 'N/A')} ms")
        print(f"   Results per query: {len(result.get('results', []))}")
        for i, q in enumerate(queries):
            q_results = result.get('results', [])[i] if i < len(result.get('results', [])) else []
            print(f"     Query '{q}': {len(q_results)} results")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Stats
    print("\n5. Server Statistics:")
    try:
        resp = requests.get(f"{base_url}/stats", timeout=5)
        result = resp.json()
        print(f"   Engine stats: {result.get('engine_stats', {})}")
        print(f"   Logger stats: {result.get('logger_stats', {})}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\nStage 3 complete!")


def stage_4_manage_ivf_lists():
    """
    Stage 4: Manage IVF lists (evict/load) - GPU only.
    """
    print("=" * 60)
    print("Stage 4: Managing IVF Lists (GPU Only)")
    print("=" * 60)

    try:
        import requests
    except ImportError:
        print("ERROR: requests library not found. Install with: pip install requests")
        return

    base_url = "http://localhost:8000/api/v1"

    # Check if server is running and GPU is available
    print("\n1. Checking server status...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        status = resp.json()
        print(f"   Status: {status}")
        if not status.get('engine_ready'):
            print("   ERROR: Engine not ready")
            return
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Is the server running? Start it with: python demo_ivf_server.py 2")
        return

    # Get evicted lists
    print("\n2. Get Evicted Lists:")
    try:
        resp = requests.get(f"{base_url}/admin/evicted_lists", timeout=5)
        if resp.status_code == 200:
            result = resp.json()
            evicted = result.get('evicted_lists', [])
            print(f"   Currently evicted lists: {evicted}")
            print(f"   Count: {len(evicted)}")
        else:
            print(f"   Note: {resp.json().get('detail', 'GPU features not available')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Evict specific lists
    print("\n3. Evict IVF Lists:")
    try:
        lists_to_evict = [0, 1, 2]
        resp = requests.post(
            f"{base_url}/admin/evict",
            json={"list_ids": lists_to_evict},
            timeout=5
        )
        if resp.status_code == 200:
            result = resp.json()
            reclaimed = result.get('reclaimed_bytes', {})
            print(f"   Evicted lists: {lists_to_evict}")
            print(f"   Reclaimed bytes: {reclaimed}")
            total_reclaimed = sum(reclaimed.values())
            print(f"   Total reclaimed: {total_reclaimed} bytes ({total_reclaimed / 1024 / 1024:.2f} MB)")
        else:
            print(f"   Note: {resp.json().get('detail', 'GPU features not available')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Load lists back to GPU
    print("\n4. Load IVF Lists to GPU:")
    try:
        lists_to_load = [0, 1]
        resp = requests.post(
            f"{base_url}/admin/load",
            json={"list_ids": lists_to_load},
            timeout=5
        )
        if resp.status_code == 200:
            result = resp.json()
            loaded = result.get('loaded_bytes', {})
            print(f"   Loaded lists: {lists_to_load}")
            print(f"   Loaded bytes: {loaded}")
            total_loaded = sum(loaded.values())
            print(f"   Total loaded: {total_loaded} bytes ({total_loaded / 1024 / 1024:.2f} MB)")
        else:
            print(f"   Note: {resp.json().get('detail', 'GPU features not available')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Get cluster stats
    print("\n5. Get Cluster Statistics:")
    try:
        resp = requests.get(f"{base_url}/admin/cluster_stats", timeout=5)
        if resp.status_code == 200:
            result = resp.json()
            stats = result.get('stats', [])
            print(f"   Total clusters with stats: {len(stats)}")
            if stats:
                print(f"   Top 5 most probed clusters:")
                sorted_stats = sorted(stats, key=lambda x: x.get('probe_count', 0), reverse=True)[:5]
                for s in sorted_stats:
                    print(f"     List {s['list_id']}: {s['probe_count']} probes, {s['load_count']} loads")
        else:
            print(f"   Note: {resp.json().get('detail', 'GPU features not available')}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\nStage 4 complete!")


def main():
    parser = argparse.ArgumentParser(
        description="IVF Server Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
    0 - Create and train IVF index
    1 - Start server in CPU mode
    2 - Start server in GPU mode with eviction
    3 - Query the server via HTTP API
    4 - Manage IVF lists (GPU only)

Examples:
    # Create index
    python demo_ivf_server.py 0

    # Start CPU server (terminal 1)
    python demo_ivf_server.py 1

    # In another terminal, query the server
    python demo_ivf_server.py 3

    # Or start GPU server with eviction
    python demo_ivf_server.py 2
        """
    )
    parser.add_argument("stage", type=int, choices=[0, 1, 2, 3, 4],
                        help="Demo stage to run")

    args = parser.parse_args()

    stages = {
        0: stage_0_create_index,
        1: stage_1_start_server_cpu,
        2: stage_2_start_server_gpu,
        3: stage_3_query_server,
        4: stage_4_manage_ivf_lists,
    }

    stages[args.stage]()


if __name__ == "__main__":
    main()
