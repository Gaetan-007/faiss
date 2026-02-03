
# GPU功能测试
# python -m pytest faiss/gpu/test/test_ivfflat_evict_load.py
# python -m pytest tests/test_engine_eviction_policy.py
# python -m pytest tests/test_engine_scheduler.py

# GPU池测试
# python -m pytest faiss/gpu/test/test_gpu_memory_reservation.py
python -m pytest tests/test_gpu_ivfflat_selective_init.py

# 服务相关测试
# python -m pytest tests/test_server.py
# python -m pytest tests/test_server_integration.py

# GPU池IPC测试
python -m pytest tests/test_gpu_pool_ipc.py