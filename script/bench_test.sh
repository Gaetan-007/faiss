# python benchs/bench_big_batch_ivf_gpu.py \
#     --nb 100000 --dim 1024 --nlist 256 --gpu 0 \
#     --nq 100 \
#     --nruns 50 \
#     --auto-fetch \
#     --gpu-lists-num -1

# NOTE: base-pool-mb must be large enough to hold:
#   - Index data: nb * dim * 4 bytes (for float32)
#   - Temp memory: ~1/4 of pool
#   - IVF structure overhead
# For nb=500000, dim=256: data ~= 488 MB, so pool should be ~2048 MB
export CUDA_VISIBLE_DEVICES=7
python benchs/bench_gpu_pool_resize.py \
    --nb 500000 \
    --dim 1024 \
    --base-pool-mb 2048 \
    --batch-sizes "100,500,1000,2000,5000" \
    --delta-sizes-mb "32,64,128,256,512" \
    --nruns 10 \
    --output-dir ./bench_results \
    --prefix my_bench
