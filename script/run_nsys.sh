export CUDA_VISIBLE_DEVICES=0


# python demos/benchmark_ivfflat_evict.py \
#        --use_profiler 0 \
#        --mode copy

# Use nsys profile with CUDA + NVTX tracing. The CUDA profiler range is
# controlled from Python via torch.cuda.cudart().cudaProfilerStart/Stop,
# so we do not need legacy --profile-from-start flags here.
# We also disable CPU sampling and allow overwriting existing reports to
# avoid noisy warnings in the output. Restrict GPU metrics to device 0;
# change --gpu-metrics-device if you want to profile a different GPU.

nsys profile --trace=cuda,nvtx \
             --gpu-metrics-device=0 \
             --sample=none \
             --force-overwrite true \
             -o no_ivf-temp-0228 \
             python demos/benchmark_ivfflat_evict.py \
             --use_profiler 1 \
             --mode no_copy \

