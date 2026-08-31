#!/bin/bash
# DreamZero HTTP serving on the Pro 6000 — validated fast config (2026-08-31):
# dyncache + CFG-off + per-block compile (+3-tap smoothing in-server) = 0.98s/chunk,
# quality 1.78x (dense gate). --async_pipeline: one-chunk-ahead prediction,
# ~0.18s stall per 0.8s chunk. Compile cache persisted: warmup paid once.
V=/dev/shm/dzvenv
export PATH=$V/bin:$PATH
export PYTHONPATH=${HOME}/dreamzero:${HOME}/dreamzero/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DZ_SKIP_LORA_FP32=1
export CUDA_HOME=/usr/local/cuda
export DYNAMIC_CACHE_SCHEDULE=true
export DZ_CFG_SCALE=1.0
export DZ_COMPILE_BLOCKS=perblock
export DZ_NO_ENCODER_COMPILE=1
export DZ_NO_SCHED_COMPILE=1
export TORCHINDUCTOR_CACHE_DIR=${HOME}/.inductor_cache
cd ${HOME}/dreamzero
$V/bin/python scripts/serve_deepcybo_lite_dreamzero_http.py \
  --model_path ${HOME}/b300_ckpts/v3-checkpoint-2500 \
  --base_path /dev/shm/dz/DreamZero-AgiBot \
  --tokenizer_path ${HOME}/umt5-xxl-tokenizer \
  --port 9090 --async_pipeline --ground_split --smooth_taps 3 \
  > ${HOME}/eval_results/http_server.log 2>&1
echo "SERVER EXIT: $?" >> ${HOME}/eval_results/http_server.log
