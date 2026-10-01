vllm serve "/home/lin/vllm_code/model/qwen3-0.6b" \
  --served-model-name qwen3-0.6b \
  --enforce-eager \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --port "13311" \
