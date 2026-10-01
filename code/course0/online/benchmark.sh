#!/usr/bin/env bash


vllm bench serve \
    --model /home/lin/vllm_code/model/qwen3-0.6b \
    --served-model-name qwen3-0.6b \
    --host 127.0.0.1 \
    --random-input-len 128 \
    --port 13311 \
    --request-rate 10 \
    --num-prompts 100 \
    --save-result \
    --result-dir ./bench_results \
    --label "qwen3-0.6b-test"
        
