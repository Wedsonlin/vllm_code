# Exercises / 练习说明

Aligned with curriculum redesign (**vLLM 0.30.x V1**).  
与修订版课纲对齐：模块 A–F，每课带可验证 pytest。

## Install / 安装

```bash
# from the vllm_code repository root
pip install -r requirements-exercises.txt
```

Optional GPU / torch:

```bash
pip install torch  # then: pytest -m gpu
```

## Run tests / 跑测

```bash
# all CPU-default exercises (should be green with reference solutions)
python -m pytest -q

# skip slow
python -m pytest -q -m "not slow"

# only GPU-marked (skipped without CUDA)
python -m pytest -q -m gpu
```

## Layout / 目录

Each lesson directory:

| Path | Role |
|------|------|
| `README.md` | 题目 + 验收标准 |
| `*.py` (module) | **Reference solution** (pytest green) + TODO comments for learners |
| `starter/` | Blank / stub copy for homework |
| `tests/test_*.py` | Self-contained verification |

## Markers

- `@pytest.mark.gpu` — needs CUDA
- `@pytest.mark.slow` — longer jobs

## Course map

| Dir | Module | Topic |
|-----|--------|-------|
| A1_env_smoke | A | Env / serve args smoke |
| B1_zmq_patterns | B | REQ/REP framing (queue sim) |
| B2_executor_handshake_sim | B | Executor handshake |
| B3_scheduler_token_budget | B | Token budget admission |
| B4_block_pool | B | Refcounted KV blocks |
| B5_weight_loader_stub | B | TP weight sharding |
| C1_triton_vector_add | C | Vector add (numpy default) |
| C2_cudagraph_constraints | C | Capture constraint checklist |
| D1_quant_math | D | Symmetric quant/dequant |
| D2_quant_config_parse | D | Quant JSON config |
| E1_dp_lb_modes | E | DP LB modes |
| E2_tp_shard_math | E | Column/row shard shapes |
| E3_moe_dispatch_sim | E | Expert routing counts |
| F1_rejection_sampler_sim | F | Speculative accept/reject |
| F2_profiler_checklist | F | Profiler field validation |
| F3_kv_state_machine | F | KV transfer states |
