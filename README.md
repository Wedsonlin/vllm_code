# vLLM Course Exercises (`vllm_code`)

练习代码脚手架，对齐飞书修订版课纲 **「vllm从入门到精通-修订版」**（目标版本 vLLM **0.30.x / V1**）。  
Target repo after PR: https://github.com/Wedsonlin/vllm_code

## Course alignment / 课纲对齐

| Curriculum module | Exercise dirs |
|-------------------|---------------|
| A 上手与调试 | `exercises/A1_env_smoke` |
| B 运行时内核 | `B1`–`B5` (ZMQ, Executor, Scheduler, BlockPool, WeightLoader) |
| C 算子与图优化 | `C1_triton_vector_add`, `C2_cudagraph_constraints` |
| D 量化 | `D1_quant_math`, `D2_quant_config_parse` |
| E 分布式 | `E1_dp_lb_modes`, `E2_tp_shard_math`, `E3_moe_dispatch_sim` |
| F 高级特性与性能 | `F1_rejection_sampler_sim`, `F2_profiler_checklist`, `F3_kv_state_machine` |

Deep Feishu-aligned lesson bodies live in `course/revised/`; this package is the **verifiable practice track**.

## Repository layout

The exercise track sits next to the original lesson demos. `pytest` does not import `code/` or `course/` and does not download models.

| Path | Role |
|------|------|
| `course/revised/` | Deep Feishu-aligned Chinese lesson bodies (A1–F4 + R1), mermaid, source walkthroughs |
| `exercises/` | Practice track for Feishu modules A–F |
| `common/` | Shared fixtures and the CUDA skip helper |
| `code/` | Original course demos (unchanged) |
| `pytest.ini`, `conftest.py`, `requirements-exercises.txt` | CPU-first test setup |

## Quick start

```bash
pip install -r requirements-exercises.txt
python -m pytest -q
```

See `exercises/README.md` for markers, GPU notes, and per-lesson layout.

## Learner workflow

1. Read `course/revised/<lesson>.md` (or the matching Feishu page: 动机 → 可视化 → 源码要点).
2. Open `exercises/<ID>/README.md` for acceptance criteria.
3. Optionally replace the reference module with `starter/` stubs and re-implement.
4. Run `pytest exercises/<ID> -q` until green.

Reference implementations ship **complete** so CI/`pytest` is green out of the box; `starter/` holds the blank homework copy.
