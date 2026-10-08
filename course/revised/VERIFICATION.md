# vLLM 课件【待核】标记核验报告

- 核验基准：vLLM **v0.30.0** tag（commit `ced6857a`，2026-09-22 06:32 UTC+8）。v0.30.x 系列中唯一的正式版本是 v0.30.0（v0.30.1rc0 只是 RC 标签）。截至核验日，最新正式版本为 v0.31.0，本次核验未采用。
- 源码副本：`/workspace/vllm-src`（浅克隆，未安装 GPU 依赖）。证据列中的 `vllm-src/<path>:<line>` 均指该树。
- 输入：`/workspace/feishu-lessons-formal/`（21 课 + STYLE-CHANGES.md + images/）。输出：`/workspace/feishu-lessons-verified/`。

## 1. 统计

课件中实际共有 **80** 处【待核】标记（委派说明中写的是 76 处）：页首标注 21 处、正文 58 处、页脚引文 1 处（B6）。另外 STYLE-CHANGES.md 中有 4 处提及“【待核】”，均属历史说明而非标记，按原样保留。

| 类别 | 数量 |
|---|---|
| CONFIRMED（原述正确，仅删除标记或以陈述句补充事实） | 19 |
| CORRECTED（原述错误、过时或不精确，已更正） | 39 |
| UNVERIFIABLE-REWORDED（无法由源码确认，已改写为可核实表述） | 1 |
| HEADER（页首“源码标注”说明，统一改写为已核实声明） | 21 |
| KEPT（保留为【待核：…】） | 0 |

正文与页脚共 59 处，其中 CONFIRMED 19、CORRECTED 39、UNVERIFIABLE-REWORDED 1、保留 0。

edits.json 共 120 条替换（其中第 8 节所述未标注处修正 26 条）。与【待核】标记相关的替换：其中标记替换 68 条、页首 21 条，另有 5 条依赖性修改（这些文字本身没有标记，但内容直接依赖于某项更正）。

## 2. 最重要的更正

1. **ModelRunner V2 已是默认实现**（F4）。`VLLM_USE_V2_MODEL_RUNNER` 默认不设置，只要 Triton 可用且没有不支持的特性，就使用 V2；遇到不支持的特性时自动回退到 V1。`=0` 强制使用 V1，`=1` 强制使用 V2。（`vllm/config/vllm.py:675-723,2815-2882`）
2. **`VLLM_TORCH_PROFILER_DIR` 已移除**（F2，代码块），改为 `--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'`。nsys 区间采集改用 `--profiler-config.profiler cuda`。
3. **`VLLM_ATTENTION_BACKEND` 已移除，`vllm/attention/` 目录不再存在**（C1/A1）。后端选择改用 `--attention-backend` / `--attention-config`；selector 与 ops 移到 `vllm/v1/attention/`，Attention 层移到 `vllm/model_executor/layers/attention/attention.py`。
4. **`-O` 是 `--optimization-level`（O0–O3，默认 O2）**（C3/C2），不是编译 mode 的取值。`level` 字段已移除，`full_cuda_graph` 字段也已移除。默认的 `FULL_AND_PIECEWISE` 对应 O2/O3。
5. **EngineCore.step 拆分为 `execute_model` + `sample_tokens`**（B1/B6）。`ModelRunnerOutput` 不再包含 `spec_token_ids`，草稿改由 `take_draft_token_ids()` 获取。
6. **DP 负载均衡打分公式已变**（E1）。新公式为 `max(client_count×inflight, waiting+running) + waiting×6×max(0, kv_usage−0.5)`，数值示例已同步重算。
7. **量化**（D2）：`gptq`/`awq` 被改写为 `auto_gptq`/`auto_awq`，kernel 由 `choose_mp_linear_kernel` 逐层选择；`gptq_marlin.py`、`awq_marlin.py`、`bitsandbytes.py`、`gguf.py` 均已不存在，bnb 与 GGUF 迁到外部插件；Machete 需要 SM90。
8. **MoE**（E3）：`FusedMoEModularKernel` 和 `FusedMoEPermuteExpertsUnpermute` 分别更名为 `FusedMoEKernel` 和 `FusedMoEExperts`；all2all 的 `naive`/`pplx` 已移除，默认值为 `allgather_reducescatter`；双批次重叠的开关是 `--enable-dbo`。
9. **KV Connector**（F3）：`P2pNcclConnector` 已不存在，`SharedStorageConnector` 对应的是 `ExampleConnector`；KV 加载失败的默认策略是 `kv_load_failure_policy="fail"`，而不是回退重算。
10. **async scheduling 默认自动开启**（B2），并写明了自动关闭的条件和 `max_concurrent_batches` 的取值。

## 3. 练习影响

已检查 `Wedsonlin/vllm_code` 分支 `cursor/vllm-course-exercises-75c8`（commit `661ea24`）下的 `exercises/`，未修改该仓库。所有练习都是不依赖 vLLM 的玩具实现，没有引用任何 vLLM 符号或参数，因此**没有练习需要修改**：
- E1 `classify_dp_lb_mode`：只区分 internal/hybrid/external 三种模式，不涉及打分公式，不受 Q40 影响。
- D2：`ALLOWED_METHODS = {"fp8","awq","gptq","smoothquant","int8"}` 是教学用列表，与 Q35/Q36 的更正无冲突（如需更贴近 v0.30.0，可在讲解中说明 gptq/awq 会被改写为 auto_gptq/auto_awq，但不必改代码）。
- F2：只检查通用字段，不包含 `VLLM_TORCH_PROFILER_DIR` 等具体名称。
- F3：状态机中 abort→RELEASED，与 v0.30.0 默认的 `"fail"` 策略一致。
- 注意：该仓库 `course/revised/` 下也有课件副本。如需保持一致，应同样应用 edits.json（由上层决定是否修改）。

## 4. 发现但未修改的无标记差异（供课程负责人决定）

> 后续更新：本节所列各项（STYLE-CHANGES.md 除外）已按要求在第 8 节「未标注处的过时表述」中修正。

以下文字没有【待核】标记，且不直接依赖于本次更正，按“不改其他内容”的要求在首轮核验中未作修改。按 v0.30.0 源码，它们同样已过时，建议另行处理：
- **E3 第 15、48–51 行**：`FusedMoE` 类在 v0.30.0 中已不存在。MoE 层改为 `vllm/model_executor/layers/fused_moe/routed_experts.py` 中的 `RoutedExperts`，`layer.py` 中已无类定义；路由 `select_experts` 位于 `fused_moe/router/fused_moe_router.py:49`。
- **E4 第 49 行**：`vllm/distributed/eplb/rebalance_algo.py` 已不存在，`rebalance_experts` 现位于 `vllm/distributed/eplb/policy/default.py:275`（抽象接口见 `policy/abstract.py`）。**E4 第 51 行**的 `fused_moe/layer.py` 中的 `select_experts` 也应改为 `fused_moe/router/`。
- **B3 第 138 行**：参数表中 `--async-scheduling` 的默认值写作“视版本而定”，可按 Q09 的结论改为“自动（无不兼容项时开启）”。
- **B6 全课**：描述的是 V1 `gpu_model_runner.py`。v0.30.0 中默认已是 ModelRunner V2（F4/Q75），V1 仍保留并作为回退实现。建议在 B6 开头补充一句说明。
- **D2 第 109 行**：“GPTQ 与 AWQ 在满足条件时将被自动升级为 Marlin 实现”。现行机制是由 `choose_mp_linear_kernel` 逐层选择 kernel（见 Q35/Q37），可酌情统一措辞。
- **STYLE-CHANGES.md**：其中“【待核】标记均予保留”及统计表属于上一轮改写的历史记录，原样保留。

## 5. 文件

- `/workspace/feishu-lessons-verified/`：完整课件副本（21 课、STYLE-CHANGES.md、images/），标记已全部处理。已校验：课件中不再有“待核”，各文件代码围栏数量和图片引用与原稿逐一一致。
- `/workspace/daihe/inventory.csv`：原始清单（80 行）。`/workspace/daihe/inventory_verified.csv`：在清单基础上增加 status、evidence、note 三列。
- `/workspace/daihe/edits.json`：飞书替换清单，每条包含以下字段：
  - `lesson_id`、`item_ids`、`kind`；
  - `old_text` / `new_text`：纯文本，已去除行内代码、加粗、链接和列表/标题标记，便于在飞书中匹配；
  - `old_text_md` / `new_text_md`；
  - `in_code_block`、`spans_formatting`、`notes`；
  - `plain_occurrences`：old_text 在纯文本渲染后的原文中出现的次数，所有条目均为 1。
  - `spans_formatting=true` 表示原文跨越行内代码或加粗的格式边界，替换时需整段替换，并按 new_text_md 重新设置行内代码格式。
- 脚本：`/workspace/daihe/items.py`（逐项结论与替换文本）、`apply_edits.py`、`make_report.py`。

## 6. 逐项明细

| ID | 课 | 行 | 状态 | 原标记 | 结论说明 | 证据（v0.30.0） |
|---|---|---|---|---|---|---|
| Q01 | A1 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q02 | A1 | 30 | CONFIRMED | 【待核：参数名】 | `--torch-backend=auto` 是 uv 的参数，vLLM 官方安装文档采用相同写法；注释略作精确化 | vllm-src/docs/getting_started/installation/gpu.cuda.inc.md:25,34 |
| Q03 | B1 | 5 | HEADER | 【待核】 | 页首源码标注约定：改写为已核实声明，并将 v0.30.x 改为实际 tag v0.30.0 | - |
| Q04 | B1 | 56 | CORRECTED | 【待核：0.30.x 中 serving 类可能拆分至 `entrypoints/openai/chat_completion/` 等子目录】 | build_async_engine_client 实际位于 launchers/api_server/entry.py；serving 类已拆分至 chat_completion/ 子目录 | vllm-src/vllm/entrypoints/openai/api_server.py:5-30（已弃用的转发模块）; vllm/entrypoints/launchers/api_server/entry.py:36,67,94; vllm/entrypoints/openai/chat_completion/api_router.py:42,54; vllm/entrypoints/openai/chat_completion/serving.py:118,242 |
| Q05 | B1 | 59 | CORRECTED | 【待核：新版本中命名为 `input_processor.py` / `InputProcessor`】 | processor.py 已不存在；v0.30.0 中为 input_processor.py / InputProcessor，属性名为 self.input_processor | vllm-src/vllm/v1/engine/async_llm.py:50,146,439; vllm/v1/engine/input_processor.py:38,281 |
| Q06 | B1 | 72 | CORRECTED | 【待核：新版本中 `execute_model` 与 `sample_tokens` 可能拆分为两次调用】 | step() 先以 non_block 方式调用 execute_model，返回 None 时再调用 sample_tokens(grammar_output) | vllm-src/vllm/v1/engine/core.py:589-619 |
| Q07 | B2 | 5 | HEADER | 【待核】 | 页首源码标注说明：改写为已核实声明 | - |
| Q08 | B2 | 52 | CONFIRMED | 【待核：0.30.x 可能已将 `UniProcExecutor` 移至 `uniproc_executor.py`】 | get_class/_init_executor 位置正确；UniProcExecutor 确在 uniproc_executor.py，以陈述句补入 | vllm-src/vllm/v1/executor/abstract.py:38,49,110,117; vllm/v1/executor/uniproc_executor.py:51 |
| Q09 | B2 | 84 | CORRECTED | 【待核：0.30.x 中 async scheduling 是否默认开启，以及其与投机解码、PP 的兼容范围】 | async_scheduling 默认值为 None，无不兼容项时自动开启；兼容范围与 max_concurrent_batches 取值按源码写明 | vllm-src/vllm/config/scheduler.py:190-193; vllm/config/vllm.py:589-599,1407-1487; vllm/engine/arg_utils.py:1644 |
| Q10 | B3 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q11 | B3 | 48 | CONFIRMED | 【待核】 | 公式与字段含义与源码一致 | vllm-src/vllm/v1/core/sched/scheduler.py:656-660; vllm/v1/request.py:159-160 |
| Q12 | B3 | 133 | CORRECTED | 【待核：0.30.x 按显卡与用途设有不同默认值】 | 默认值依显存、是否为 A100、入口（LLM 类 / API 服务）与 performance_mode 而定，写出确切取值 | vllm-src/vllm/engine/arg_utils.py:2700-2792,2919-2940; vllm/config/scheduler.py:42-44 |
| Q13 | B4 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q14 | B4 | 67 | CONFIRMED | 【待核：类名】 | 三个类名均存在 | vllm-src/vllm/v1/core/kv_cache_coordinator.py:477,529,615 |
| Q15 | B5 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q16 | B6 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q17 | B6 | 56 | CORRECTED | 【待核：0.30.x 中 dispatch 的签名】 | dispatch 不接收 BatchDescriptor，而是接收 num_tokens 等参数并返回 (CUDAGraphMode, BatchDescriptor) | vllm-src/vllm/v1/cudagraph_dispatcher.py:235-243; vllm/v1/worker/gpu_model_runner.py:3999-4006 |
| Q18 | B6 | 62 | CORRECTED | 【待核：新版本中第 7～10 步可能被拆分至独立的 `sample_tokens()`，以配合异步调度】 | 第 8～10 步已拆至 sample_tokens()；ModelRunnerOutput 不再含 spec_token_ids，草稿经 take_draft_token_ids() 获取 | vllm-src/vllm/v1/worker/gpu_model_runner.py:482-495,4187,4529-4548,4566,4624; vllm/v1/outputs.py:324-363,430; vllm/v1/engine/core.py:626 |
| Q19 | B6 | 186 | CORRECTED | 【待核：0.30.x 中 dispatch 的签名】 | 页脚引文同步删除标记 | 同 Q17 |
| Q20 | C1 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q21 | C1 | 97 | CORRECTED | 【待核：0.30.x 中具体启发式】 | 给出 2D/3D 内核切换的确切启发式 | vllm-src/vllm/v1/attention/backends/triton_attn.py:53-54,133-154; vllm/v1/attention/ops/triton_unified_attention.py:1041-1055,1079-1086 |
| Q22 | C1 | 108 | CORRECTED | 【待核：0.30.x 可能改为 `--attention-backend` 或 `attention_config` 配置项】 | selector 已移至 vllm/v1/attention/；签名不再含 block_size；VLLM_ATTENTION_BACKEND 已移除，改用 --attention-backend / --attention-config | vllm-src/vllm/v1/attention/selector.py:104-116; vllm/platforms/cuda.py:450; vllm/config/attention.py:40-41; vllm/engine/arg_utils.py:1008,1726; vllm/v1/attention/backends/registry.py:48 |
| Q23 | C1 | 110 | CORRECTED | 【待核：ops 目录在 0.30.x 中可能已迁移至 `vllm/v1/attention/ops/`】 | vllm/attention/ 目录在 v0.30.0 中已不存在：ops 迁至 vllm/v1/attention/ops/，Attention 层迁至 vllm/model_executor/layers/attention/attention.py | vllm-src/vllm/v1/attention/ops/triton_unified_attention.py:807; vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:363; vllm/v1/attention/backends/triton_attn.py:274,381; vllm/model_executor/layers/attention/attention.py:225,569,760 |
| Q24 | C2 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q25 | C2 | 87 | CORRECTED | 【待核：0.30.x 默认值；以及 `cudagraph_mode` 与旧参数 `full_cuda_graph` 的对应关系】 | FULL_AND_PIECEWISE 为 -O2（默认）/-O3 的默认值；-O1 为 PIECEWISE，-O0 为 NONE；full_cuda_graph 字段已移除 | vllm-src/vllm/config/compilation.py:614-649; vllm/config/vllm.py:256-341,440 |
| Q26 | C2 | 97 | CORRECTED | 【待核：0.30.x 的确切默认序列】 | 写出确切默认序列与上限公式 | vllm-src/vllm/config/compilation.py:699-714; vllm/config/vllm.py:2151-2170 |
| Q27 | C3 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q28 | C3 | 45 | CORRECTED | 【待核：0.30.x 可能已重命名】 | 包装类为 wrapper.py 中的 TorchCompileWithNoGuardsWrapper（由装饰器注入为基类），并非 decorators.py 中的 TorchCompileWrapperWithCustomDispatcher | vllm-src/vllm/compilation/wrapper.py:47-54,104-125,157-160,171-187,273-286; vllm/compilation/decorators.py:342-349 |
| Q29 | C3 | 51 | CONFIRMED | 【待核：类名】 | 两类名均存在（另有 EagerAdaptor:796） | vllm-src/vllm/compilation/compiler_interface.py:251,449 |
| Q30 | C3 | 59 | CORRECTED | 【待核：0.30.x 中 `mode`/`level` 的命名与 `-O` 级别语义】 | level 字段已移除；mode 为 CompilationMode 枚举；-O 是 --optimization-level 的简写（O0～O3，默认 O2），并非 mode 的取值 | vllm-src/vllm/config/compilation.py:37-50,453-467; vllm/config/vllm.py:127-139,440,1595-1599; vllm/utils/argparse_utils.py:341-352 |
| Q31 | C3 | 83 | CONFIRMED | 【待核：默认值随版本变化】 | 未显式设置时：Inductor 后端且编译开启 → 追加 "none"；否则 → "all"，与正文描述一致；以陈述句写明 | vllm-src/vllm/config/vllm.py:1606-1613 |
| Q32 | D1 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q33 | D1 | 123 | CORRECTED | 【待核：0.30.x 支持范围】 | W4A8 需 SM90；FP4 在 Blackwell 原生执行，较早 GPU 回退为 Marlin 仅权重 FP4 | vllm-src/vllm/model_executor/kernels/linear/mixed_precision/cutlass.py:28-46; vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:49-50; vllm/model_executor/kernels/linear/nvfp4/marlin.py:20-38; vllm/model_executor/layers/quantization/modelopt.py:753; vllm/model_executor/layers/quantization/mxfp4.py:65 |
| Q34 | D2 | 5 | HEADER | 【待核】 | 页首源码标注说明：保留首句，改写为已核实声明 | - |
| Q35 | D2 | 38 | CORRECTED | 【待核：0.30.x 的升级规则】 | gptq/awq 被改写为 auto_gptq/auto_awq（gptq_marlin、awq_marlin 仅为映射至同一配置类的别名）；kernel 由 choose_mp_linear_kernel 逐层选择 | vllm-src/vllm/config/model.py:1258-1335; vllm/model_executor/layers/quantization/__init__.py:156-168; vllm/model_executor/layers/quantization/auto_gptq.py:176-235; vllm/model_executor/layers/quantization/auto_awq.py:222-281 |
| Q36 | D2 | 45 | CORRECTED | 【待核：目录结构】 | gptq_marlin.py/awq_marlin.py/bitsandbytes.py/gguf.py 已不存在；kernel 选择器位于 vllm/model_executor/kernels/linear/；bnb 与 GGUF 迁为外部插件 | vllm-src/vllm/model_executor/layers/quantization/（目录列表）; vllm/model_executor/kernels/linear/{mixed_precision,scaled_mm}/; vllm-src/docs/features/quantization/bnb.md:6-13; docs/features/quantization/gguf.md:6-13 |
| Q37 | D2 | 110 | CORRECTED | 【待核：0.30.x 中的启用条件】 | 启用条件：CUDA 且计算能力恰为 9.0、量化类型/group size/形状受支持；优先级高于 Marlin；可用 VLLM_DISABLED_KERNELS 禁用 | vllm-src/vllm/model_executor/kernels/linear/mixed_precision/machete.py:22-54; vllm/model_executor/kernels/linear/__init__.py:498-508,813-870 |
| Q38 | E1 | 5 | HEADER | 【待核】 | 同上 | - |
| Q39 | E1 | 54 | CONFIRMED | 【待核：0.30.x 中参数名与 hybrid/external 开关】 | 所列参数名均正确；补充 hybrid/external 开关的确切名称 | vllm-src/vllm/engine/arg_utils.py:1126-1188; vllm/entrypoints/launchers/cli_args.py:404-412 |
| Q40 | E1 | 58 | CORRECTED | 【待核：权重】 | 打分并非 waiting×4+running，而是 max(client_count×inflight, waiting+running) + waiting×6×max(0, kv_cache_usage−0.5)；随附数值示例同步改写 | vllm-src/vllm/v1/engine/core_client.py:1507,1546-1597 |
| Q41 | E1 | 77 | CONFIRMED | 【待核：函数位置】 | launch_core_engines 确为 utils.py 中的上下文管理器 | vllm-src/vllm/v1/engine/utils.py:1103-1104; vllm/v1/engine/coordinator.py:23 |
| Q42 | E1 | 81 | CORRECTED | 【待核：0.30.x 中的同步方式】 | 同步由 dp_utils.coordinate_batch_across_dp() 完成（all-reduce 4×dp_size 张量），异步调度时默认走 CPU/Gloo 组 | vllm-src/vllm/v1/worker/dp_utils.py:39-77,192-215; vllm/v1/worker/gpu_model_runner.py:219,4027; vllm/forward_context.py:73; vllm/config/vllm.py:1489-1500 |
| Q43 | E1 | 98 | UNVERIFIABLE-REWORDED | 【待核：0.30.x 中的稳定程度与参数】 | 参数与约束可由源码确认；'实验性/稳定程度' 源码与文档均无正式说明，改为陈述条件 | vllm-src/vllm/config/parallel.py:217-220,886-899; vllm/engine/arg_utils.py:1211-1215; vllm/entrypoints/serve/elastic_ep/api_router.py:27,83; docs/serving/online_serving/README.md:157-160 |
| Q44 | E2 | 5 | HEADER | 【待核】 | 页首源码标注说明：所有正文标记处理完毕后，改写为已核实声明 | - |
| Q45 | E2 | 61 | CORRECTED | 【待核：0.30.x 中的开启方式】 | 开启方式为 pass_config.enable_sp / fuse_gemm_comms；默认关闭；附带约束 | vllm-src/vllm/config/compilation.py:136-141,1226-1244; vllm/config/vllm.py:144-150,302-341,1677-1707 |
| Q46 | E2 | 69 | CORRECTED | 【待核：0.30.x 中默认选择逻辑】 | v0.30.0 的 all_reduce 依固定顺序尝试多种实现，FlashInfer all-reduce（默认开启）排在 vLLM 自定义 all-reduce 之前 | vllm-src/vllm/distributed/device_communicators/cuda_communicator.py:50-135,327-400; vllm/envs.py:266-268,291 |
| Q47 | E3 | 5 | HEADER | 【待核】 | 同上 | - |
| Q48 | E3 | 38 | CONFIRMED | 【待核：0.30.x 是否支持独立设置 EP 大小】 | EP 大小 = DP×TP（×PCP），无独立设置 EP 大小的参数 | vllm-src/vllm/model_executor/layers/fused_moe/config.py:1212-1258; vllm/engine/arg_utils.py:1189-1193 |
| Q49 | E3 | 52 | CORRECTED | 【待核：0.30.x 类名与目录 `fused_moe/modular_kernel.py`】 | FusedMoEModularKernel / FusedMoEPermuteExpertsUnpermute 已更名为 FusedMoEKernel / FusedMoEExperts（各有 Modular 与 Monolithic 子类）；目录正确 | vllm-src/vllm/model_executor/layers/fused_moe/modular_kernel.py:187,257,421,474,774,975,1615-1640 |
| Q50 | E3 | 54 | CORRECTED | 【待核：0.30.x 可选值】 | naive 与 pplx 已移除（自动回退为 allgather_reducescatter）；补全可选值；VLLM_ALL2ALL_BACKEND 已不存在 | vllm-src/vllm/config/parallel.py:42-55,197-207,501-507; vllm/distributed/device_communicators/all2all.py:46-1025 |
| Q51 | E3 | 74 | CORRECTED | 【待核：0.30.x 中双批次重叠的开关】 | 开关为 --enable-dbo（dual batch overlap），并有 token 阈值 | vllm-src/vllm/config/parallel.py:222-235,597-601; vllm/engine/arg_utils.py:1205-1228 |
| Q52 | E4 | 5 | HEADER | 【待核】 | 同上 | - |
| Q53 | E4 | 52 | CONFIRMED | 【待核：0.30.x 的字段名与默认值】 | 字段名正确；补充各字段默认值（num_redundant_experts 默认 0，示例中的 32 为示例值）；旧独立参数已不存在 | vllm-src/vllm/config/parallel.py:58-100; vllm/engine/arg_utils.py:1233-1234 |
| Q54 | E4 | 77 | CORRECTED | 【待核：0.30.x 的副本选择策略】 | v0.30.0 按 token 序号做 Knuth 乘法哈希后对副本数取模，不做节点亲和 | vllm-src/vllm/model_executor/layers/fused_moe/router/base_router.py:19-62 |
| Q55 | F1 | 5 | HEADER | 【待核】 | 同上 | - |
| Q56 | F1 | 33 | CONFIRMED | 【待核：0.30.x 中内置处理器列表】 | 内置批量处理器恰为 MinTokens / LogitBias / MinP 三个 | vllm-src/vllm/v1/sample/logits_processor/__init__.py:50-54; vllm/v1/sample/logits_processor/builtin.py:23,119,165 |
| Q57 | F1 | 39 | CORRECTED | 【待核：0.30.x 默认实现】 | CUDA 上默认启用 FlashInfer 采样（VLLM_USE_FLASHINFER_SAMPLER 默认 1）；回退路径在 Triton 可用时为 Triton 实现，而非直接 PyTorch | vllm-src/vllm/v1/sample/ops/topk_topp_sampler.py:44-90,100-118,368-378; vllm/envs.py:858-862 |
| Q58 | F1 | 84 | CORRECTED | 【待核：0.30.x 方法名与新增方法】 | 补全 v0.30.0 方法名：draft_model、ngram_gpu、suffix、mlp_speculator、dflash、dspark、extract_hidden_states、custom_class 等 | vllm-src/vllm/config/speculative.py:37-84,393-394,458-461; docs/features/speculative_decoding/README.md:9-17,84 |
| Q59 | F1 | 86 | CONFIRMED | 【待核】 | kernel 名称、占位值 −1、parse_output 与无草稿概率时 draft_prob=1（即以 p(x) 概率接受）均与源码一致 | vllm-src/vllm/v1/sample/rejection_sampler.py:38,107-123,268-290,719,822,860-877,997 |
| Q60 | F1 | 120 | CORRECTED | 【待核：指标名】 | Counter 在 /metrics 中以 _total 后缀暴露；分母指标名为 vllm:spec_decode_num_draft_tokens_total | vllm-src/vllm/v1/spec_decode/metrics.py:182-195,226-232 |
| Q61 | F2 | 5 | HEADER | 【待核】 | 同上 | - |
| Q62 | F2 | 53 | CONFIRMED | 【待核：0.30.x 中各指标的确切名称，旧版本中 KV 使用率为 `gpu_cache_usage_perc`、ITL 为 `time_per_output_token_seconds`】 | 表中各指标名均与源码一致；旧名在 v0.30.0 中不存在；补充 Counter 的 _total 后缀与投机解码指标名 | vllm-src/vllm/v1/metrics/loggers.py:499,509,567,590,601,667,812,822,842 |
| Q63 | F2 | 68 | CORRECTED | 【待核：0.30.x 可能改为 --profiler-config】 | 环境变量 VLLM_TORCH_PROFILER_DIR 已移除，改用 --profiler-config；代码块同步修改 | vllm-src/vllm/config/profiler.py:16,39-49; docs/contributing/profiling.md:12-15,49; vllm/v1/worker/gpu_worker.py:1248-1256（VLLM_TORCH_PROFILER_DIR 在源码中已无任何引用） |
| Q64 | F2 | 104 | CORRECTED | 【待核】 | 对应开关为 --profiler-config.profiler cuda，配合 nsys 的 --capture-range=cudaProfilerApi | vllm-src/docs/contributing/profiling.md:186-236; vllm/config/profiler.py:16,42-46 |
| Q65 | F3 | 5 | HEADER | 【待核】 | 同上（补充：后续版本可能调整） | - |
| Q66 | F3 | 47 | CORRECTED | 【待核：0.30.x 中 NixlConnector 对部分块的处理细节】 | D 侧请求整个 prompt 的 KV（Mamba 模型为 len−1），由调度器将 num_computed_tokens 回退为 len−1 以本地重算最后一个位置 | vllm-src/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:385-390; vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:34-62; vllm/v1/core/sched/scheduler.py:2957-2958 |
| Q67 | F3 | 60 | CONFIRMED | 【待核：方法名】 | 方法名 update_connector_output(connector_output) 正确 | vllm-src/vllm/distributed/kv_transfer/kv_connector/v1/base.py:572 |
| Q68 | F3 | 70 | CORRECTED | 【待核：0.30.x 中的完整列表】 | P2pNcclConnector 与 SharedStorageConnector 不在 v0.30.0 注册表中（后者对应 ExampleConnector）；补全完整列表 | vllm-src/vllm/distributed/kv_transfer/kv_connector/factory.py:152-245; vllm/distributed/kv_transfer/kv_connector/v1/（目录列表） |
| Q69 | F3 | 76 | CONFIRMED | 【待核：支持的组合与约束，如要求 D 的 TP 能整除 P 的 TP 或反之】 | 双向异构 TP 均受支持，约束为较大者须能被较小者整除；以陈述句写明 | vllm-src/vllm/distributed/kv_transfer/kv_connector/utils.py:486-504; vllm/distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py:100-150 |
| Q70 | F3 | 113 | CORRECTED | 【待核：0.30.x 的失败恢复策略】 | 失败恢复由 kv_load_failure_policy 控制，默认 "fail"（请求以错误结束），"recompute" 才回退本地重算 | vllm-src/vllm/config/kv_transfer.py:103-106; vllm/distributed/kv_transfer/kv_connector/v1/base.py:405-423 |
| Q71 | F3 | 130 | CONFIRMED | 【待核】 | 代理脚本路径与参数均正确；注释改为陈述句 | vllm-src/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:89-115; docs/features/nixl_connector_usage.md:103-109 |
| Q72 | F4 | 5 | HEADER | 【待核】 | 页首说明：V2 目录说明标注为 [Experimental]、under active development；改写为已核实声明 | vllm-src/vllm/v1/worker/gpu/README.md:1-3 |
| Q73 | F4 | 44 | CONFIRMED | 【待核】 | 四条设计原则与 V2 实现相符；以指向第 6 节的陈述替换标记，并注明 token 缓冲采用 UVA | vllm-src/vllm/v1/worker/gpu/states.py:9-60; vllm/v1/worker/gpu/input_batch.py:262-705 |
| Q74 | F4 | 74 | CORRECTED | 【待核】 | 逐项给出 v0.30.0 的实际文件与类名；小节标题去除标记，引导句改为陈述 | vllm-src/vllm/v1/worker/gpu/{model_runner.py:187,1631,1997; states.py:9,28; input_batch.py:19,44,311,375,457; block_table.py:17,191,277; attn_utils.py:186,392; cudagraph_utils.py:141,558; sample/sampler.py:33; spec_decode/rejection_sampler.py:77; spec_decode/speculator.py:60} |
| Q75 | F4 | 85 | CORRECTED | 【待核】 | V2 在 v0.30.0 中为默认（环境变量未设置且无不支持特性时）；给出回退条件与强制开关 | vllm-src/vllm/envs.py:300,2037-2038; vllm/config/vllm.py:73-75,675-723,2815-2882; vllm/v1/worker/gpu_worker.py:469-478 |
| Q76 | F4 | 132 | CONFIRMED | 【待核】 | 路径正确 | vllm-src/vllm/v1/worker/gpu/model_runner.py:187 |
| Q77 | R1 | 4 | HEADER | 【待核】 | R1 版本说明：改写为已核实声明 | - |
| Q78 | R1 | 75 | CORRECTED | 【待核：0.30.x 中的后端列表】 | 公共逻辑已移至 mla_attention.py；补全 decode/稀疏后端列表与 prefill/ 子目录 | vllm-src/vllm/model_executor/layers/attention/mla_attention.py:1570,2232,3226; vllm/v1/attention/backends/mla/（目录，无 common.py）; vllm/v1/attention/backends/registry.py:56-147 |
| Q79 | R1 | 99 | CORRECTED | 【待核】 | 以 mamba_cache_mode（none/align/all）写明快照策略 | vllm-src/vllm/config/cache.py:178-186; vllm/v1/kv_cache_interface.py:993 |
| Q80 | R1 | 107 | CONFIRMED | 【待核：0.30.x 中的实现位置】 | 描述正确；补充 v0.30.0 中的实现位置 | vllm-src/vllm/v1/attention/backends/mla/indexer.py:184,239; vllm/model_executor/models/deepseek_v2.py:637-670; vllm/v1/attention/backends/registry.py:78-120 |

## 7. 替换文本全文

各条替换的原文与新文本（markdown 版）见 `edits.json` 的 `old_text_md` / `new_text_md` 字段；逐课差异可用 `diff -r /workspace/feishu-lessons-formal /workspace/feishu-lessons-verified` 查看。

## 8. 未标注处的过时表述

本节为后续一轮核验（2026-10-08，UTC+8）的结果。此前未加【待核】标记的文字中，凡与 v0.30.0 源码明显不符的路径、类名、函数名、参数和环境变量，本轮均已修正。检索方法见 `/workspace/daihe/sweep.py`、`sweep2.py`：

- 提取 21 课中全部行内代码标识符、`VLLM_*` 环境变量与 `--` 参数，逐一在 `/workspace/vllm-src` 中检索是否存在；
- 对同一行中“文件路径 + 符号”的组合，检查该符号是否确实定义在所写的文件中。

修正共 26 处（X01–X22 为第二轮检索结果，X23–X26 为第三轮图示核对时为保持正文与图示一致而补充）（`kind` 为 `extra` / `extra-insert`），均已写入 `/workspace/feishu-lessons-verified/` 与 `edits.json`。保留旧名称的括注，是为了方便阅读旧版本代码的读者对照。

| ID | 课 | 修正内容 | 证据（v0.30.0） |
|---|---|---|---|
| X01 | E3 | E3 学习目标：前向入口 FusedMoE.forward → MoERunner.forward（注明旧名） | vllm-src/vllm/model_executor/layers/fused_moe/layer.py:88-126,198; vllm/model_executor/layers/fused_moe/routed_experts.py:45-100,171,1172-1240; vllm/model_executor/layers/fused_moe/runner/moe_runner.py:227,300-325,595-640,681-700,862; vllm/model_executor/models/deepseek_v2.py:374 |
| X02 | E3 | E3 §4-1：模型中构造 FusedMoE(...) → 调用 FusedMoEFactory(...)，说明其创建 router / RoutedExperts / MoERunner | vllm-src/vllm/model_executor/layers/fused_moe/layer.py:88-126,198; vllm/model_executor/layers/fused_moe/routed_experts.py:45-100,171,1172-1240; vllm/model_executor/layers/fused_moe/runner/moe_runner.py:227,300-325,595-640,681-700,862; vllm/model_executor/models/deepseek_v2.py:374 |
| X03 | E3 | E3 §4-2：fused_moe/layer.py 的 FusedMoE.__init__ → fused_moe/routed_experts.py 的 RoutedExperts.__init__ | vllm-src/vllm/model_executor/layers/fused_moe/layer.py:88-126,198; vllm/model_executor/layers/fused_moe/routed_experts.py:45-100,171,1172-1240; vllm/model_executor/layers/fused_moe/runner/moe_runner.py:227,300-325,595-640,681-700,862; vllm/model_executor/models/deepseek_v2.py:374 |
| X04 | E3 | E3 §4-3：前向调用链改为 MoERunner.forward → moe_forward 自定义算子 → _forward_impl → _apply_quant_method（modular/monolithic 两路） | vllm-src/vllm/model_executor/layers/fused_moe/layer.py:88-126,198; vllm/model_executor/layers/fused_moe/routed_experts.py:45-100,171,1172-1240; vllm/model_executor/layers/fused_moe/runner/moe_runner.py:227,300-325,595-640,681-700,862; vllm/model_executor/models/deepseek_v2.py:374 |
| X05 | E3 | E3 §4-4：FusedMoE.select_experts → fused_moe/router/ 下的路由器类（FusedTopKRouter、GroupedTopKRouter） | vllm-src/vllm/model_executor/layers/fused_moe/router/fused_moe_router.py:12,49; router/fused_topk_router.py:80,127; router/grouped_topk_router.py:80,246; router/base_router.py:159,204-216,260 |
| X06 | E4 | E4 §3 源码要点-1：EplbState.build() → add_model() | vllm-src/vllm/distributed/eplb/eplb_state.py:235,359-366,553,750; vllm/v1/worker/gpu_model_runner.py:5375,5397 |
| X07 | E4 | E4 §3 源码要点-1：负载张量与映射表按模型存放于 EplbModelState | vllm-src/vllm/distributed/eplb/eplb_state.py:105-175 |
| X08 | E4 | E4 §3 源码要点-2：eplb/rebalance_algo.py → eplb/policy/default.py（DefaultEplbPolicy.rebalance_experts，参数 num_gpus → num_ranks） | vllm-src/vllm/distributed/eplb/policy/default.py:21,104,275-330; vllm/distributed/eplb/policy/abstract.py:9-12; vllm/distributed/eplb/policy/__init__.py:10 |
| X09 | E4 | E4 §3 源码要点-4：select_experts 的 EPLB 映射位置 fused_moe/layer.py → fused_moe/router/base_router.py | vllm-src/vllm/model_executor/layers/fused_moe/router/base_router.py:129-160,204-224,260 |
| X10 | B3 | B3 参数表：--max-num-partial-prefills 在 v0.30.0 中已不存在，标注为已移除 | vllm-src/vllm/config/scheduler.py（无 max_num_partial_prefills 字段）; vllm/engine/arg_utils.py（无对应参数）; 全仓库 rg max_num_partial_prefills / max-num-partial-prefills 无结果 |
| X11 | B3 | B3 参数表：--async-scheduling 默认值“视版本而定” → 自动开启（默认 None） | vllm-src/vllm/config/scheduler.py:190-193; vllm/config/vllm.py:1407-1487; vllm/engine/arg_utils.py:361,1644 |
| X12 | D2 | D2 §3-3 能力检查：Marlin 最低计算能力 80 → 75（MarlinLinearKernel.get_min_capability） | vllm-src/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:31-34 |
| X13 | D2 | D2 §3-7 前向计算：gptq_marlin_gemm → ops.marlin_gemm（经 apply_gptq_marlin_linear） | vllm-src/vllm/model_executor/layers/quantization/utils/marlin_utils.py:685,727; vllm/_custom_ops.py:1222-1240（gptq_marlin_gemm 在 vllm/ 与 csrc/ 中均已不存在） |
| X14 | D2 | D2 §6.6：“GPTQ/AWQ 自动升级为 Marlin” → choose_mp_linear_kernel() 逐层选择；Marlin 最低 SM75（原文 Ampere） | vllm-src/vllm/model_executor/kernels/linear/__init__.py:498-508,813-870; vllm/model_executor/kernels/linear/mixed_precision/marlin.py:31-40 |
| X15 | C3 | C3 §4-5：vllm/compilation/pass_manager.py → vllm/compilation/passes/pass_manager.py | vllm-src/vllm/compilation/passes/pass_manager.py:91; vllm/compilation/passes/fusion/rms_quant_fusion.py:773; vllm/compilation/passes/fusion/attn_quant_fusion.py:362; vllm/compilation/passes/fusion/act_quant_fusion.py:283; vllm/compilation/passes/fusion/allreduce_rms_fusion.py:990 |
| X16 | C3 | C3 §4-5：FusionPass → RMSNormQuantFusionPass（passes/fusion/rms_quant_fusion.py） | vllm-src/vllm/compilation/passes/pass_manager.py:91; vllm/compilation/passes/fusion/rms_quant_fusion.py:773; vllm/compilation/passes/fusion/attn_quant_fusion.py:362; vllm/compilation/passes/fusion/act_quant_fusion.py:283; vllm/compilation/passes/fusion/allreduce_rms_fusion.py:990 |
| X17 | C3 | C3 §4-5：AttnFusionPass → AttnQuantFusionPass（passes/fusion/attn_quant_fusion.py） | vllm-src/vllm/compilation/passes/pass_manager.py:91; vllm/compilation/passes/fusion/rms_quant_fusion.py:773; vllm/compilation/passes/fusion/attn_quant_fusion.py:362; vllm/compilation/passes/fusion/act_quant_fusion.py:283; vllm/compilation/passes/fusion/allreduce_rms_fusion.py:990 |
| X18 | C3 | C3 §12 延伸阅读：pass_manager.py、fusion.py → passes/pass_manager.py、passes/fusion/ | vllm-src/vllm/compilation/passes/pass_manager.py:91; vllm/compilation/passes/fusion/rms_quant_fusion.py:773; vllm/compilation/passes/fusion/attn_quant_fusion.py:362; vllm/compilation/passes/fusion/act_quant_fusion.py:283; vllm/compilation/passes/fusion/allreduce_rms_fusion.py:990 |
| X19 | F3 | F3 §12 延伸阅读：kv_connector/v1/nixl_connector.py → kv_connector/v1/nixl/ 子包 | vllm-src/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py; vllm/distributed/kv_transfer/kv_connector/factory.py:152-245 |
| X20 | B2 | B2 §4-5：init_device 中的 torch.cuda.set_device → torch.accelerator.set_device_index | vllm-src/vllm/v1/worker/gpu_worker.py:360,424 |
| X21 | B6 | B6 §0：在先修要求段落之后新增“版本说明”段落（V2 为默认、V1 为回退、参见 F4） | vllm-src/vllm/envs.py:300,2037-2038; vllm/config/vllm.py:675-723,2815-2882; vllm/v1/worker/gpu_worker.py:469-478; vllm/v1/worker/gpu/model_runner.py:187,1631,1997 |
| X22 | B6 | B6 §4：在介绍 GPUModelRunner 处注明其为 V1 实现，V2 位置见 F4 | vllm-src/vllm/v1/worker/gpu/model_runner.py:187; vllm/config/vllm.py:675-723 |
| X23 | E2 | E2 §PP 消除气泡：max_concurrent_batches 属于 VllmConfig 而非 Executor，异步调度 + V2 时为 PP+1（与图 E2-fig2 一致） | vllm-src/vllm/config/vllm.py:589-599; vllm/v1/engine/core.py:210-237 |
| X24 | E2 | E2 §源码：max_concurrent_batches 的位置 multiproc_executor.py → config/vllm.py | vllm-src/vllm/config/vllm.py:589（multiproc_executor.py 中已无 max_concurrent_batches） |
| X25 | B6 | B6 §4 第 10 步：细化首轮 Q18 的表述，区分同步/异步调度下草稿的回传方式（与图 F1-fig1/fig2 一致） | vllm-src/vllm/v1/engine/core.py:621-628,727-735; vllm/v1/core/sched/async_scheduler.py:16,42-44; vllm/v1/core/sched/scheduler.py:2404-2460 |
| X26 | C1 | C1 §源码-4：补入 KV 写入算子 unified_kv_cache_update（与图 C1-fig2 一致） | vllm-src/vllm/model_executor/layers/attention/attention.py:540-575,716-740; vllm/v1/attention/backends/triton_attn.py:316,756; vllm/v1/attention/backends/flash_attn.py:320,1438 |

### 8.1 不确定、未作修改的项

- **B1 第 78 行 `DecodeStream`**：这是 `tokenizers` 库的类，并非 vLLM 自身符号。`vllm/v1/engine/detokenizer.py:23,62` 确实使用它，且要求 tokenizers ≥ 0.22.0。表述正确，未修改。
- **B1 第 117 行 `LLM.generate()` 内部的 `_run_engine()`**：该方法现定义于 `vllm/entrypoints/offline_utils.py:581` 的 mixin 中，由 `LLM` 继承（`vllm/entrypoints/llm.py:67`）。原文未写文件位置，表述仍成立，未修改。
- **D2 第 43 行“GPTQ Marlin 在此将 checkpoint 布局的 `qweight` 重排为 Marlin kernel 所需的布局”**：v0.30.0 中重排实际由 `MarlinLinearKernel.process_weights_after_loading()` 调用 `ops.gptq_marlin_repack` 完成（`kernels/linear/mixed_precision/marlin.py:107`）。原文按功能描述，且未写错误的符号名，属措辞层面的差异，未修改。
- **E3 第 49 行 `expert_map` 的语义**：一般情况下与原文一致，即非本卡专家为 −1（`fused_moe/expert_map_manager.py:51,71`）。但 ROCm AITER kernel 改用 0/1 的 `expert_mask`（`routed_experts.py:216-225`）。这属于平台特例，未补充。
- **B6 第 4 节逐步流程中的细节**（如 `drafter.propose(...)`）：仅核对了名称存在性，未逐行对照 V1 runner 的全部实现。B6 已加版本说明，指出本课讲解的是 V1 路径。
- **课件中的练习符号**（如 `BlockPoolExhausted`、`KVTransferSM`、`DelayedFreeTracker`、`RejectResult`、`ReqRepBus`、`Handshake` 等）：均为练习中自定义的名称，不属于 vLLM 源码，检索时予以排除。
- **以“旧版本中为……”括注保留的旧名称**（如 `VLLM_ALL2ALL_BACKEND`、`VLLM_ATTENTION_BACKEND`、`VLLM_TORCH_PROFILER_DIR`、`gpu_cache_usage_perc`、`P2pNcclConnector`、`SharedStorageConnector`、`FusedMoE`、`FusionPass`、`AttnFusionPass`、`rebalance_algo.py`、`--eplb-window-size` 等）：属于有意保留的对照说明。
- **检索方法的局限**：本轮检索只能发现“名称已不存在或位置不符”的问题，无法发现名称仍存在但语义或默认值已变化的表述。对于默认值类表述，除已核对的 B3 参数表（`--max-num-seqs` 为 256～1 024，`--long-prefill-token-threshold` 为 0，均与 `arg_utils.py:2730-2792`、`config/scheduler.py:70` 一致）外，未作系统性复核。


## 9. 图示核对

本节为 2026-10-08（UTC+8）对全部 45 幅图示的核对结果。Mermaid 源文件位于 `/workspace/mmdc/src/<ID>/<ID>-figN.mmd`，图中标签逐项核对了两方面：

- 是否与 v0.30.0 源码（`/workspace/vllm-src`）一致，核对对象包括标识符、路径、参数、环境变量与行为描述；
- 是否与已更正的课件正文（`/workspace/feishu-lessons-verified/`）一致。

仍然存在的名称不作修改，例如 `LLMEngine.step`、`FusedMoEMethodBase`、`Fp8LinearMethod`、`cutlass_scaled_mm`、`RayDistributedExecutor`、`WorkerWrapperBase`、`rpc_broadcast_mq`、`FreeKVCacheBlockQueue`、`find_longest_cache_hit`、`CudagraphDispatcher`、`initialize_cudagraph_keys`、`PostGradPassManager`、`START_DP_WAVE`、`execute_dummy_batch`、`rearrange_expert_weights_inplace`、`VLLM_NIXL_SIDE_CHANNEL_PORT`，以及 A1 图 2 中的各条日志文本。改动遵循以下原则：只修正确属过时或与正文不一致的标签，保持原有布局，标签尽量简短。修改前的源文件与 PNG 备份于 `/workspace/daihe/diagrams/`。

### 9.1 图示改动（共 19 处标签修改，涉及 14 幅图；另有 F4 图 1、图 2 因源文件已删除【待核】而重新渲染）

| 图 | 修改内容 | 原标签 | 新标签 | 证据（v0.30.0） |
|---|---|---|---|---|
| B1-fig1 | 前端组件名统一为 InputProcessor（与 B1 正文 Q05 一致） | IP["Processor / InputProcessor<br/>tokenize + 校验"] | IP["InputProcessor<br/>tokenize + 校验"] | vllm-src/vllm/v1/engine/input_processor.py:38; vllm/v1/engine/async_llm.py:146 |
| B1-fig2 | 参与者 Processor → InputProcessor | participant P as Processor | participant P as InputProcessor | vllm-src/vllm/v1/engine/input_processor.py:38,281 |
| B1-fig2 | 每步执行拆分为 execute_model 与 sample_tokens（与 B1 正文 Q06 一致） | BL->>BL: executor.execute_model() | BL->>BL: executor.execute_model() → sample_tokens() | vllm-src/vllm/v1/engine/core.py:589-619 |
| B2-fig2 | torch.cuda.set_device → torch.accelerator.set_device_index（与 B2 正文 X20 一致） | init_device()（set_device、init_distributed_environment、建 TP/PP 组） | init_device()（set_device_index、init_distributed_environment、建 TP/PP 组） | vllm-src/vllm/v1/worker/gpu_worker.py:360,424,432 |
| B4-fig2 | save_new_computed_blocks → allocate_new_computed_blocks | M->>C: save_new_computed_blocks()（命中块 touch：ref+1，移出空闲队列） | M->>C: allocate_new_computed_blocks()（命中块 touch：ref+1，移出空闲队列） | vllm-src/vllm/v1/core/kv_cache_coordinator.py:233; vllm/v1/core/kv_cache_manager.py:571（save_new_computed_blocks 已不存在） |
| B5-fig1 | 删除已迁出主仓库的 BitsAndBytes / GGUF 加载器，补入仍在注册表中的 Tensorizer（与 D2 正文 Q36 一致） | OT["ShardedStateLoader / BitsAndBytes / GGUF / RunAI streamer ..."] | OT["ShardedStateLoader / Tensorizer / RunAI streamer ..."] | vllm-src/vllm/model_executor/model_loader/__init__.py:50-66; docs/features/quantization/bnb.md:6-13; docs/features/quantization/gguf.md:6-13 |
| B6-fig1 | Processor → InputProcessor | LLMEngine / AsyncLLM + Processor + OutputProcessor | LLMEngine / AsyncLLM + InputProcessor + OutputProcessor | vllm-src/vllm/v1/engine/input_processor.py:38 |
| B6-fig2 | Processor → InputProcessor | add_request() × N（Processor 转为 EngineCoreRequest） | add_request() × N（InputProcessor 转为 EngineCoreRequest） | vllm-src/vllm/v1/engine/llm_engine.py; vllm/v1/engine/input_processor.py:281 |
| B6-fig2 | 忙循环调用 step_fn：启用异步调度（默认）或 PP>1 时为 step_with_batch_queue（与 B1 正文第 7 步一致） | C->>K: （另一进程）run_busy_loop → step() | C->>K: （另一进程）run_busy_loop → step_fn()（step 或 step_with_batch_queue） | vllm-src/vllm/v1/engine/core.py:210-237,630; vllm/config/vllm.py:589-599 |
| B6-fig2 | 补入 sample_tokens 调用：ModelRunnerOutput 由 sample_tokens 返回（与 B6 正文 Q18 一致） | X->>R: Worker.execute_model → GPUModelRunner.execute_model⏎    R-->>X: ModelRunnerOutput | X->>R: Worker.execute_model → GPUModelRunner.execute_model⏎    K->>X: sample_tokens(grammar_output)⏎    X->>R: GPUModelRunner.sample_tokens⏎    R-->>X: ModelRunnerOutput | vllm-src/vllm/v1/engine/core.py:589-619; vllm/v1/worker/gpu_model_runner.py:4187,4566 |
| C1-fig2 | KV 写入在 Triton/FlashAttention 等后端中已从 Impl.forward 拆出，改由 unified_kv_cache_update → do_kv_cache_update 先行完成 | L->>I: forward(layer, q, k, v, kv_cache, attn_metadata, output)⏎    I->>I: reshape_and_cache 写新 K/V → 调 kernel 计算注意力 | L->>I: do_kv_cache_update（经 unified_kv_cache_update 写入新 K/V）⏎    L->>I: forward(layer, q, k, v, kv_cache, attn_metadata, output)⏎    I->>I: 调用 kernel 计算注意力 | vllm-src/vllm/model_executor/layers/attention/attention.py:540-575,716-740; vllm/v1/attention/backends/triton_attn.py:316,756-790; vllm/v1/attention/backends/flash_attn.py:320,1438 |
| C2-fig2 | dispatch 签名：接收 num_tokens 等参数，返回 (mode, BatchDescriptor)（与 B6 正文 Q17 一致） | MR->>D: dispatch(BatchDescriptor(num_tokens, uniform_decode)) | MR->>D: dispatch(num_tokens, uniform_decode, ...) | vllm-src/vllm/v1/cudagraph_dispatcher.py:235-243; vllm/v1/worker/gpu_model_runner.py:3999-4006 |
| E1-fig1 | 引擎上报的统计含 KV 使用率（与 E1 正文 Q40 的打分公式一致） | EC0 -- "队列长度 (waiting, running)" --> COORD | EC0 -- "负载统计 (waiting, running, KV 使用率)" --> COORD | vllm-src/vllm/v1/engine/coordinator.py:142; vllm/v1/engine/core_client.py:1546-1597 |
| E2-fig2 | max_concurrent_batches 取值（与 B2 正文 Q09 一致） | Note over EC,S1: max_concurrent_batches = PP 大小，两个 stage 同时忙碌 | Note over EC,S1: max_concurrent_batches ≥ PP 大小（异步调度 + V2 时为 PP+1），两个 stage 同时忙碌 | vllm-src/vllm/config/vllm.py:589-599 |
| E4-fig2 | FusedMoE 类已不存在，前向入口为 MoERunner（与 E3 正文 X01-X04 一致） | participant M as FusedMoE 前向（每层） | participant M as MoE 层前向（MoERunner） | vllm-src/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:227,681; vllm/model_executor/layers/fused_moe/layer.py:88 |
| E4-fig2 | rebalance 算法 → DefaultEplbPolicy（与 E4 正文 X08 一致） | participant A as rebalance 算法 | participant A as EPLB 策略（DefaultEplbPolicy） | vllm-src/vllm/distributed/eplb/policy/default.py:21,275; vllm/distributed/eplb/policy/__init__.py:10 |
| E4-fig2 | 参数 num_gpus → num_ranks | S->>A: rebalance_experts(负载, 副本总数, 组数, 节点数, GPU 数) | S->>A: rebalance_experts(负载, 副本总数, 组数, 节点数, rank 数) | vllm-src/vllm/distributed/eplb/policy/default.py:275-283 |
| F1-fig1 | 草稿不随 ModelRunnerOutput 返回：同步调度时经 take_draft_token_ids 交回调度器，异步调度（默认）时留在 Worker 侧 | DR -->\|"下一步的 spec_token_ids"\| SCH | DR -->\|"下一步草稿：同步调度经 take_draft_token_ids 交回；<br/>异步调度时留在 Worker，调度器仅预留占位"\| SCH | vllm-src/vllm/v1/engine/core.py:621-628,727-735; vllm/v1/core/sched/scheduler.py:2404-2460; vllm/v1/core/sched/async_scheduler.py:16,42-44; vllm/v1/outputs.py:324-363,430 |
| F1-fig2 | ModelRunnerOutput 不含 spec_token_ids，改为注释说明草稿的回传方式（与 B6 正文 Q18 及 X25 一致） | R-->>S: sampled_token_ids=[d1,d2,y3]，spec_token_ids=[e1,e2,e3] | R-->>S: ModelRunnerOutput：sampled_token_ids=[d1,d2,y3]⏎  Note over S,R: 新草稿 [e1,e2,e3] 不在输出中：同步调度时经 take_draft_token_ids() 交回；异步调度时留在 Worker | vllm-src/vllm/v1/outputs.py:324-363（ModelRunnerOutput 无 spec_token_ids 字段）; vllm/v1/engine/core.py:621-628; vllm/v1/core/sched/async_scheduler.py:16,42-44 |
| F4-fig1 | 源文件中已删除【待核】，子图标题为“ModelRunner V2（GPU 为中心，v0.30.0 默认）”；仅重新渲染 | — | — | vllm/config/vllm.py:675-723（V2 为默认） |
| F4-fig2 | 源文件中已删除【待核】（“V2：CPU 只传最少信息”）；仅重新渲染 | — | — | vllm/v1/worker/gpu/README.md:1-3 |

重新渲染的 PNG 共 16 个：B1-fig1、B1-fig2、B2-fig2、B4-fig2、B5-fig1、B6-fig1、B6-fig2、C1-fig2、C2-fig2、E1-fig1、E2-fig2、E4-fig2、F1-fig1、F1-fig2、F4-fig1、F4-fig2。渲染命令与 `render.sh` 相同（`-b white -s 2`），渲染后已逐一目视检查：中文与符号均正常显示，无缺字方框，文字清晰可读，布局与原图一致。新 PNG 已同步至以下三处：

- `/workspace/feishu-diagrams/<ID>/`
- `/workspace/feishu-lessons-verified/images/<ID>/`
- 归档中的 `course/revised/images/<ID>/`

此外，`/workspace/feishu-diagrams/manifest.json` 中对应图示的 `width_px` / `height_px` 已更新为新尺寸，其余字段未变。

**图题**：各图的含义未发生变化，课件正文中的图题（及 feishu-paste-verified 中的对应文本）均未修改。

**为保持正文与图示一致而补充的正文修改**：见第 8 节 X23–X26。

- X23、X24（E2）：`max_concurrent_batches` 属于 `VllmConfig`，异步调度与 V2 同时启用时为 PP+1。
- X25（B6）：区分同步与异步调度下草稿 token 的回传方式。这是对首轮 Q18 表述的细化，因为在异步调度（默认）下，草稿并不经 `take_draft_token_ids()` 交回调度器。
- X26（C1）：补入 KV 写入算子 `unified_kv_cache_update`。

### 9.2 核对后未修改、但需说明的项

- **C1-fig2**：选择器注释中的“block_size”仍是后端选择的依据，只是在 v0.30.0 中改为在函数内部从 `cache_config` 读取，不再是参数。图中的表述不构成错误，因此保留。
- **C2-fig1**：连线标签“运行时：BatchDescriptor(num_tokens, uniform_decode)”描述的是分发器内部用于查找的键（`dispatch()` 内部构造 `BatchDescriptor` 后查表，见 `cudagraph_dispatcher.py:235-260`），与正文 C2 第 99 行一致，因此保留。
- **F2-fig2**：“torch.profiler.start()”对应 `--profiler-config` 中 `profiler="torch"` 的情形；若设置为 `cuda`，则改为调用 CUDA profiler API。图中描述的是常用路径，因此保留。
- **F3-fig2**：“num_computed_tokens = 已加载数”是简化写法。按正文 Q66，当已加载数等于请求总长度时，调度器会将其回退为 `len−1`。这一细节由正文说明，图中未改。
- **E4-fig2**：“完成后原子切换映射表”。`EPLBConfig.use_async` 默认为 `true`，即重排以非阻塞方式进行，但映射表最终仍是整体切换，因此保留。
- **B3 正文第 106 行**：“请求的 `spec_token_ids` 由上一步的 drafter 产生”。在异步调度（默认）下，调度器侧的 `spec_token_ids` 实际是值为 −1 的占位（`async_scheduler.py:16,42-44`），真实草稿保留在 Worker 侧。这属于概念层面的简化，未修改，建议课程负责人决定是否补充说明。
- **B5-fig1**：加载器列表以“...”结尾，未列出 ModelExpress、IPC cache 等较新的加载器，不构成错误。



## 10. 补充更正

- B3「投机解码（F1）」段：原文称 `spec_token_ids` 均由上一步 drafter 产生；已区分同步调度与默认启用的异步调度（异步调度下调度器仅保留 −1 占位符，草稿保留在 Worker 端），依据见第 9 节 F1 图示核对（`core.py:621-628`、`async_scheduler.py:16,42-44`）。
