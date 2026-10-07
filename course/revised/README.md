# 《vLLM 从入门到精通 · 修订版》深度课件（feishu-lessons-deep）

> 版本锚定：**vLLM 0.30.x（V1 引擎）**｜练习仓库：https://github.com/Wedsonlin/vllm_code （PR #1 合并前使用分支 `cursor/vllm-course-exercises-75c8`）
> 本目录是 `/workspace/feishu-lessons/`（浅版）的**逐篇深度重写**，文件名与浅版一一对应，用于在飞书中**替换**同名文档正文。操作步骤见 `00-粘贴说明.md`。

## 一、相对浅版的变化

浅版每篇约 3 000～4 400 字符（含英文与代码），以概念介绍为主。深度版每篇统一满足：

1. **3 000+ 个汉字的技术正文**（不含英文、代码与符号；含这些后每篇约 7 000～14 000 字符）；
2. **≥2 张 mermaid 图**：至少一张架构图 + 一张时序/数据流图（部分课含状态图），全部 45 张已用 mermaid-cli 11 实际渲染校验通过；
3. **数字例子**：KV 字节与块数、token 预算分配、TP/GPTQ 分片形状、通信量、padding 浪费、接受长度、PD 传输时间、理论下界等，均可手算复核；
4. **源码走读**：按"模块 → 类 → 方法"的调用顺序书写，锚定 vLLM 0.30.x V1；**不确定的路径/类名/参数一律标注【待核】**，请在 0.30.x tag 上用 `rg` 核对后再定稿；
5. **常见坑 / 故障模式**小节；
6. **精确的 pytest 命令**，对应 `exercises/<目录>`；文中出现的全部 19 条命令已在练习仓库参考实现上实跑通过（C1 的 GPU 用例在无 CUDA 时按设计 skip）；
7. 文首统一版本行；**无广告**；外部推理框架不作为主路径出现，仅 R1 末尾以"其他框架可对照、非主路径"一句带过。

## 二、文件映射与字数统计

"汉字数"只统计 CJK 统一汉字；"总字符"为 Python `len()` 字符数（含英文、代码、符号、空白）。

| 序 | 本地文件 | 飞书子文件夹 | 飞书文档标题（替换目标） | 练习 | 原课 | 汉字数 | 总字符 | mermaid |
|---|---|---|---|---|---|---|---|---|
| 1 | `A1-环境搭建.md` | 模块A-上手与调试 | A1-环境搭建与离线在线推理 | A1_env_smoke | 第1课 | 3029 | 9891 | 2 |
| 2 | `B1-Engine与流式执行.md` | 模块B-运行时内核 | B1-Engine与流式执行 | B1_zmq_patterns | 第2课 | 3048 | 13892 | 2 |
| 3 | `B2-Worker与Executor.md` | 模块B-运行时内核 | B2-Worker与Executor | B2_executor_handshake_sim | 第3课 | 3069 | 12642 | 3 |
| 4 | `B3-调度器.md` | 模块B-运行时内核 | B3-调度器 | B3_scheduler_token_budget | 第4课 | 3048 | 11769 | 2 |
| 5 | `B4-PagedAttention.md` | 模块B-运行时内核 | B4-PagedAttention与显存 | B4_block_pool | 第5课 | 3094 | 10197 | 3 |
| 6 | `B5-ModelRunner加载.md` | 模块B-运行时内核 | B5-ModelRunner与权重加载 | B5_weight_loader_stub | 第6课 | 3016 | 12561 | 2 |
| 7 | `B6-架构总览.md` | 模块B-运行时内核 | B6-V1架构总览与推理主路径 | B1–B5 综合回归 | 第7课 | 3029 | 11112 | 2 |
| 8 | `C1-Triton算子.md` | 模块C-算子与图优化 | C1-Triton算子入门与Attention后端 | C1_triton_vector_add | 第8课(Triton) | 3012 | 10018 | 2 |
| 9 | `C2-CUDAGraph.md` | 模块C-算子与图优化 | C2-CUDA Graph | C2_cudagraph_constraints | 第9课 | 3074 | 9938 | 2 |
| 10 | `C3-模型编译.md` | 模块C-算子与图优化 | C3-torch.compile与vLLM编译栈 | 复用 C2 | 第17课 | 3013 | 9102 | 2 |
| 11 | `D1-量化基础.md` | 模块D-量化 | D1-量化基础 | D1_quant_math | 第10课 | 3061 | 8343 | 2 |
| 12 | `D2-量化实践.md` | 模块D-量化 | D2-vLLM量化模块实践 | D2_quant_config_parse | 第11课 | 3028 | 11123 | 2 |
| 13 | `E1-数据并行.md` | 模块E-分布式 | E1-数据并行DP | E1_dp_lb_modes | 第12课 | 3065 | 8976 | 2 |
| 14 | `E2-张量与流水并行.md` | 模块E-分布式 | E2-张量并行TP（及PP补全） | E2_tp_shard_math | 第13课 | 3001 | 9516 | 2 |
| 15 | `E3-专家并行.md` | 模块E-分布式 | E3-专家并行EP | E3_moe_dispatch_sim | 第14课 | 3003 | 8747 | 2 |
| 16 | `E4-EP负载均衡.md` | 模块E-分布式 | E4-EP负载均衡 | 复用 E3 + 文内扩展 | 第15课 | 3050 | 7030 | 2 |
| 17 | `F1-采样与投机解码.md` | 模块F-高级特性与性能 | F1-采样与投机解码 | F1_rejection_sampler_sim | 第16课 | 3026 | 9064 | 2 |
| 18 | `F2-性能分析.md` | 模块F-高级特性与性能 | F2-性能分析与瓶颈定位 | F2_profiler_checklist | 第18课(加厚) | 3046 | 9596 | 2 |
| 19 | `F3-PD分离.md` | 模块F-高级特性与性能 | F3-PD分离部署 | F3_kv_state_machine | 第19课(新精修) | 3004 | 10500 | 3 |
| 20 | `F4-ModelRunnerV2.md` | 模块F-高级特性与性能 | F4-ModelRunnerV2 | B3+B4+C2 组合回归 | 第20课 | 3046 | 7258 | 2 |
| 21 | `参考-Attention架构.md` | 参考阅读 | R1-Attention架构变体（非主路径） | 可选回归 B4 | 番外篇 | 3035 | 7752 | 2 |
| | **合计** | | | | | **63797** | **209027** | **45** |

## 三、练习命令（在练习仓库根目录执行）

```bash
pip install -r requirements-exercises.txt
python -m pytest exercises/<目录> -q          # 单课
python -m pytest -q                            # 全部
```

各课的确切命令写在每篇的"动手练习"一节。无独立练习目录的课：B6（B1～B5 综合回归）、C3（复用 C2）、E4（复用 E3 + 文内扩展题）、F4（B3+B4+C2 组合回归）、R1（可选 B4）。

## 四、【待核】项说明

以下几类内容在 vLLM 近几个版本中变动频繁，文中均已就地标注【待核】，定稿前建议逐条核对 0.30.x 源码：

- 前端 `Processor` / `InputProcessor` 命名，OpenAI serving 类的目录位置（B1）；
- `execute_model` 是否拆分出 `sample_tokens`、async scheduling 默认值（B1、B2、B6）；
- `max_num_batched_tokens` 等默认值（B3）；注意力后端选择方式（环境变量 vs 配置项，C1）；
- `cudagraph_mode` 默认值与默认捕获尺寸序列（C2）；编译 `mode`/`level` 命名（C3）；
- 量化方法自动升级规则、kernel 选择器（D2）；DP 参数名与负载打分权重（E1）；
- all2all 后端可选值、双批次重叠开关（E3）；EPLB 配置字段（E4）；
- 投机解码方法名与指标名、内置 logits processor 列表（F1）；Prometheus 指标名与 profiler 开启方式（F2）；
- NixlConnector 部分块处理、异构 TP 约束、失败回退策略、示例代理路径（F3）；
- **F4 ModelRunner V2 的全部具体路径/类名/开关**（全篇按设计思路讲解，实现细节均待核）。

## 五、目录内其他文件

- `00-粘贴说明.md`：飞书"替换正文"的操作步骤与检查清单（不需要粘贴到飞书）。
- `README.md`：本文件（不需要粘贴到飞书）。
