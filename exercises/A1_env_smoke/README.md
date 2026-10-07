# A1 · 环境与 serve 参数冒烟 / Env smoke

**课纲**: 模块 A · 对应原第 1 课  
**目标**: 解析假 `vllm serve` CLI 参数，并做版本字符串检查 stub。

## 验收标准 / Acceptance

1. `parse_serve_args` 能解析 `--model`、`--port`、`--tensor-parallel-size`（短名 `-tp` 亦可）。
2. 缺 `--model` 时抛 `ValueError`。
3. `check_version_compatible(current, minimum)`：当 current >= minimum（semver 主.次.补）返回 True。
4. `pytest exercises/A1_env_smoke -q` 通过。

## 文件

- `env_smoke.py` — 参考实现（含 TODO 注释）
- `starter/env_smoke.py` — 学员空白版
