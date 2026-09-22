# BG6022-v3 R1 execution-boundary evidence

状态：`implementation_verified_live_pending` / `awaiting_user_acceptance`。实施提交：`b1fc5d59ce2fbc7cd063fbf353af037ad7c60bee`，已推送至 `origin/codex/structural-rebuild`。本文件是 R1 的小型脱敏索引；原始日志、JUnit、临时配置、Run、Artifact 和真实 ORCA 输出均保存在仓库外。

## 范围

R1 将单一 Run 生命周期 owner、确认 admission、Tool 普通异常收口、Result/Run checkpoint 顺序、保守取消以及 Tool-local parameter/preflight/budget hooks 接入现有 v3 底座。没有新增 Kernel/Service/bus/outbox，也没有进入 R2 的 Task、`result_field`、dialogue 或旧格式迁移范围。

实现入口：[`src/bg6022/execution.py`](../../src/bg6022/execution.py)、[`src/bg6022/agent.py`](../../src/bg6022/agent.py)、[`src/bg6022/models.py`](../../src/bg6022/models.py)。方案入口：[`R1.md`](../reconstruction/stages/R1.md)。

## 可复验证据

| 项目 | 结果 |
|---|---|
| 缺口与生命周期回归 | `tests/unit/test_execution_gap_repros.py`：9 passed |
| 全部非 live 测试 | 386 passed、16 deselected、退出码 0 |
| Windows 进程测试 | 2 passed、400 deselected、退出码 0 |
| 代码质量 | `ruff check .`、`ruff format --check .`、`compileall`、`git diff --check` 通过 |
| 依赖/构建 | `uv lock --check` 与 `uv build` 通过；wheel SHA-256 `47C1F4ADC364402A6E1C66D6F3776A9E2E828F078E4122CC9CDA7B626A20D93B` |

## 真实 ORCA 边界

配置和数据根均为本阶段临时路径，未使用旧 `E:\BG6022-v3-data`。一次真实水分子 SP 通过 `bg6022 run-tool single_point --execute` 进入现有 ORCA runner：

- Run：`run_f2250da3c18d45ed836243d0039242f4`
- 状态：`succeeded`
- 电子能：`-76.417246084177 Eh`
- 检查：normal termination、exit code 0、SCF convergence、input hashes match、process tree empty
- 输入/预算：4 cores、1024 MB total、`%maxcore 192`、并发 1
- 原始证据：`E:\BG6022-r1-live-20260922\runs\run_f2250da3c18d45ed836243d0039242f4\`

## 限制和接受

本阶段没有运行真实 LLM/PubChem，也没有声明用户已经接受。R1 代码、测试、文档和推送完成后仍等待用户验收；用户明确接受前不标记为 complete。
