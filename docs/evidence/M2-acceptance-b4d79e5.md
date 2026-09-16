# BG6022-v3 M2 阶段验收记录

**状态：等待用户验收。** 本记录完成候选版本的修复、验证和证据整理，
不替用户将 M2 标记为已验收，也不授权进入 M3。

## 验收候选与环境

| 项目 | 值 |
|---|---|
| 分支 | `codex/v3-m2-implementation` |
| 代码提交 | `b4d79e50c072e23a3d8f0939b379db4bfb808630` |
| 验收记录提交 | `b750a1f4b3329001c0bd59e8b0553773ef94dd2d` |
| 修复范围 | 确认页身份展示；未修改 ORCA、Planner、结构绑定或修复执行链 |
| Python / RDKit | Python 3.11.4 / RDKit 2026.03.6 |
| LLM | `deepseek-flash`；真实调用元数据保存在隔离会话记录中 |
| ORCA | 6.1.1（本轮 U1 仅到确认页，不启动计算） |
| 资源 | 4 核、总内存 1024 MB、`%maxcore 192`、并发 1 |
| 期限与预算 | 单次 1200 秒、Run 3600 秒；单步最多 3 次、额外 ORCA 最多 3 次 |
| 配置与 U1 原始记录 | `E:\BG6022-v3-data\acceptance_m2_u1_20260916` |

Windows `doctor --probe-orca` 在上述隔离目录通过，记录了 ORCA 6.1.1、
配置资源、预期缺少输入文件的退出码 2，以及 `process_tree_empty=true`。

## 本轮修复：U1 身份展示

修复仅作用于确认页的展示函数。名称为 `water`、`Water` 或 `O` 时不再
丢弃已有 SMILES；名称、分子式或 SMILES 单独缺失时保留其余事实并明确
显示缺失字段。完整身份继续使用同行标题格式，例如：
`Hexane  C₆H₁₄  (SMILES:CCCCCC)`。

| 用例 | 证据 | 结果 |
|---|---|---|
| U1-1 水分子名称变体 | `tests/unit/test_chat_presentation.py`；真实水请求 `run_1c8b369a1ff046cb9e143de5c9056881` | 通过。`water`、`Water`、`O` 均保留 `H₂O` 和 `SMILES:O`；真实 `Water` 确认页输出已核对并取消 |
| U1-2 正己烷 | `tests/unit/test_chat_presentation.py`；真实己烷请求 `run_177fd8102ba54c94a9cb923add683ed2` | 通过。确认页输出 `Hexane  C₆H₁₄  (SMILES:CCCCCC)`，目标为 Opt 末态电子能，已核对并取消 |
| U1-3 名称缺失 | 展示回归测试的 `formula + canonical_smiles` 事实 | 通过。显示 `名称未提供`，仍显示分子式和 SMILES |
| U1-4 SMILES 缺失 | 展示回归测试的 `title + formula` 事实 | 通过。显示 `SMILES:未提供`，不编造连接关系 |
| U1-5 原有确认内容 | 同一展示测试及真实两次确认预览 | 通过。计划步骤、目标、方法、气相、电荷、多重度、资源、修复范围、预算和 `/confirm` 入口保持正确 |

真实 U1 会话的无凭证结果保存在：

- `E:\BG6022-v3-data\acceptance_m2_u1_20260916\u1-live-results.json`
- `E:\BG6022-v3-data\acceptance_m2_u1_20260916\sessions\acceptance_u1_water.json`
- `E:\BG6022-v3-data\acceptance_m2_u1_20260916\sessions\acceptance_u1_hexane.json`

两次真实会话都在 `waiting_for=confirmation` 时执行 `/cancel`；Run 没有
启动 ORCA。记录仅包含模型调用元数据、结构事实、计划和展示文本，
`DEEPSEEK_API_KEY` 值未写入。

## M2 核心验收证据

核心计算链的真实证据继续使用已核对的最终复验目录
`E:\BG6022-v3-data\acceptance_live_20260916_finalplan`。仓库索引
[`M2-final-live-run-index.json`](M2-final-live-run-index.json) 共 417 个文件，
本轮逐项复核结果为：实际文件 417、缺失 0、大小或 SHA-256 不匹配 0、
`credential_values_recorded=false`。

| 验收范围 | 证据与结论 |
|---|---|
| O1–O8 离线组合、频率、结构来源、检查、预算和安全停止 | `tests/unit/test_plan_composition.py`、`test_frequency_parser.py`、`test_frequency_support.py`、`test_m2_runtime.py`、`test_context_query.py`；全部通过 |
| P1–P7 真实规划与确认边界 | [`M2-final-minimal-fix-acceptance.md`](M2-final-minimal-fix-acceptance.md) 及 [`M2-postfix-live-recertification.md`](M2-postfix-live-recertification.md)；未确认不启动 ORCA，历史结构和初始结构边界通过 |
| B1–B8 修复回归与边界 | 同上证据和现有参数、LLM、执行契约测试；有限修复、虚频、预算、取消和重复确认均通过 |
| R1–R5 真实 ORCA | 水 Opt→SP Run `run_8af28d593f034d0ea29ae933c88adf15`；乙醇失败→修复→Freq→SP Run `run_0f3504469c0740e5867fdf31eefa94a8`；其他 R1–R3 见 `M2-postfix-live-recertification.md`。原始 ORCA 输出、输入、结构、Hessian、Result、修复记录和进程清理均已核对 |
| M1 前置能力 | [`M1-agent-evidence.md`](M1-agent-evidence.md)；真实自然语言计算、确认和有界修复证据存在，M1 仍等待用户验收 |

`fffadc7` 和本提交相对上一验收基线只涉及确认展示及测试；因此不因纯
展示修复重算乙醇链或篡改原始 ORCA 证据。真实核心证据的代码、配置、
依赖和文件哈希仍可追溯。

## 离线与 CI 验证

本机使用锁定依赖运行：

```text
248 passed, 2 skipped, 1 deselected
ruff check .                         passed
ruff format --check .                passed (68 files)
python -m compileall -q src          passed
uv build                             passed
git diff --check                     passed
```

确认展示专项测试为 `10 passed`。候选提交的 Windows CI：
[offline 工作流 35057582412](https://github.com/az87988799/BG6022-v3/actions/runs/35057582412)，
依赖安装、pytest、Ruff、编译和构建全部成功。

## 判定

- U1-1 至 U1-5：通过。
- M2 核心离线、真实规划边界、真实 ORCA 和有界修复：沿已核对证据通过。
- 阻断项：当前没有发现新的代码或证据阻断项。
- 未完成事项：等待用户验收 M1 和 M2；本记录不声明用户已接受，也不将 M2 标记为完成。
