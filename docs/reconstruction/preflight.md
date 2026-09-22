# BG6022-v3 R0 preflight

阶段：R0
方案版本：R0-1.0
采集日期：2026-09-22（Asia/Hong_Kong）
工程状态：`ready_for_r1_repair_after_review_followup`
用户接受状态：`awaiting_user_acceptance`
tested_commit：`979042eb5054e6130987c525aff7b6dbba0517b3`
documentation_commit：本次证据更新随 tested_commit 后的文档提交发布；最终提交哈希以 Git 推送记录为准

本文是 R0 的唯一准备记录。它只记录本次在 Windows 工作机上实际核对到的事实；总方案、R0 方案中的模板和未执行的目标不被当作已验证证据。

## 1. 基线、工作区与数据边界

| 项目 | 实际值 |
|---|---|
| reference_base_commit | `72b0c3cffe514a84d46dd681f8e3fb4f7aa9341f` |
| actual_base_commit | `72b0c3cffe514a84d46dd681f8e3fb4f7aa9341f` |
| 原工作区 | `E:\BG6022-v3` |
| 原工作区分支 | `codex/v3-m3-extension` |
| 原工作区基线状态 | 无已跟踪、已暂存或未暂存修改；有未跟踪本地诊断文件 |
| R0 worktree | `E:\BG6022-rebuild` |
| R0 分支 | `codex/structural-rebuild` |
| R0 worktree 创建点 | `72b0c3cffe514a84d46dd681f8e3fb4f7aa9341f` |
| origin/codex/v3-m3-extension | `72b0c3cffe514a84d46dd681f8e3fb4f7aa9341f` |
| origin/main | `79685f8adee3d25a92b3a9e57a8242be4522afe0` |
| 远程 | `https://github.com/az87988799/BG6022-v3.git` |

原工作区中的未跟踪内容仅登记、不导入、不删除：`bg6022_diagnose.py`、`bg6022_diagnostic_20260916_205211_f0cefb/`、`bg6022_fix_smoke_20260916_2220/`、`bg6022_fix_smoke_20260916_2230/`。它们包括本地 Run/Artifact/Session 运行数据，不属于 R0 候选补丁。没有使用 stash、reset、clean 或强制覆盖。

R0 前的进程检查没有观察到名称为 ORCA 的进程，也没有观察到路径指向 `E:\BG6022-v3` 或 `E:\BG6022-rebuild` 的项目 Python/ORCA 进程；Codex 自身后台进程不作为项目执行证据。该检查是当时的进程快照，不替代未来 Run 的 PID、创建时间和进程树事实。

旧数据根 `E:\BG6022-v3-data` 在本机存在。R0 使用的测试配置全部把 `data_root` 解析到各自的 pytest 临时目录；项目同步环境、wheel 冒烟环境和证据根均在仓库外，未读取或改写旧数据根。

## 2. 方案入口与范围决议

已将用户指定文件原样纳入仓库：

| 文件 | 来源 | SHA-256 |
|---|---|---|
| `docs/reconstruction/master_plan.md` | `E:\chrome\MASTER_PLAN (1).md`，SR-1.0 | `98B9E4629FEA5F972C215668A396FABB09F43C75448E137359E499A8191E8566` |
| `docs/reconstruction/stages/R0.md` | `E:\chrome\R0_IMPLEMENTATION_PLAN.md`，R0-1.0 | `CBA7E5D7884C71C9F9681146C993BF9067902C1CEF5D807AACCDED870F84E6E6` |

仓库 `AGENTS.md` 现以 `docs/reconstruction/master_plan.md` 为总方案入口，并指出 `docs/reconstruction/stages/R0.md` 和 `docs/reconstruction/preflight.md` 的职责。用户的 R0 请求授权本阶段实施；R0 方案仍然限制实际变更范围，不授权 R1 及后续生产迁移。

| 决议 | 本阶段状态与行为 | 最晚决定/依赖阶段 |
|---|---|---|
| 继承 V3 底座，不做空白重写 | 已按本仓库基线执行 | R0 已决定 |
| 旧格式活动 Run 是否跨版本续跑 | `pending`；R0 只读，不迁移、不删除、不续跑 | R1 首次修改持久化/版本边界前 |
| 真实 PubChem/LLM/ORCA | R0 未运行；只做离线、替身来源和 ORCA 启动边界探针 | R1 首次 live 验证或真实计算验收前 |
| `max_plan_steps=12`、`max_llm_calls_per_run=20` | `proposed/pending`；未写入当前配置或 schema | 实施规划/运行预算的首个阶段前 |
| prompt loader 抽取 | `deferred`；R0 只完成调用方盘点，未改生产 loader | 发生生产导入路径重构前；不阻塞 R0 |
| 推送目标 | 本阶段验证后推送 `codex/structural-rebuild`；不合并 `main`、不强推 | R0 收尾 |

## 3. 环境、配置与证据隔离

| 项目 | 实际值/结果 |
|---|---|
| 目标解释器 | `C:\Users\阿妆\AppData\Local\Programs\Python\Python311\python.exe`，Python 3.11.4 |
| 默认 `python` | Python 3.14.6；未用于 R0 项目测试 |
| uv | 初始不在 PATH；在 `E:\BG6022-r0-tools` 一次性工具环境安装 `uv 0.12.17` |
| 项目环境 | `E:\BG6022-r0-venv`，由锁文件同步 |
| 证据根 | `E:\BG6022-r0-evidence-20260922` |
| wheel 冒烟环境 | `E:\BG6022-r0-evidence-20260922\wheel-smoke-venv` |
| wheel 冒烟工作目录 | `E:\BG6022-r0-wheel-smoke` |
| live 网络/ORCA | 未启用；离线命令使用 `--disable-socket` 和排除三个 live marker |
| 锁文件 | `uv.lock --check` 通过，SHA-256 `4EBCFF69853F57680C368E061B57E1EA0A1CD2171EBFB8E426C95A5B40128BF3` |

执行 `uv` 时 shell 报告了继承的 `VIRTUAL_ENV=E:\BG6022-r0-tools` 与项目环境不一致的提示；每条项目命令显式清除了该变量并指定 `UV_PROJECT_ENVIRONMENT=E:\BG6022-r0-venv`，实际测试解释器为上表的 Python 3.11.4。项目依赖由 `uv sync --frozen --group dev --python 3.11` 安装，未改 `uv.lock`、`pyproject.toml` 或依赖版本。

当前配置默认值实际核对为：4 cores、1024 MB、`maxcore_mb=192`、并发 1、单 attempt 1200 s、Run 活动时间 3600 s、输出 64 MiB、工作目录 512 MiB。配置示例中的 `E:\BG6022-v3-data` 仅被读取用于确认旧数据边界；没有将 R0 测试指向该路径。

## 4. 原始基线与构建证据

原始基线在新增 R0 文件和测试之前运行；日志和 JUnit XML 位于外部证据根，命令的退出码另存为同名 `.exitcode` 文件。

| 命令/筛选 | 结果 |
|---|---|
| `uv run --frozen python -m pytest --collect-only -q` | 391 tests collected，退出码 0 |
| `pytest -m "not live_orca and not live_llm and not live_pubchem" --disable-socket -ra` | 375 passed，16 deselected，退出码 0 |
| `pytest -m "windows_process and not live_orca and not live_llm and not live_pubchem" --disable-socket -ra` | 2 passed，389 deselected，退出码 0；在本机 Windows 执行 |
| `uv run --frozen ruff check .` | All checks passed，退出码 0 |
| `uv run --frozen ruff format --check .` | 87 files already formatted，退出码 0 |
| `uv run --frozen python -m compileall -q src` | 通过，退出码 0 |
| `git diff --check` | 通过，退出码 0 |
| `uv build --out-dir <external evidence>\dist` | source distribution 与 wheel 均构建成功，退出码 0 |

构建出的唯一 wheel 在仓库外的独立环境中以 `python -I` 安装；`bg6022.__file__` 指向 `wheel-smoke-venv\Lib\site-packages`，不是源码目录。四份提示词在 wheel 中的 SHA-256 与源码资源一致：

```text
intake.md  cc7d326fab9f8d3067eee8e38c1e4d5c7bd80fd96bb676de02c400f163bfa8ab
planner.md c88dc578545b2ebbc2add3b80fb73d70183ed56934e8a9fed984671b5b8deee7
answer.md  0cbd124a53b81f44481b9f3e4fba88f805befaea80b3a33073f196914942a353
repair.md  b5220c281ec8936ae2e79462bd63f5f16e1dc00e0686ba1d593b62c2974fe6d2
```

wheel 冒烟同时执行 `python -I -m bg6022 --help`，成功显示 CLI 子命令。没有由此启动 ORCA、LLM 或 PubChem。

提交 `b472c58c4ba732d6086702bc15add830351ed8a9` 的最终候选回归重新执行为：396 items collected；离线选择 380 项，其中 377 passed、3 strict xfailed、16 deselected，退出码 0；Windows 选择集 2 passed、394 deselected，退出码 0；ruff check、ruff format、compileall、`git diff --check` 和候选 source/wheel 构建均退出码 0。最终候选 wheel 在新的仓库外环境中再次以 `python -I` 导入，四份提示词 hash 保持一致，CLI `--help` 成功。

评审收尾提交 `979042eb5054e6130987c525aff7b6dbba0517b3` 只改测试，不改生产源码、提示词、锁文件或科学 fixture。该提交的 follow-up 证据为：399 items collected；离线选择 383 项，其中 378 passed、5 strict xfailed、16 deselected，退出码 0；Windows 选择集 2 passed、397 deselected，退出码 0；缺口文件单独运行得到 1 passed、5 xfailed，退出码 0；同一文件 `--runxfail` 得到 5 failed、1 passed，退出码 1；C6H14 分流/身份补充文件 4 passed，退出码 0；lock check、ruff check、format check、compileall、`git diff --check` 和 source/wheel build 均退出码 0。新 wheel `bg6022-0.1.0-py3-none-any.whl` 的 SHA-256 为 `9f4ac3e38349323c7b00a92cae64ae37abb1ce5b119e13a3824ab0da219ef780`。

本次 follow-up 目标环境仍为 Windows/Python 3.11.4；没有据此声称 Linux 全绿。无关 ValueError 入口突变在 xfail 标记保持存在时仍以 1 failed、退出码 1 结束，证明未知故障未被预期失败吞掉。日志均位于 `E:\BG6022-r0-evidence-20260922`，相对文件名、SHA-256 和退出码见第 10 节。

## 5. 保留合同与实际测试映射

以下是 R0 盘点用的代表性 nodeid；它们来自 collect-only 或当前源码，不把待 R2 替换的旧限制当作新合同。

| 合同 | 已有/新增实际断言 |
|---|---|
| BC01 新请求分流 | `tests/unit/test_turn_routing.py::test_new_complete_request_is_not_swallowed_by_waiting_identity_run` 与新增 `::test_new_complete_request_is_not_swallowed_by_waiting_formula_candidate_run`；后者使用旧 C6H14 候选等待快照，比较 Request/Plan/目标/候选/授权语义，新请求为水 Opt→Freq 且确认前 ORCA 启动数为 0 |
| BC02 身份补充 | `tests/unit/test_turn_routing.py::test_identity_supplement_updates_same_waiting_run_with_structured_intake` 现在显式断言原始身份、目标和 `execution_permission`；新增 `::test_identity_supplement_keeps_real_advance_until_confirmation` 覆盖真实 advance、PubChem 替身、RDKit 几何和 ORCA 哨兵 |
| BC03 身份替换 | `tests/unit/test_formula_input.py::test_wrong_bare_smiles_does_not_replace_waiting_formula_or_change_target`；`::test_invalid_original_smiles_is_not_silently_replaced_by_model_case`；`::test_name_not_found_supplement_updates_lookup_only_on_same_run` |
| BC04 词法边界 | `tests/unit/test_formula_input.py::test_explicit_smiles_forms_are_extracted_once_and_never_formula_normalized`；`::test_formula_labels_skip_horizontal_space_and_keep_complete_value`；`::test_formula_suffixes_are_rejected_as_whole_tokens` |
| BC05 候选与来源 | `tests/unit/test_formula_input.py::test_structure_identity_binding_allows_selected_cid_and_rejects_other_queries`；`::test_formula_resolution_does_not_auto_accept_unverified_source_record`；`::test_formula_resolution_reports_incomplete_search_even_with_one_candidate` |
| BC06 授权 | `tests/unit/test_execution_contract.py::test_tool_requires_explicit_permission_and_accepted_content`；`tests/unit/test_chat_controls.py::test_chat_prepares_then_waits_for_one_confirmation`；`tests/unit/test_m3_extension.py::test_rejected_method_update_does_not_mutate_waiting_run` |
| BC07 电子态和参数 | `tests/unit/test_m1_minimal_repairs.py::test_invalid_electronic_state_is_not_coerced_or_dropped_as_absent`；`::test_negated_or_conflicting_spin_words_do_not_become_parameters`；`tests/unit/test_root_cause_contracts.py::test_unresolved_requirements_block_but_deferred_charge_and_multiplicity_do_not` |
| BC08 原始结构 | `tests/unit/test_session.py::test_artifact_copy_and_run_json_are_atomic`；`tests/unit/test_m3_public_output.py::test_verified_initial_geometry_is_delivered_and_reindexed`；`tests/unit/test_parser.py::test_valid_d_exponent_and_input_hash_check_are_explicit` |
| BC09 科学结果 | `tests/unit/test_parser.py::test_failed_output_keeps_evidence_and_does_not_succeed`；`::test_opt_does_not_publish_geometry_when_a_later_scf_is_unbound`；`tests/unit/test_frequency_support.py::test_energy_results_keep_opt_sp_and_frequency_categories_distinct`；`::test_missing_hessian_cannot_succeed` |
| BC10 查询与交付 | `tests/unit/test_context_query.py::test_catalog_selects_the_indexed_water_result_without_falling_back_to_latest`；`tests/unit/test_generic_output_contracts.py::test_result_answer_rejects_free_prose_and_renders_verified_file_content`；`tests/unit/test_m3_public_output.py::test_dynamic_output_contract_is_strict_and_public` |
| BC11 日志 | `tests/unit/test_rdkit_diagnostics.py::test_rdkit_logging_filters_known_console_warning_but_keeps_file_diagnostics`；`::test_rdkit_helper_preserves_stderr_and_geometry_tool_writes_raw_diagnostic` |
| BC12 控制与恢复 | `tests/unit/test_chat_controls.py::test_chat_prepares_then_waits_for_one_confirmation`；`tests/unit/test_runner.py::test_cancel_request_stops_process_tree_and_clears_execution_guard`；新增 `tests/unit/test_execution_gap_repros.py::test_cancel_between_confirmation_and_tool_start_prevents_start` |

当前限制与新缺口分开：`tests/unit/test_m3_extension.py::test_one_chat_request_cannot_contain_two_distance_steps` 等现有“唯一 operation/单 distance”断言在 R0 保留，R2 才按新任务实例合同替换；R0 新增的执行边界探针不被当作永久正确行为。

## 6. 当前写入归属与目标负责人

| 字段/职责 | 当前写入位置（实际函数） | 目标负责人 | R0 动作 |
|---|---|---|---|
| Request/Plan 与修订 | `Agent.handle_message`；`planner.request_from_intake`（`planner.py:701`）、`proposal_to_plan`（`:495`）、`validate_request_plan`（`:566`）；`Agent._apply_parameter_update`、`_apply_molecule_clarification`、`_try_repair` | 唯一协调链 | 盘点，不迁移 |
| `execution_permission` 与授权 hash | `Agent._create_chat_run`、`Agent.confirm`、identity/parameter update 路径；`Agent.advance` 的 config fallback；`tools.orca._check_execution_contract` | 单一授权校验入口 | 记录所有入口，未增加第二套授权 |
| attempt 号、prepared/started/finished | `tools.orca._execute_prepared_attempt`（`:366`）及 `on_started`；本地 molecule/PubChem/geometry Tool 各自的 `run.attempts.append`；`Agent._reserve_attempt` 维护预算计数 | 一处生命周期管理入口 | 记录顺序和双写风险，不迁移 |
| Run/Step 状态、当前结果索引 | `Agent.advance`（`:709`）统一推进/`step_status`/`current_results`；ORCA adapter 在 prepared 阶段也写 `step_status` 和 attempt；`cancel`/repair/update 路径修改状态 | 协调链统一语义 | 登记 Tool 与 Agent 双重处理风险 |
| PID、创建时间、进程退出事实 | `orca.runner.run_orca`（`:62`）和 `ProcessFacts`；ORCA Tool 的 `on_started` 回调把 facts 写入 attempt/guard | runner | 保留事实来源 |
| 数值和科学检查 | `orca.parser.inspect_attempt`（`:111`）、`orca.checks.evaluate_success`（`:19`）、frequency parser、`tools.orca.execute_orca_step` | ORCA 领域适配器 | 不改科学语义 |
| JSON/Artifact 落盘 | `session.save_run`（`:323`）、`save_result`（`:345`）、`register_file_artifact`（`:353`）、`register_bytes_artifact`（`:395`）、`atomic_write_json/bytes`；Agent 和各 Tool 是调用方 | 单一持久化入口 | 记录异常、原子性和多调用方缺口 |

重点接口清单：

| 接口 | 输入/返回 | 副作用和当前异常边界 | 调用方/测试/后续阶段 |
|---|---|---|---|
| `Agent.advance(run, cancel)` | `Run`、可选 `Event` → `Result | None` | 保存 Run，调用 Tool，保存 Result；Tool `RuntimeError` 和 Result 落盘 `OSError` 当前可越出并留下 active 状态 | `handle_message`、`confirm`、clarification/update；新增 gap tests；R1 统一边界 |
| `Agent.confirm(run)` | Run/id → `AgentResponse` | 写授权快照/hash 后调用 `advance`；并发确认没有一次性消费锁 | CLI/chat、确认测试；R1 修复授权消费 |
| `Agent.cancel(run)` | Run/id → `AgentResponse` | 设置请求/Run Event；等待态直接写 cancelled；实际进程停止由 Tool/runner 观察 Event | chat controls、runner Windows tests；R1 复核竞态 |
| `Tool.execute(step, run, cancel)` | 单个 Step/Run/Event → Result | 执行边界；无 implementation 会抛 RuntimeError；不应让模型越过该接口执行命令 | Tool registry、Agent；R1 统一异常闭合 |
| `session.save_run` / `save_result` | data root + domain object → path/None | 原子 JSON 写入，可抛 `OSError`；当前调用方未统一将落盘失败转成终态诊断 | Agent/Tool/session tests；R1 统一持久化错误处理 |
| `orca.execute_orca_step` / `runner.run_orca` | ORCA 配置、Step、Run、Event → Result/ProcessFacts | 创建 attempt 文件、execution guard、进程事实、原始输出和科学 checks；runner 负责进程树和退出事实 | ORCA/runner tests；R1 保持单一实现 |

R0 没有创建 `ExecutionService`、`Kernel`、`Store` 或第二个工作流引擎。

## 7. 缺口登记与 R1 交接

缺口测试在强制无标记等价运行（`--runxfail`）时保存原始失败；最终测试只对确认的既有缺口使用 `strict=True`，并把 `raises` 限定为各自的专用 `AssertionError` 子类。每个注入记录命中次数、异常类型/消息和 Tool 调用；未知异常、前置条件失败、线程超时和清理失败不会落入 xfail。Event/Barrier 探针均在 `finally` 中释放并有界收拢 worker。无关 ValueError 突变的普通失败日志另行保存，xfailed 不计入通过。

| gap_id | 最小复现与当前实际行为 | 期待行为 | 阶段/风险 |
|---|---|---|---|
| `R1-GAP-TOOL-UNEXPECTED-EXCEPTION` | `tests/unit/test_execution_gap_repros.py::test_unexpected_tool_exception_is_closed_at_tool_boundary` 参数化 `RuntimeError`/`TypeError`；两种测试 Tool 异常均命中一次并越出 `Agent.advance`，当前 Run/Step 不能形成失败诊断闭合 | 普通 Tool 异常在 Tool 边界被结构化记录，Run/Step 终止且不交付成功结果，原始诊断保留 | R1 优先；可能使真实任务卡在 `running` |
| `R1-GAP-RESULT-PERSISTENCE` | `::test_result_persistence_failure_is_acknowledged_and_stops_successor`；隔离 monkeypatch 让 `save_result` 抛 `OSError`，当前异常越出且 successor 不能获得持久化闭合事实 | 保存失败被承认并终止后续步骤，Run 有明确持久化诊断；不能声称 Result 成功 | R1 优先；影响结果可追溯性 |
| `R1-GAP-RUN-PERSISTENCE` | 新增 `::test_run_persistence_failure_stops_successor_without_claiming_success`；在首个 Result 已写入、后继 Tool 尚未启动的 Run 快照上注入一次 `save_run` `OSError`，当前异常越出，调用列表只有首步且 Run 不能宣称成功 | Run 写入失败被承认并形成可观察诊断；磁盘持续不可写时不要求伪造成功的 Run 文件更新 | R1 持久化错误闭合；与 Result persistence 同一批处理 |
| `R1-GAP-CONFIRM-RACE` | `::test_duplicate_confirmation_cannot_consume_one_waiting_authorization_twice`；Barrier 同步两个确认，当前同一个等待授权启动同一 Tool 两次 | 授权绑定 revision/消费边界，一次有效确认最多启动一次；重复消息不重复执行 | R1 优先；真实 ORCA 重复启动风险 |
| `R0-PROBE-CANCEL-BOUNDARY` | `::test_cancel_between_confirmation_and_tool_start_prevents_start`；Barrier/事件在确认写入与 advance 之间取消，当前探针通过，工具启动数为 0 | 保持该保护并在 R1 重构后回归 | R0 保护，非缺口 |

优先级之外的探针设计已登记但未在 R0 扩大范围：下游产物缺失/篡改、来源/模型中断。R0 没有触碰真实 ORCA、LLM 或外部 PubChem，所以没有把 live 行为伪装成离线证据。上述四个 R1 gap_id 仍归入三类 R1 工作包：Tool 异常、Run/Result 持久化、确认授权竞态。

## 8. 可选整理、候选范围与回滚

没有执行 R0-C1 prompt loader 抽取。当前 `planner.py:278` 的 `load_prompt` 被 `answer.py`、`repair.py` 和 Planner 使用；虽然抽取可能是小改动，但会改变生产导入路径，且 R0 的文档＋测试＋隔离目标已经可以独立交付，因此将其后置，不把它混入执行生命周期缺口修复。

R0 候选文件只包含：

- `AGENTS.md`：总方案入口和 R0 边界；
- `docs/reconstruction/master_plan.md`、`docs/reconstruction/stages/R0.md`、`docs/reconstruction/preflight.md`：方案和证据入口；
- `tests/unit/test_turn_routing.py`：一条不替换 `advance` 的真实离线身份链；
- `tests/unit/test_execution_gap_repros.py`：隔离缺口探针和严格 xfail。

冻结文件未改：`uv.lock`、`pyproject.toml`、`config.example.toml`、四份 prompts、`src/bg6022` 生产源码、ORCA input/runner/parser/checks/profiles、已有旧科学样本。回滚方式是对上述 R0 候选提交做可审查的 `git revert`；不使用强制 reset，不删除原工作区未跟踪数据，不覆盖 `E:\BG6022-v3-data`。

已在干净临时 worktree `E:\BG6022-r0-rollback-rehearsal-final` 从 `979042e` 演练 `git revert --no-edit 979042e`：生成回滚提交 `37e2d08948c6df45e0283c699a4be038abca82de`，回滚后 `git status --porcelain` 为空且 `git diff --check` 无输出，随后移除该临时 worktree。演练没有触碰 `E:\BG6022-v3`、其未跟踪诊断文件或 `E:\BG6022-v3-data`；完整输出及退出码见 `followup-rollback-rehearsal.log`。

## 9. R0 验收门快照

| 门 | R0 结果 |
|---|---|
| G01 基线是确认的开发提交 | pass：reference/actual/base、分支和远端已核对 |
| G02 原工作区和旧数据未被无授权修改 | pass：独立 worktree、临时 data roots；原工作区诊断文件保留 |
| G03 仓库内单一总方案和阶段入口 | pass：相对路径入口、源文件 hash 已记录 |
| G04 Windows/Python 3.11/锁依赖 | pass（本机证据）：Python 3.11.4、lock check、frozen sync、wheel；不冒充其他用户机器 |
| G05 原始离线/Windows/构建基线 | pass：原始命令、退出码、JUnit/XML、wheel smoke 已保存 |
| G06 关键保留行为映射 | pass：新增精确 C6H14 候选等待→水 Opt→Freq 场景，比较旧 Run 的 Request/Plan/目标/候选/授权语义；真实 advance 身份补充链新增并通过 |
| G07 状态和接口归属 | pass：当前函数、目标负责人和后续阶段已列出 |
| G08 异常/写盘/竞态探针 | pass as evidence：Tool RuntimeError/TypeError、Result 写入、Run 写入、确认竞态均命中明确注入；取消保护通过；线程在 finally 中有界收拢 |
| G09 未把缺口隐藏为成功 | pass：专用 `raises` 类型只接受对应已知 gap；无关 ValueError 突变退出码为 1；`--runxfail` 的 5 个缺口均为普通失败 |
| G10 候选无不明退化 | pass：`979042e` 后同条件离线/Windows/静态/构建回归无新增普通失败；预期 gap 不计入 passed |
| G11 loader 专门验收 | not applicable：可选整理未执行 |
| G12 回滚、外部调用和数据政策 | pass for R0：live 调用 pending，干净 worktree revert 演练通过，回滚和数据边界及 pending 最晚阶段已记录 |

## 10. Follow-up 证据索引

证据根为 `E:\BG6022-r0-evidence-20260922`；以下路径均相对该目录。每个命令的退出码保存在同名 `.exitcode` 文件，日志 SHA-256 用于审查时确认未被替换。

| 日志 | 命令/范围 | 结果与退出码 | SHA-256 |
|---|---|---:|---|
| `followup-collect.log` | `uv run --frozen python -m pytest --collect-only -q` | 399 collected / 0 | `e6d97f1e1d71730b10eb5c77cb79e8907b164a12e7a513b6c6ebf1bbc4543362` |
| `followup-offline.log` | 离线选择：`not live_orca and not live_llm and not live_pubchem` + `--disable-socket` | 378 passed、5 xfailed、16 deselected / 0 | `00ffe236fe3c39bd6ae07c2c5593ecb3634e2ab6d36af216dd8d1a2378fe653f` |
| `followup-windows.log` | Windows 选择：`windows_process` + 三个非 live marker + `--disable-socket` | 2 passed、397 deselected / 0 | `d0a081e538beb74dc337595b97875c77f3ab63d10f0d5f57f6477ac66366a29e` |
| `followup-gap-xfail.log` | `tests/unit/test_execution_gap_repros.py` | 1 passed、5 xfailed / 0 | `7407c662339f01c5a71889190173a00c7d962ba439a8fe8c84128fd24a9c9b02` |
| `followup-gap-runxfail.log` | 同一缺口文件 `--runxfail` | 5 failed、1 passed / 1（预期） | `879b23a5d1abd5f1240893607fdee71f2ee0c9b086c9b90cfa126ec306a3be7a` |
| `followup-turn-routing.log` | `tests/unit/test_turn_routing.py` | 4 passed / 0 | `65f6a6b3a989ed961bc31820b5a188da75bb7c6cd73ef3e82052538fce7432a5` |
| `followup-mutation-xfail-scope.log` | 入口无关 `ValueError` 突变，保留 xfail 标记 | 1 failed / 1（预期） | `fbca3702a4339d62fefc09c635a1583bc3a531a29516ce03dd6af0bb34ad205a` |
| `followup-lock-check.log` | `uv lock --check` | 通过 / 0 | `5c6e993601615baaac0e3dc6fb73fc19d5019eb640206110761751c28d8ca154` |
| `followup-ruff-check.log` | `uv run --frozen ruff check .` | 通过 / 0 | `a4443afdcfb6d7363adb285762515ccf7cf50473b1a05c20c1a50f6bed4d26b0` |
| `followup-ruff-format.log` | `uv run --frozen ruff format --check .` | 通过 / 0 | `8541720920065c6f4abafc25ae8007259670007e69795a0b6e56ce957601e373` |
| `followup-compile.log` | `uv run --frozen python -m compileall -q src` | 通过 / 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `followup-diff-check.log` | `git diff --check` | 通过 / 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `followup-build.log` | `uv build --out-dir followup-dist-979042e` | sdist/wheel 成功 / 0 | `84a8ba8418ed88f38f97ffe75d2be42075525951f6e0987fb161ec2a840f1b85` |
| `followup-rollback-rehearsal.log` | clean worktree `git revert --no-edit 979042e` | 回滚后 clean、diff check 通过 / 0 | `f9f69a329821106c535a032078e0df5e10bdf775a635cc1fcd2b0ea0a975a0f7` |

`followup-gap-runxfail.log` 和 `followup-mutation-xfail-scope.log` 的非零退出码是故障探针的预期证据，不是候选回归失败。外部构建产物 `followup-dist-979042e/bg6022-0.1.0-py3-none-any.whl` SHA-256 为 `9f4ac3e38349323c7b00a92cae64ae37abb1ce5b119e13a3824ab0da219ef780`；未把 wheel、Run、Artifact、密钥或大日志提交到仓库。

## 11. 交接与未运行项目

测试内容已提交为 `979042eb5054e6130987c525aff7b6dbba0517b3`；本文件和小型证据摘要随本次推送补齐。R0 不自行进入 R1，不启动真实 ORCA，不宣称用户机器或全链可用。

R1 第一批工作应从本文件中的三个真实 gap 重新开始：统一 Tool 普通异常闭合、Result/Run 持久化失败闭合、确认授权的单次消费/并发边界。所有修复必须继续保留当前真实 ORCA input/runner/parser/checks 的单一生产实现、4 cores/1024 MB/`%maxcore 192`/并发 1 约束和原始文件诊断。

用户明确验收前，本阶段只能报告为 `awaiting_user_acceptance`，不能标记为已完成。

## 12. R1 执行边界实施记录（2026-09-22）

本节追加 R1 的实际实施事实，不改写上面的 R0 历史记录。R1 详细方案已复制到仓库入口 [`stages/R1.md`](stages/R1.md)，源文件 SHA-256 为 `4566D4F066C91D6D9F22AC2C8173D45DAC51C9CF9F2E3DDE4B6DBBA1E6E96684`。用户当前请求明确授权开始 R1；R0 的用户验收状态仍保持未接受。

| 项目 | R1 实际事实 |
|---|---|
| 隔离工作树 | `E:\BG6022-rebuild` |
| 分支/基线 | `codex/structural-rebuild`；R1 开始时 `5f0719d0567757fedae93d2d1c058e4cee553a6a` |
| 依赖/冻结项 | `uv.lock`、prompts、ORCA input/runner/parser/checks、配置科学标准未改；R1 未增加 Task、`result_field`、新 dialogue 或旧格式自动迁移 |
| 运行边界 | Run owner 为短生命周期跨线程/跨进程锁；Agent 负责一次生命周期收口；Tool hooks 负责参数准备、preflight、attempt 预算；生产工具的 Run checkpoint 统一经 `src/bg6022/execution.py` |
| 失败收口 | 普通 Tool `Exception`、Result 写盘失败、Run checkpoint 失败均形成结构化 `pending_data`；候选 Result 在 Run checkpoint 前不进入 `current_results`；连续写盘失败只做一次 best-effort checkpoint，并标记 `not_persisted` |
| 确认/取消 | 确认 admission 在 `_begin_request` 前完成；同一授权最多进入一个 Tool；竞争确认返回占用状态；取消不会改写其他 owner 正在持有的 Run |
| attempt | ORCA、分子、PubChem、几何 Tool 共用 `execution.allocate_attempt`，同时检查 durable attempt 记录和既有目录，避免重启覆盖 raw 文件 |

### R1 验证快照

| 验证 | 结果 |
|---|---|
| R1 gap 回归 | 9 passed：普通 RuntimeError/TypeError、Result/Run 持久化、取消边界、同 Agent 重复确认、跨 Agent 确认 owner、已有成功结果保留、连续磁盘失败有界 |
| 离线全套 | 386 passed、16 deselected、退出码 0；JUnit/log 在 `E:\BG6022-r1-evidence-20260922\r1-final-offline.*` |
| Windows 进程测试 | 2 passed、400 deselected、退出码 0；日志在 `E:\BG6022-r1-evidence-20260922\r1-final-windows.log` |
| 静态/构建 | `ruff check .`、`ruff format --check .`、`compileall`、`git diff --check`、`uv lock --check`、`uv build` 均通过 |
| 真实 ORCA | 独立数据根 `E:\BG6022-r1-live-20260922` 完成一次授权水分子 SP；Run `run_f2250da3c18d45ed836243d0039242f4` succeeded，能量 `-76.417246084177 Eh`，输入为 4 cores / `%maxcore 192`，process tree empty；日志在 `E:\BG6022-r1-evidence-20260922\r1-live-orca-sp-actual.log` |

R1 当前报告状态为 `implementation_verified_live_pending` / `awaiting_user_acceptance`。未执行真实 LLM 或 PubChem，不以离线结果冒充全链外部服务验收；仍需用户审阅本次提交及证据后明确接受，才能标记 R1 为完成。
