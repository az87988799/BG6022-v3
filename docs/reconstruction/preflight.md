# BG6022-v3 R0 preflight

阶段：R0
方案版本：R0-1.0
采集日期：2026-09-22（Asia/Hong_Kong）
工程状态：`ready_for_r1_repair`
用户接受状态：`awaiting_user_acceptance`
tested_commit：`b472c58c4ba732d6086702bc15add830351ed8a9`
documentation_commit：`pending`（本次证据摘要更新）

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

| 决议 | 本阶段状态与行为 |
|---|---|
| 继承 V3 底座，不做空白重写 | 已按本仓库基线执行 |
| 旧格式活动 Run 是否跨版本续跑 | `pending`；R0 只读，不迁移、不删除、不续跑 |
| 真实 PubChem/LLM/ORCA | R0 未运行；只做离线、替身来源和 ORCA 启动边界探针 |
| `max_plan_steps=12`、`max_llm_calls_per_run=20` | `proposed/pending`；未写入当前配置或 schema |
| prompt loader 抽取 | `deferred`；R0 只完成调用方盘点，未改生产 loader |
| 推送目标 | 本阶段验证后推送 `codex/structural-rebuild`；不合并 `main`、不强推 |

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

## 5. 保留合同与实际测试映射

以下是 R0 盘点用的代表性 nodeid；它们来自 collect-only 或当前源码，不把待 R2 替换的旧限制当作新合同。

| 合同 | 已有/新增实际断言 |
|---|---|
| BC01 新请求分流 | `tests/unit/test_turn_routing.py::test_new_complete_request_is_not_swallowed_by_waiting_identity_run`；新 Run、旧等待 Run、Opt→Freq 输入和 ORCA 启动数 |
| BC02 身份补充 | `tests/unit/test_turn_routing.py::test_identity_supplement_updates_same_waiting_run_with_structured_intake`（旧测试含 advance 替身）；新增 `::test_identity_supplement_keeps_real_advance_until_confirmation` 覆盖真实 advance、PubChem 替身、RDKit 几何和 ORCA 哨兵 |
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

当前限制与新缺口分开：`tests/unit/test_m3_extension.py::test_one_chat_request_cannot_contain_two_distance_steps` 等现有“唯一 operation/单 distance”断言在 R0 保留，R2 才按新任务实例合同替换；R0 新增的三项执行边界探针不被当作永久正确行为。

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

缺口测试在第一次无标记运行时保存原始失败；最终测试只对确认的既有缺口使用 `strict=True`、明确 `raises=AssertionError` 的 xfail。`--runxfail` 的复跑日志也保存在证据根，xfailed 不计入通过。

| gap_id | 最小复现与当前实际行为 | 期待行为 | 阶段/风险 |
|---|---|---|---|
| `R1-GAP-TOOL-UNEXPECTED-EXCEPTION` | `tests/unit/test_execution_gap_repros.py::test_unexpected_tool_runtime_error_is_closed_at_tool_boundary`；测试 Tool 抛 `RuntimeError`，当前异常越出 `Agent.advance`，Run/Step 不能形成失败诊断闭合 | 普通 Tool 异常在 Tool 边界被结构化记录，Run/Step 终止且不交付成功结果，原始诊断保留 | R1 优先；可能使真实任务卡在 `running` |
| `R1-GAP-RESULT-PERSISTENCE` | `::test_result_persistence_failure_is_acknowledged_and_stops_successor`；隔离 monkeypatch 让 `save_result` 抛 `OSError`，当前异常越出且 successor 不能获得持久化闭合事实 | 保存失败被承认并终止后续步骤，Run 有明确持久化诊断；不能声称 Result 成功 | R1 优先；影响结果可追溯性 |
| `R1-GAP-CONFIRM-RACE` | `::test_duplicate_confirmation_cannot_consume_one_waiting_authorization_twice`；Barrier 同步两个确认，当前同一个等待授权启动同一 Tool 两次 | 授权绑定 revision/消费边界，一次有效确认最多启动一次；重复消息不重复执行 | R1 优先；真实 ORCA 重复启动风险 |
| `R0-PROBE-CANCEL-BOUNDARY` | `::test_cancel_between_confirmation_and_tool_start_prevents_start`；Barrier/事件在确认写入与 advance 之间取消，当前探针通过，工具启动数为 0 | 保持该保护并在 R1 重构后回归 | R0 保护，非缺口 |

优先级之外的探针设计已登记但未在 R0 扩大范围：下游产物缺失/篡改、来源/模型中断。R0 没有触碰真实 ORCA、LLM 或外部 PubChem，所以没有把 live 行为伪装成离线证据。

## 8. 可选整理、候选范围与回滚

没有执行 R0-C1 prompt loader 抽取。当前 `planner.py:278` 的 `load_prompt` 被 `answer.py`、`repair.py` 和 Planner 使用；虽然抽取可能是小改动，但会改变生产导入路径，且 R0 的文档＋测试＋隔离目标已经可以独立交付，因此将其后置，不把它混入执行生命周期缺口修复。

R0 候选文件只包含：

- `AGENTS.md`：总方案入口和 R0 边界；
- `docs/reconstruction/master_plan.md`、`docs/reconstruction/stages/R0.md`、`docs/reconstruction/preflight.md`：方案和证据入口；
- `tests/unit/test_turn_routing.py`：一条不替换 `advance` 的真实离线身份链；
- `tests/unit/test_execution_gap_repros.py`：隔离缺口探针和严格 xfail。

冻结文件未改：`uv.lock`、`pyproject.toml`、`config.example.toml`、四份 prompts、`src/bg6022` 生产源码、ORCA input/runner/parser/checks/profiles、已有旧科学样本。回滚方式是对上述 R0 候选提交做可审查的 `git revert`；不使用强制 reset，不删除原工作区未跟踪数据，不覆盖 `E:\BG6022-v3-data`。

## 9. R0 验收门快照

| 门 | R0 结果 |
|---|---|
| G01 基线是确认的开发提交 | pass：reference/actual/base、分支和远端已核对 |
| G02 原工作区和旧数据未被无授权修改 | pass：独立 worktree、临时 data roots；原工作区诊断文件保留 |
| G03 仓库内单一总方案和阶段入口 | pass：相对路径入口、源文件 hash 已记录 |
| G04 Windows/Python 3.11/锁依赖 | pass（本机证据）：Python 3.11.4、lock check、frozen sync、wheel；不冒充其他用户机器 |
| G05 原始离线/Windows/构建基线 | pass：原始命令、退出码、JUnit/XML、wheel smoke 已保存 |
| G06 关键保留行为映射 | pass：BC01–BC12 代表性 nodeid；真实 advance 身份补充链新增并通过 |
| G07 状态和接口归属 | pass：当前函数、目标负责人和后续阶段已列出 |
| G08 异常/写盘/竞态探针 | pass as evidence：三项缺口可控复现，一项取消保护通过 |
| G09 未把缺口隐藏为成功 | pass：缺口 strict xfail；不计入 passed |
| G10 候选无不明退化 | pass：`b472c58` 后同条件离线/Windows/静态/构建/wheel 回归无新增普通失败 |
| G11 loader 专门验收 | not applicable：可选整理未执行 |
| G12 回滚、外部调用和数据政策 | pass for R0：live 调用 pending，回滚和数据边界已记录 |

## 10. 交接与未运行项目

测试内容已提交为 `b472c58c4ba732d6086702bc15add830351ed8a9`；本文件和小型证据摘要的 documentation commit 在推送前补齐。R0 不自行进入 R1，不启动真实 ORCA，不宣称用户机器或全链可用。

R1 第一批工作应从本文件中的三个真实 gap 重新开始：统一 Tool 普通异常闭合、Result/Run 持久化失败闭合、确认授权的单次消费/并发边界。所有修复必须继续保留当前真实 ORCA input/runner/parser/checks 的单一生产实现、4 cores/1024 MB/`%maxcore 192`/并发 1 约束和原始文件诊断。

用户明确验收前，本阶段只能报告为 `awaiting_user_acceptance`，不能标记为已完成。
