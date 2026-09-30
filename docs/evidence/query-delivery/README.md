# 查询交付修复证据（2026-09-30）

用户授权方案：`E:\chrome\BG6022_Query_Delivery_Test_and_Minimal_Repair_Plan_v3.md`。
基线：`4d558556173285fe082bd0c029cae3a7d23b36c4`，分支 `codex/v3-m3-extension`。
本轮状态为**实现与验证后等待用户验收**，不代表整个项目或阶段已经获得验收。

## 改动与行为

- 核心生产修改限于方案列出的 10 个 Python 文件与 3 个提示词；没有新增生产模块。
- Opt 附带偶极矩报告仍只规划一个 Opt。按结构块、数值行和已核验阶段选择证据；配置开关和文献不与数值块同等优先。
- 两入口共用查询、澄清和补槽协议。短回复补充读取方式或性质，恢复原问题与来源；目录、缺失和普通知识问答都是合法业务状态。
- 同一来源的原文与计数视图共用本轮查询编号。重复显示使用最近实际交付的目标；换性质重新建立目标，保留来源焦点。
- 电子数由已核验 XYZ 中的元素及实际执行的整数电荷计数，未登记为正式科学 Result；ORCA 打印的显式、相关、α/β 字段保留各自定义。
- 目标升级先在副本上完整验证，再提交；共享问题不会因另一个性质升级而丢失。旧 QuerySpec 没有使用新字段时不回填字段。
- 历史几何在复制、创建新 Run 前执行完整 Tool-local 校验。有无命名 checks 均需通过；选中单独数值时允许其他附件缺失形成逐目标 partial。
- 元数据目录只读有界 JSON，不完整哈希大型输入；选中内容后共用字节、时间、取消检查。JSON 预览精确计算 UTF-8 编码、空容器与分隔符开销，呈现不重新展开原值。
- 每次首次/纠正 schema 失败保留脱敏字段路径、错误类别和响应摘要；仍只有原来的一次结构化纠正。损坏的待澄清记录提示原因并保留原会话文件。
- `.gitattributes` 为新 fixture 和真实模型回放记录添加 `-text`，确保 Git 不转换换行、破坏保存的 SHA-256。原始计算文件的空白按原字节保留，单独排除空白检查；生产代码仍执行正常检查。

ORCA input、runner、parser、checks、profiles、科学 Tool 与执行授权逻辑没有修改；本轮没有运行新 ORCA。

## 验证记录

完整命令、退出码及日志索引见 [validation.json](validation.json)。测试详情见 [offline.xml](offline.xml)，摘要见 [summary.json](summary.json)。

- 48 个逻辑场景 N01–N36、F-R01–F-R12 对应已通过的参数化测试，映射见 [case-map.json](case-map.json)。既有回归也全部保留。
- 最终离线回归为 701 passed、38 skipped、6 deselected；全部在禁用网络和真实 ORCA 的条件下执行。
- 静态检查、格式检查、compileall、源码包和 wheel 构建、`git diff --check` 均通过。
- 最终真实模型样本：3 条链 × Semantic/Intake 两入口 × 每入口 3 次，18/18。之前一批也为 18/18。
- 128 份真实模型会话记录全部保留，其中 23 份为开发过程中失败的样本。最终 18 份的文件名和摘要哈希由 `summary.json.final_live.files` 精确列出。
- 所有 128 份回放都记录零科学执行且所有复制的 Run 文件前后字节一致。成功重跑没有删除失败记录。
- Git 暂存区的 28 个计算文件和 128 份模型记录均与清单 SHA-256 及本地原字节一致，暂存差异检查通过；见 [git-integrity.json](git-integrity.json)。

最终 18/18 是有限样本结果，不能推断任意自然语言对话的总体可靠率为 100%。L-B 允许模型提出来源澄清后补充一次来源选择；该轮和原问题均保留在会话记录中。

环境：Windows、Python 3.11.4、httpx 0.28.1、pydantic 2.13.5、RDKit 2026.3.6、pytest 9.1.1、ruff 0.16.6。实际配置模型为 `deepseek-flash`；每次最多一次结构化纠正。回放使用现有配置的 4 核、1024 MB、MaxCore 192、单并发资源事实，但禁用科学执行和发布入口。

## 本次真实来源与尚缺资料

本次用户 Run 是 `run_7f9df59c898049fa99bc949ac7ff316e`，原 session 为 `session_c071268a86e34a43999bb6f908195d0f`。13 个登记文件按原字节复制，清单和哈希见 [用户来源 manifest](../../../tests/fixtures/query_delivery/user_water/manifest.json)。

- 原始 stdout SHA-256：`d4c8390841f501a19923ca9d10eec46af790e3f9ed85be1450cc5327ef4fcbf8`，144588 字节。
- 偶极矩原 token：`1.861296656` Debye；该后置电性质块缺少足够的独立最终阶段绑定，故数值可报告而该数值目标标 partial。不会改写原 Run 的成功状态。
- 核对该真实输出最终 SCF 轨道表后才获得 HOMO `-7.4317` eV、LUMO `1.9216` eV、同表派生轨道能隙 `9.3533` eV。没有预设这些实际数值，也没有以人工轨道表替代此 Run。
- 已执行中性 H₂O 的组成计数为 10。q=+1 的 9 电子反例只在测试副本中构造，明确属于 synthetic；不宣称进行了带电水的真实 ORCA 计算。
- 旧水 fixture 的 `1.861363429` 属于另一个来源，不作为本次 Run 的期望值。

L-B 另使用 15 个文件的 [独立两方法 fixture](../../../tests/fixtures/query_delivery/existing_methods/manifest.json)，标为 `repository_fixture`。这是已有真实验证 Run，不是本次失败用户对话的 Run。PBE0 来源数值为 `1.953232539` Debye。首次回放暴露旧 attempt 仅有 `relative_path`、没有 `result_relative_path` 的格式差异；兼容读取仍要求精确 Run/Step/attempt/登记 Artifact 一致，没有编辑旧记录。

原用户电子数失败轮的完整首次/纠正模型响应和字段路径**无法取得**：原 session 仅保存 schema_error 类别。因此本轮不声称已经定位那一轮的唯一根因。新测试与真实回放已保存可定位诊断，验证了合法缺失/澄清/普通问答不再被旧协议误拒绝。

## 证据层次和覆盖补充

`case-map.json` 的 L1 是纯函数或协议单元测试；L2 是固定模型响应经过真实 Agent/查询/会话逻辑，或受控 HTTP mock；L3 是 `live/` 中的实际模型请求。人工数据、固定响应、实际模型读取和真实科学求解不混作同一种证据。

- N01/N27 的 Agent 双入口证据另见 `test_agent_delivers_each_readonly_adapter_with_zero_new_steps`；N23 的 -6/+1 eV 为人工轨道表，额外长精度反例验证 Decimal 差不受默认 28 位精度舍入。
- N20 同时由 `test_pending_survives_restart_and_unrelated_question_but_not_new_session` 覆盖无关问答保留 pending，由 `test_repeat_then_new_property_preserves_source_but_changes_goal` 覆盖明确改题。
- N22 正向方法选择及歧义候选保存由 `test_explicit_method_query_then_ambiguous_source_preserves_candidates` 覆盖两入口；真实 L-B 验证实际方法来源、换性质和重复目标。
- N31/N32 既有 `test_electron_schema_diagnostics_are_redacted_bounded_and_keep_both_calls` 的纠正成功/仍失败两对照，也有 `test_electron_schema_failure_keeps_delivered_focus_and_run` 的双入口持久化及历史交付保留。
- N36 目录、内容、呈现取消分别覆盖；内容 deadline/字节上限、metadata 取消与超限还由 `test_bounded_read_failures_never_deliver_unverified_bytes`、`test_metadata_budget_rejects_oversize_before_read_and_cancels_cached_read` 覆盖。
- F-R08 保留成功复制，覆盖错角色、缺失、变更字节、伪造引用、源 Step 变化；F-R10 覆盖 8 个 raw attempt，以及目录取消、预算和已签发翻页引用。
- F-R12 的正常分页由原 `test_output_views_preserve_context_and_require_explicit_link_only` 保留，三目标共用预算和原值不变由新增预览测试覆盖。
- `test_baseline_generated_acceptance_snapshot_keeps_original_hash` 与 `test_report_waiting_fixture_from_b9eced9_restores_without_reclassification` 均通过。6113161、b9eced9 旧期望哈希没有重写。

## 支持边界与回滚

局部结构化读取限于偶极矩、完整闭壳层整数占据轨道表和明确的打印电子数字段；不支持的 spin/fractional/non-Aufbau/缺行等轨道格式返回原文或 partial。轨道能隙不是激发能。ECP 显式电子数与化学总电子数不互相替代。

未知性质仍可进行有来源的通用原文查询。本次视图不写入 Run.current_results、不作为正式 Tool 输出、不能直接供下游科学 Tool 使用。阶段证据不足时保持 source_only；不能仅因位于文件最后或 Run 成功而升级为最终优化结构的正式性质。

读取预算在协作检查点生效；不宣称操作系统阻塞 I/O、现有解析/验证回调或模型请求可被 5 秒硬中断。普通会话不保存模型原文，只保存摘要与有限诊断；完整脱敏响应仅由测试显式启用捕获。

回滚这组修改可恢复有限原文交付；应保留带 pending 的旧会话文件并要求完整问题，不能将 pending 解释为计算授权。没有新增组合开关、编排框架或数据库迁移。

## 复跑

```powershell
.venv\Scripts\uv.exe run --frozen python -m pytest -m "not live_orca" --disable-socket -q
.venv\Scripts\uv.exe run --frozen python -m pytest tests/live/test_root_fix_live.py -k saved_result_dialogue --run-live-llm --enable-socket --orca-config config.toml -q
```

第二条命令启用真实模型，仍不启用真实 ORCA；复制既有 Run 到临时目录并拦截科学执行/发布入口。模型凭据通过本地配置提供，不进入证据文件。
