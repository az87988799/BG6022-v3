# 输出与查询根因修复验收记录

状态：实施与验证后等待用户验收；本记录不代表用户已验收。

范围来自用户提供的 `BG6022_Latest_Commit_Review_and_Minimal_Root_Fix_Plan_v2.md`。
实现 A–F；可选 G1/G2 未启用。没有新增生产模块、持久领域对象或科学 Tool，
没有修改 ORCA input/runner/parser/scientific-check 实现。计算预算仍为 4 核、
1024 MB、`%maxcore 192`、单作业并发。

## 结果与根因

- 原先 LLM 已参与需求读取，但后置校验仍要求报告引文出现输出/原文字样，且用整句关键词判定不支持目标。
  现在由瞬态 `intent_items` 表达 compute/report/query/explain/exclude/unresolved，程序核验引文、目标覆盖和生产者绑定。
  自然偶极矩表述只生成 Opt；当前未注册正式偶极矩输出，因此附加读取该生产者 stdout 的报告，不伪装为 verified property。
  若实际 registry 注册同名正式输出，优先升级正式输出。
- 来源目录保留主体、方法、操作、几何、attempt/历史状态；连续压缩不丢标签。
  partial 仅将实际展示的正式结果更新为后续引用焦点；取消不提交新焦点。
- 真实字段与性质统一经实际 registry 解析。并排比较允许 Opt/SP 不同字段，差值继续由已注册派生 Tool 验证。
  先解析全部依赖再处理比较，两个 SP 可以共用一个优化结构。历史结构按 Subject 绑定并在复制前重新验证。
- 正式结果草稿只选引用和展示方式；自由科学断言不再靠“数字曾出现过”放行。
  程序渲染属性、值、单位、检查与精度；仅请求显示精度时舍入，原始 token/Result 不变。
- 表格先取视图再序列化；默认 30 行、最多 50 行/12 列、单值预览 8 KiB、模型输出目录 16 KiB。
  分开记录 full_value_shown 与 requested_scope_complete；完整大表只显示一页时保持 partial。
- session 增加普通 run_history 索引；recent_results 仍只保留 6 个 Run。
  历史浏览每页 24 条，发现最多 64 份元数据、单文件 4 MiB、累计 16 MiB、协作式 2 秒截止。
  session 索引 8 MiB，超限保留既有记录并报告限制；按需发现仍可返回新来源。
  来源页只做导航核验，内容完整性在选中后验证；选中 raw source 不重复枚举全部 Result。
  stdout 原有 64 MiB/5 秒内容预算独立计算；磁盘 IO 取消仍是协作式，并非强制中断系统调用。

## 分批记录

| 批次 | 提交 | 内容 |
| --- | --- | --- |
| I | dfb94de | 来源上下文、实际交付焦点、合并窗口歧义 |
| II | 5fabbd6 | 瞬态意图、实际 registry、自然报告 |
| III | 2e9a45c | 比较与依赖、Subject 历史几何 |
| IV | 74fbe27 | 有界展示、历史目录、选中读取预算与验证补充 |

真实验证还发现：自动授权路径只准备了第一个 Step，后续默认参数会改变已经接受的 Plan。
已调整为首次接受前准备全部相关 Step，新增“执行前截停”的离线回归；不放松执行指纹或旧确认校验。
真实模型曾输出长短不同的报告引文以及不存在的能量字段。现在接受唯一、互相嵌套的同一目标引文，
拒绝重复定位；提示词要求从公开契约复制字段名。保留既有一次结构化纠正上限，没有增加无条件重试。

## 验证结果

- 离线：618 通过、20 跳过、6 未选择；完整报告由 `pytest -m "not live_orca" --disable-socket` 生成。
- 真实 LLM：四种偶极矩表达 × Semantic/Intake 两个入口，共 8 项通过。
  每项均实际调用当前配置模型，并检查规范 Request/Plan 只含一个 Opt 计算任务。
  [模型调用、提案与计划证据](root-fix-live-llm.json) 包含纠正次数，不包含凭据。
  这是有限样本验证，不宣称模型在所有表述上绝对稳定。
- 真实 ORCA：1 项通过；Opt → 两个共用优化结构的 SP → 派生能差，4 个 Step 均成功。
  Run `run_cad2e4b9ca1347f5a8ad84feda1aefbe`，两个 SP 的几何 SHA-256 均为
  `2f3ef063c200ea080f92d148002aae21b51641f7b167150341bd395a52575ee7`。
  [实际 Run、输入绑定、资源和原文片段证据](root-fix-real-orca.json)。
  此链使用确定性测试提案和真实 ORCA，明确不充当“真实模型驱动端到端计算”的证据。
- 兼容：6113161 原 fixture 未改写；新增由未修改 b9eced9 代码生成的等待报告 fixture，
  固定 expected hash `1bb3c348b514086f2a80469c844a1b55c11388bc6cacf3a6e053b941b704bc71`。
  新代码只校验旧 hash，没有重新签署旧接受记录。
- Ruff、格式检查、compileall、wheel/sdist 构建：通过；用户预先存在的未跟踪草稿目录不纳入格式修改。
- GitHub CI：[offline 36584078341](https://github.com/az87988799/BG6022-v3/actions/runs/36584078341)，
  SHA `74fbe274ecfdce1b45ebc552a126e8df16c9ee7c`，success；619 通过、19 跳过、6 未选择。
  Ruff、格式、compileall 与 uv build 全部通过。CI 与本机的一个平台条件跳过差异已保留。

## 验收矩阵

以下“通过”为自动化验证结果，不是用户验收结论。除明确标注 live 的证据外，测试使用本地固定输入或测试专用 Tool，
不能充当真实化学证据。矩阵使用直接或等价的行为回归；不表示将全部 53 条表述逐条交给真实 LLM 重放。默认测试路径为 `tests/unit/`；完整逐测试结果见 [离线结果索引](root-fix-offline-tests.json)。

| 项目 | 验证输入/边界 | 证据 | 结果 |
| --- | --- | --- | --- |
| R01 | 水/PBE0 与乙醇/B3LYP 各有电子能 | test_semantic_proposal.py；test_parameter_scoping.py；test_semantic_agent_route.py | 自动化通过，等待用户验收 |
| R02 | 同一主体两种方法、两个同类 Step | test_semantic_proposal.py；test_parameter_scoping.py；test_semantic_agent_route.py | 自动化通过，等待用户验收 |
| R03 | 对目录连续压缩两次 | test_semantic_proposal.py；test_parameter_scoping.py；test_semantic_agent_route.py | 自动化通过，等待用户验收 |
| R04 | 旧失败 attempt 与之后改变参数的当前 Step | test_output_reports.py；test_chat_controls.py；test_context_query.py | 自动化通过，等待用户验收 |
| R05 | 正式结果完整、原文缺失 | test_output_reports.py；test_chat_controls.py；test_context_query.py | 自动化通过，等待用户验收 |
| R06 | 正式能量已交付，另一正式结果缺失 | test_output_reports.py；test_chat_controls.py；test_context_query.py | 自动化通过，等待用户验收 |
| R07 | 查询取消且没有真正交付任何新结果 | test_output_reports.py；test_chat_controls.py；test_context_query.py | 自动化通过，等待用户验收 |
| R08 | 不要计算 Gibbs 自由能，只算单点能 | test_root_contracts.py（排除/解释意图与正向任务分别验证） | 自动化通过，等待用户验收 |
| R09 | 解释为什么计算 Gibbs 需要频率 | test_root_contracts.py（排除/解释意图与正向任务分别验证） | 自动化通过，等待用户验收 |
| R10 | 不要全局构象搜索，只优化此结构 | test_root_contracts.py（排除/解释意图与正向任务分别验证） | 自动化通过，等待用户验收 |
| R11 | `Show Gibbs from the previously computed output.` | test_output_reports.py；test_orca_output_query.py | 自动化通过，等待用户验收 |
| R12 | 计算 Gibbs，另报告输出中的偶极矩 | test_output_reports.py；test_orca_output_query.py | 自动化通过，等待用户验收 |
| R13 | 给出一个测试用新性质的正式值，但模型只安排默认能量 | test_root_contracts.py（目标覆盖、未解析目标、一次结构化纠正） | 自动化通过，等待用户验收 |
| R14 | 引文不在原消息中、重复短引文无法定位、模型直接写保留 constraints 键 | test_output_reports.py；test_orca_output_query.py | 自动化通过，等待用户验收 |
| R15 | 只有原文报告，模型试图据此宣布科学成功或新增科学端口 | test_output_reports.py；test_orca_output_query.py | 自动化通过，等待用户验收 |
| R16 | 新临时协议缺少必要意图条目 | test_root_contracts.py（目标覆盖、未解析目标、一次结构化纠正） | 自动化通过，等待用户验收 |
| R16A | `优化水，然后给出它的偶极矩` | test_root_contracts.py + tests/live/test_root_fix_live.py（双入口共 8 个真实模型样本） | 自动化通过，等待用户验收 |
| R16B | `优化水，然后告诉我偶极矩` / `优化水并报告偶极矩` | test_root_contracts.py + tests/live/test_root_fix_live.py（双入口共 8 个真实模型样本） | 自动化通过，等待用户验收 |
| R16C | `优化水，然后计算它的偶极矩`，没有额外方法/Job/设置要求 | test_root_contracts.py + tests/live/test_root_fix_live.py（双入口共 8 个真实模型样本） | 自动化通过，等待用户验收 |
| R16D | 上述请求 `/confirm` 后 stdout 有/无偶极矩 | test_output_reports.py；test_attached_report_promotes_to_registered_verified_output；真实 ORCA 报告 | 自动化通过，等待用户验收 |
| R17 | 通过测试专用 registry 增加已实现 Tool 与新 property | test_root_contracts.py；test_generic_output_contracts.py | 自动化通过，等待用户验收 |
| R18 | 未注册 property 或错误类型的输出 | test_root_contracts.py；test_generic_output_contracts.py | 自动化通过，等待用户验收 |
| R19 | 同一 selector 同时可能指字段名与另一个 property | test_root_contracts.py；test_generic_output_contracts.py | 自动化通过，等待用户验收 |
| R20 | SP 的 `sp_electronic_energy` 与 Opt 的 `opt_final_electronic_energy` | test_root_contracts.py；test_generic_output_contracts.py | 自动化通过，等待用户验收 |
| R21 | 旧 AnswerGoal 使用真实字段名 | test_root_cause_contracts.py；test_execution_contract.py；旧确认 fixture | 自动化通过，等待用户验收 |
| R22 | Opt → 同一优化结构上的 A/B 双 SP → 方法能差 | test_root_contracts.py；test_energy_data.py；test_plan_composition.py；真实共享几何链 | 自动化通过，等待用户验收 |
| R23 | 差值两边几何或电子态不同 | test_root_contracts.py；test_energy_data.py；test_plan_composition.py；真实共享几何链 | 自动化通过，等待用户验收 |
| R24 | 同一依赖关系集合换排列顺序 | test_root_contracts.py；test_energy_data.py；test_plan_composition.py；真实共享几何链 | 自动化通过，等待用户验收 |
| R25 | 源只有普通值，没有可消费科学输出端口 | test_root_contracts.py；test_energy_data.py；test_plan_composition.py；真实共享几何链 | 自动化通过，等待用户验收 |
| R26 | 同类型多个候选端口 | test_root_contracts.py；test_energy_data.py；test_plan_composition.py；真实共享几何链 | 自动化通过，等待用户验收 |
| R27 | numeric_difference 只要求显示差值 | test_root_contracts.py；test_energy_data.py；test_plan_composition.py；真实共享几何链 | 自动化通过，等待用户验收 |
| R28 | 沿用这个结构计算单点能；历史上下文唯一 | test_root_contracts.py；test_context_query.py；test_deterministic_plan_builder.py | 自动化通过，等待用户验收 |
| R29 | 同样的请求换行或增加正常修饰语 | test_root_contracts.py；test_context_query.py；test_deterministic_plan_builder.py | 自动化通过，等待用户验收 |
| R30 | 水与乙醇各使用自己的历史优化结构 | test_root_contracts.py；test_context_query.py；test_deterministic_plan_builder.py | 自动化通过，等待用户验收 |
| R31 | 一部分 Subject 复用历史结构，另一部分需准备新结构 | test_root_contracts.py；test_context_query.py；test_deterministic_plan_builder.py | 自动化通过，等待用户验收 |
| R32 | 历史 Artifact 改变、跨会话、不再是选定有效来源 | test_root_contracts.py；test_context_query.py；test_deterministic_plan_builder.py | 自动化通过，等待用户验收 |
| R33 | 只有能量，回答草稿说“无虚频、已确认稳定极小值” | test_output_views.py；test_generic_output_contracts.py；test_chat_presentation.py | 自动化通过，等待用户验收 |
| R34 | 原子表有编号 1，草稿说总能量 1 Eh | test_output_views.py；test_generic_output_contracts.py；test_chat_presentation.py | 自动化通过，等待用户验收 |
| R35 | 用户要求少显示几位小数 | test_output_views.py；test_generic_output_contracts.py；test_chat_presentation.py | 自动化通过，等待用户验收 |
| R36 | 一万行测试 record_list，用户只要第 21–40 行 | test_output_views.py；test_generic_output_contracts.py；test_chat_presentation.py | 自动化通过，等待用户验收 |
| R37 | 用户明确要完整大表，但只交付了一页 | test_output_views.py；test_generic_output_contracts.py；test_chat_presentation.py | 自动化通过，等待用户验收 |
| R38 | 同一任务超过 24 个性质，目标在后面的属性页 | test_history_directory.py（30 字段、9/27/66 个 Run、读取预算、直接复核） | 自动化通过，等待用户验收 |
| R39 | 同会话超过 6 个 Run，明确选择第一个 | test_history_directory.py（30 字段、9/27/66 个 Run、读取预算、直接复核） | 自动化通过，等待用户验收 |
| R40 | 扫描历史目录时取消或超过元数据读取预算 | test_history_directory.py（30 字段、9/27/66 个 Run、读取预算、直接复核） | 自动化通过，等待用户验收 |
| R41 | 单个来源已选定，历史里存在大量其他 Result | test_history_directory.py（30 字段、9/27/66 个 Run、读取预算、直接复核） | 自动化通过，等待用户验收 |
| R42 | 一个原文段落同时命中标题、Magnitude 两个词 | test_orca_output_query.py；test_output_reports.py | 自动化通过，等待用户验收 |
| R43 | 两个独立候选段，或输出发生截断 | test_orca_output_query.py；test_output_reports.py | 自动化通过，等待用户验收 |
| R44 | 查询已保存来源，但本机 ORCA 当前不能运行 | test_orca_output_query.py；test_output_reports.py | 自动化通过，等待用户验收 |
| R45 | 请求不存在的 cursor、来源短引用、页范围或列名 | test_history_directory.py；test_output_views.py；test_orca_output_query.py；test_output_reports.py | 自动化通过，等待用户验收 |
| R46 | 查询前后比较来源文件清单和 SHA-256 | test_history_directory.py；test_output_views.py；test_orca_output_query.py；test_output_reports.py | 自动化通过，等待用户验收 |
| R47 | `6113161` 生成的旧确认 fixture 用新代码加载 | test_output_reports.py（6113161 / b9eced9 固定 fixture） | 自动化通过，等待用户验收 |
| R48 | `b9eced9` 生成的带 report_queries 的等待任务恢复 | test_output_reports.py（6113161 / b9eced9 固定 fixture） | 自动化通过，等待用户验收 |
| R49 | 新旧对话入口接受同一规范目标 | test_root_contracts.py；test_turn_routing.py；tests/live/test_root_fix_live.py（双入口真实调用） | 自动化通过，等待用户验收 |

## 已知边界与复现

- 未实现的正式性质仍必须拒绝或澄清，raw evidence 不改变科学检查结果。
- source_ref 与性质页 cursor 是当前 Agent 会话发出的临时引用；重启后重新浏览目录。
  历史身份由 session 元数据和磁盘 Run 归属恢复，不依赖把旧短引用当永久路径。
- 超大元数据明确报告需要受控读取；本轮没有增加数据库、向量检索或后台索引服务。
- 核心离线命令：`.venv/Scripts/python.exe -m pytest -m "not live_orca" --disable-socket`。
- 显式真实验证：`tests/live/test_root_fix_live.py`，分别使用 `--run-live-llm --enable-socket`
  与 `--run-live-orca --orca-config config.toml`。真实测试默认跳过，不会静默执行计算或访问网络。

下一状态仅可由用户验收决定；本任务保持“等待用户验收”。
