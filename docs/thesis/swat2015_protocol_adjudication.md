# SWaT 2015 label and benchmark protocol adjudication

裁决日期：2026-09-14

阶段：Thesis Phase T2.5

状态：`VALID_DIAGNOSTIC`

下一门禁：T3 protocol freeze

## 1. 与 T2 的关系

T2 的 `source_readiness=E. label_alignment_conflict` 和 `protocol_readiness=D. needs_official_semantics` 是发现冲突时的历史审计结论，完整保留在 `swat2015_source_audit.md`，本阶段不回写或改写该报告。

T2.5 的作用是对已发现的冲突作显式、可复核的 benchmark policy 裁决。它不声称原始材料从未冲突，也不产生修改后的官方 XLSX/PDF。裁决完成后，标签语义可进入 T3；正式 SWaT 2015 benchmark 仍未运行。

## 2. Primary point-wise ground truth

冻结 `SWaT_Dataset_Attack_v0.xlsx` 的 `Normal/Attack` 列作为主 point-wise ground truth。只接受以下 exact mapping：

| raw_label | canonical_label | canonical value |
|---|---|---:|
| `Normal` | `Normal` | 0 |
| `Attack` | `Attack` | 1 |
| `A ttack` | `Attack` | 1 |

实现必须同时保留 `raw_label` 和 `canonical_label`。匹配模式固定为 `exact_only_no_strip_no_casefold_no_fallback`；任何其他字符串，包括前后空格、大小写变化或额外空格，都必须报错，不允许自动 `strip`、猜测或映射为 0。

裁决理由：

- Physical 工作簿是官方发布的逐秒物理数据及其逐秒标签载体。
- 官方数据集论文说明，Physical 数据在汇总、移除冗余字段后，依据 attack log 的开始和结束时间人工标注。
- SWaT benchmark 的通常可比做法是直接使用发布的 Physical label 作为 point-wise ground truth。
- 保留 Physical label 可避免为追求当前模型结果而反向编辑标签，并维持与既有文献的 point-wise 定义可比性。

因此：

- `2015-12-29 06:30:00–06:42:00` 的额外 Physical `Attack` 在主 point-wise 评价中保持异常。
- ID 41 后额外 15 秒（至 `2016-01-02 13:41:11`）在主 point-wise 评价中保持 Physical 原值。
- 不对 point prediction 使用传统 point adjustment。

## 3. Event-level ground truth

Event metadata 的唯一来源是 `List_of_attacks_Final.xlsx`。全部 41 条 raw records 必须保留，原始字段同时写入 canonical metadata 的 `raw_*` 字段和 `raw_record_json`。

ID 5、9、12、15、18 固定标记为 `no_physical_impact_marker=true`：

- 保留 raw start；
- `event_id=null`；
- `canonical_end=null`；
- 不伪造 end time；
- 不进入 event hit/delay 的 36 个可评价区间。

其余 36 条记录各自形成一个独立 event，`event_id=attack_id`。即使 point-wise 标签连续，ID 10/11 和 ID 27/28 仍分别保留为两个事件编号，不合并。

### 3.1 IDs 37–41 chronology correction

IDs 37–41 同时保留：

- `raw_start/raw_end`：年份 2015；
- `canonical_start/canonical_end`：年份 2016；
- `chronology_correction=true`。

裁决只依据以下证据链：ID 36 已位于 2016-01-01；Attack Physical 覆盖 2016-01-02；原始 2015-01-02 落在该次 Physical 采集范围之外；将年份规范为 2016 后与 Physical 中相应攻击标签对齐。月、日和时分秒均不改变，官方 XLSX/PDF 不修改。

### 3.2 Event boundary、hit 与 delay

每个 event 使用 canonical attack-list start/end，按记录秒构成闭区间 `[start, end]`：

- `event_hit=true`：区间内至少出现一个合法报警时间戳；
- `detection_delay = first alarm timestamp inside event - event start`；
- 区间外报警不产生该 event 的 hit；
- point prediction 不因 event hit 而扩张或改写。

ID 10 的 end 与 ID 11 的 start 同为 2015-12-28 14:19:00；在闭区间定义下，该秒属于两个不同事件边界，但事件 ID 仍不合并。ID 27/28 相邻一秒，各自保持原清单边界。

### 3.3 ID 41

ID 41 event metadata 固定为：

- start：`2016-01-02 13:13:02`
- end：`2016-01-02 13:40:56`

Physical point label 延续至 `13:41:11` 的 15 秒只属于 point-wise raw ground truth，不延长 ID 41 event metadata。

### 3.4 Physical-only interval

`2015-12-29 06:30:00–06:42:00` 固定记录为：

- `physical_only_label_interval=true`；
- point-wise canonical label 为 Attack；
- 不创建虚假 event ID；
- 不加入 36 个 attack-list event；
- 可作为预先声明的 label-sensitivity 分析对象，但不得根据模型表现决定是否删除。

## 4. Gap 与 session segmentation

Attack Physical 的 `2015-12-31 21:00:36 -> 21:01:58` 82 秒间隔固定为 `hard_session_boundary`。禁止插值、forward fill、zero fill 或跨 gap 构造窗口。

Test 固定为至少两个连续 segment：

| Segment | Start | End inclusive | Rows |
|---|---|---|---:|
| `attack_segment_001` | 2015-12-28 10:00:00 | 2015-12-31 21:00:36 | 298,837 |
| `attack_segment_002` | 2015-12-31 21:01:58 | 2016-01-02 14:59:59 | 151,082 |

两段合计 449,919 行，与 Attack Physical 数据行数一致。T3 只能在单个 segment 内生成窗口；本阶段不选择窗口长度。

## 5. Primary normal source

Primary normal source 固定为完整 `SWaT_Dataset_Normal_v1.xlsx`，共 495,000 行，不再额外删除 5/6 小时 startup：

- v1 是官方 `Physical/readme.txt` 已明确处理的版本，已去掉最初 30 分钟维护过程；
- 保留完整 v1 有利于标准 benchmark 可比性；
- 额外 startup trimming 没有统一、可复核的官方时间边界。

未来可以预先注册 `startup-trim sensitivity`，但不得替代 primary result，也不得依据 Test 或模型结果选择 trimming 边界。

## 6. 输入字段

主模型候选输入固定为全部 51 个 physical variables。不得依据 Test 中是否 constant/near-constant，或 sensor/actuator 分组删除通道。`LIT401` 的官方 type 冲突仅作为 metadata 问题，不改变 51 维输入。

训练期零方差通道如何处理留给 T3，在本阶段不拟合 scaler、不观察训练统计、不删除变量。

## 7. Canonical metadata

以下文件由 `scripts/thesis/00_audit/adjudicate_swat2015_protocol.py` 生成，全部位于 Git-ignored 目录 `outputs/thesis_t25_swat2015_protocol/`：

- `canonical_attack_events.csv`：41 条 raw records、5 个 no-impact markers、36 个独立 evaluable events，以及 raw/canonical 时间。
- `point_label_policy.json`：三个 exact mappings、Physical-only interval、ID41 point/event 差异。
- `segment_manifest.json`：完整 Normal v1 和两个 Attack test segments，gap policy 为 hard boundary。
- `protocol_decision.json`：裁决状态、主数据源、事件 hit/delay 定义和禁止事项。

脚本在生成前后校验原始 `List_of_attacks_Final.xlsx` SHA-256，结果保持 `991b22dd48899769e69fcb0e19fc168784dcdde85b7f55feb888e524857e7709`。没有生成修改版 XLSX，也没有修改 T2 输出。

## 8. T3 门禁

T2.5 已冻结标签语义、event identity、年份 correction、ID41 差异、Physical-only interval、gap segmentation、Normal v1 和 51 维输入。下一步仅允许进入 T3 protocol freeze，决定尚未冻结的：

- Train/Validation 的精确时间边界；
- 窗口长度；
- 训练期零方差处理；
- threshold 规则；
- point-wise/event-level 指标实现及无泄漏断言。

正式训练、scaler fit、threshold 调优、模型结果读取、F1 优化和 V3 实现均未发生。
