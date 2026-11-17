# SWaT 2015 Official Physical source / schema / timeline / label audit

审计日期：2026-09-14

审计阶段：Thesis Phase T2

官方根目录：`D:\workspace\python\mscad-thesis\dataset\Swat_2015`

审计脚本：`scripts/thesis/00_audit/audit_swat2015_source.py`

## 1. 结论与门禁

- `source_readiness = E. label_alignment_conflict`
- `protocol_readiness = D. needs_official_semantics`
- 包身份结论：目录结构、论文、攻击清单及 Physical 工作簿构成一致的 SWaT 2015 Physical 发布包；格式可读，源身份本身已核实。当前副本没有原始压缩包或签名 manifest，因此这里验证的是内容身份与结构一致性，不声称完成下载链路的密码学溯源。
- 阻断原因：数据内标签与官方攻击清单并非完全一致，且官方 XLSX/PDF 自身把攻击 37–41 的日期写成 2015-01-02，落在 Attack Physical 时间范围之外。未经权威解释或显式协议裁决，不得静默改年、改拼写、补区间或重写标签。
- T3 门禁：**不满足** `source=A 且 protocol=A/B/C`，所以本审计不授权进入 T3。
- S15-01 仅可作为 `VALID_DIAGNOSTIC` 源审计；正式 benchmark 仍是 `TODO / not run`。

本轮没有训练、没有构造训练窗口、没有拟合 scaler、没有修改标签、没有生成清洗后的正式数据，也没有修改模型、损失或异常评分公式。

## 2. 官方源身份与清单

根目录实际存在 `Physical/`，不存在 `Network/`。因此 Network 只登记为“本地包中缺失”；没有搜索硬盘、没有解析 PCAP。Physical 内实际格式只有 3 个 XLSX、1 个 TXT 和 1 个 INI，没有 CSV、XLS、MAT 或 ARFF。根目录和 Physical 中共发现 8 个文件，SHA-256 均不重复；没有 `normal.csv`、`merged.csv` 或其他 Legacy CSV 混入官方根目录。

| Relative path | Size (bytes) | Format | SHA-256 | Audit treatment |
|---|---:|---|---|---|
| `A Dataset to Support Research in the Design of Secure Water Treatment Systems.pdf` | 3,441,528 | PDF, 13 pages | `2df05db808574d85358f4403b36eb251249b1642e580d7d44fb9e740dd8878e7` | 官方数据集论文，文本提取与页面核对 |
| `List_of_attacks_Final.pdf` | 89,320 | PDF, 1 page, A3 landscape | `3cc2d0e7bbe2dfe625dde5e32126ac101f5bacc51ab1e020a6270f31e3070a4b` | 与 XLSX 人工核对 |
| `List_of_attacks_Final.xlsx` | 15,258 | XLSX | `991b22dd48899769e69fcb0e19fc168784dcdde85b7f55feb888e524857e7709` | 只读提取 41 项清单 |
| `Physical/desktop.ini` | 244 | INI | `516bda6f12780debcdd2bd401ce9bd05fc93cad4e7970f2177a95a60e77af805` | 非数据文件 |
| `Physical/readme.txt` | 572 | TXT | `26c7eda4bc6d0c49b7383d08fecc1ea374fed99518917c3d25f9e5ad4ec1ee59` | Normal v0/v1 语义依据 |
| `Physical/SWaT_Dataset_Attack_v0.xlsx` | 116,016,616 | XLSX | `6cc72c24425453879bac1565cae4c6f35ad585a578510ab2a27b325d1bac2e8d` | Attack 时序源 |
| `Physical/SWaT_Dataset_Normal_v0.xlsx` | 130,993,435 | XLSX | `40e6de65900ab9d45866c587b6c85801dd612b50b45c688968d329c9a32d460f` | 含前 30 分钟维护阶段的 Normal 源 |
| `Physical/SWaT_Dataset_Normal_v1.xlsx` | 130,566,907 | XLSX | `fb7bb0a9c10f86ad29d8f2ae4fd09284e7db70c76350f5b2afcca58d9ace6c49` | 官方 README 所述去掉前 30 分钟的 Normal 版本 |

工作簿均用 `read_only=True, data_only=True` 打开且从未保存。每个源文件在扫描前后核对 size、mtime_ns 与 SHA-256，均未变化。openpyxl 对一个未知工作簿扩展发出“读取时不支持”的警告；因为未保存源文件，该扩展没有被删除或重写。

## 3. Physical 文件与 Normal / Attack 语义

| File | Worksheet | Data rows | Columns | First timestamp | Last timestamp | Label counts |
|---|---|---:|---:|---|---|---|
| `SWaT_Dataset_Normal_v0.xlsx` | `Normal.csv` | 496,800 | 53 | 2015-12-22 16:00:00 | 2015-12-28 09:59:59 | `Normal=496800` |
| `SWaT_Dataset_Normal_v1.xlsx` | `Normal.csv` | 495,000 | 53 | 2015-12-22 16:30:00 | 2015-12-28 09:59:59 | `Normal=495000` |
| `SWaT_Dataset_Attack_v0.xlsx` | `Combined Data` | 449,919 | 53 | 2015-12-28 10:00:00 | 2016-01-02 14:59:59 | `Normal=395298`, `Attack=54584`, `A ttack=37` |

官方论文称连续采集 11 天，前 7 天不施加攻击、后 4 天实施攻击，记录频率为每秒一次，共 51 个物理属性。实际工作簿的时间覆盖约为 5 天 18 小时 Normal + 5 天 5 小时 Attack，合计约 10 天 23 小时；论文的“7/4 天”是阶段描述，不能直接替代文件内精确时间界线。论文另写有 946,722 个样本，而 Normal v0 + Attack v0 的实际数据行数为 946,719，相差 3 行；审计保留该差异，不自行补行。

`Physical/readme.txt` 明确说明：

- Normal v0 从工厂排空储水罐的 30 分钟维护过程开始；该过程不属于通常运行条件。
- Normal v1 由 v0 移除最初 30 分钟得到。

文件证据与说明一致：v1 比 v0 少 1,800 行，起点晚 1,800 秒，终点相同；对应的前 100 行抽样逐值相等。两份工作簿哈希不同，不是字节重复副本。

但官方论文另称系统从空罐启动，P1 约 5 小时、P3/P4 约 6 小时后才稳定。v1 只去掉 30 分钟，并不等价于“已去除全部 startup transient”。T3 前必须决定是否、以及依据什么规则排除其余启动段，T2 不自行裁剪。

## 4. 时间线审计

| File | Timestamp parse errors | Duplicate timestamps | Non-monotonic pairs | Interval distribution | Gaps > 1 s |
|---|---:|---:|---:|---|---:|
| Normal v0 | 0 | 0 | 0 | `1 s: 496799` | 0 |
| Normal v1 | 0 | 0 | 0 | `1 s: 494999` | 0 |
| Attack v0 | 0 | 0 | 0 | `1 s: 449917`, `82 s: 1` | 1 |

Attack v0 唯一大间隔是 `2015-12-31 21:00:36 -> 21:01:58`。按 1 Hz 期望，这相当于缺少 81 个中间时间戳。时间仍严格递增，不能把该缺口自动插值或当作连续窗口跨越；T3 必须把它冻结为会话边界或制定有证据的缺失处理规则。

所有时间戳均不含时区或 UTC offset；官方材料只支持“本地时钟、时区未说明”。不得在协议中臆造时区。

机器可读明细：`outputs/thesis_t2_swat2015_audit/timeline_audit.csv`（Git ignored）。

## 5. Schema 与 51 变量

三个工作簿的规范化列名一致：`Timestamp + 51 physical variables + Normal/Attack`，没有额外索引列。所有 51 个变量均为数值型；完整扫描没有 NaN、Inf、空值或 malformed numeric string。Attack v0 按观测值分为 25 个连续变量、19 个离散变量和 7 个常量变量。

“near-constant”在本审计中严格定义为：非常量且相邻有效观测的精确值变化率 `<0.001`。该标签只是源诊断，不授权删除通道。

官方论文 Table 2 的字面 Type 列给出 24 sensors / 27 actuators，但同一行把 `LIT401` 写作 `Actuator`，描述却是 `Level Transmitter; RO feed water tank level`。按命名和描述的语义分类则为 25 sensors / 26 actuators。审计同时保留两组计数，不静默修正官方表格；模型输入仍是全部 51 变量。

| # | Variable | Stage | Paper literal type | Semantic type | Observed kind (Attack) | Constant | Near-constant |
|---:|---|---|---|---|---|---|---|
| 1 | `FIT101` | P1 | sensor | sensor | continuous | no | no |
| 2 | `LIT101` | P1 | sensor | sensor | continuous | no | no |
| 3 | `MV101` | P1 | actuator | actuator | discrete | no | no |
| 4 | `P101` | P1 | actuator | actuator | discrete | no | yes |
| 5 | `P102` | P1 | actuator | actuator | discrete | no | yes |
| 6 | `AIT201` | P2 | sensor | sensor | continuous | no | no |
| 7 | `AIT202` | P2 | sensor | sensor | continuous | no | no |
| 8 | `AIT203` | P2 | sensor | sensor | continuous | no | no |
| 9 | `FIT201` | P2 | sensor | sensor | continuous | no | no |
| 10 | `MV201` | P2 | actuator | actuator | discrete | no | no |
| 11 | `P201` | P2 | actuator | actuator | discrete | no | yes |
| 12 | `P202` | P2 | actuator | actuator | constant | yes | no |
| 13 | `P203` | P2 | actuator | actuator | discrete | no | yes |
| 14 | `P204` | P2 | actuator | actuator | discrete | no | yes |
| 15 | `P205` | P2 | actuator | actuator | discrete | no | yes |
| 16 | `P206` | P2 | actuator | actuator | discrete | no | yes |
| 17 | `DPIT301` | P3 | sensor | sensor | continuous | no | no |
| 18 | `FIT301` | P3 | sensor | sensor | continuous | no | no |
| 19 | `LIT301` | P3 | sensor | sensor | continuous | no | no |
| 20 | `MV301` | P3 | actuator | actuator | discrete | no | no |
| 21 | `MV302` | P3 | actuator | actuator | discrete | no | no |
| 22 | `MV303` | P3 | actuator | actuator | discrete | no | no |
| 23 | `MV304` | P3 | actuator | actuator | discrete | no | no |
| 24 | `P301` | P3 | actuator | actuator | constant | yes | no |
| 25 | `P302` | P3 | actuator | actuator | discrete | no | yes |
| 26 | `AIT401` | P4 | sensor | sensor | continuous | no | no |
| 27 | `AIT402` | P4 | sensor | sensor | continuous | no | no |
| 28 | `FIT401` | P4 | sensor | sensor | continuous | no | no |
| 29 | `LIT401` | P4 | actuator | sensor | continuous | no | no |
| 30 | `P401` | P4 | actuator | actuator | constant | yes | no |
| 31 | `P402` | P4 | actuator | actuator | discrete | no | yes |
| 32 | `P403` | P4 | actuator | actuator | discrete | no | yes |
| 33 | `P404` | P4 | actuator | actuator | constant | yes | no |
| 34 | `UV401` | P4 | actuator | actuator | discrete | no | yes |
| 35 | `AIT501` | P5 | sensor | sensor | continuous | no | no |
| 36 | `AIT502` | P5 | sensor | sensor | continuous | no | no |
| 37 | `AIT503` | P5 | sensor | sensor | continuous | no | no |
| 38 | `AIT504` | P5 | sensor | sensor | continuous | no | no |
| 39 | `FIT501` | P5 | sensor | sensor | continuous | no | no |
| 40 | `FIT502` | P5 | sensor | sensor | continuous | no | no |
| 41 | `FIT503` | P5 | sensor | sensor | continuous | no | no |
| 42 | `FIT504` | P5 | sensor | sensor | continuous | no | no |
| 43 | `P501` | P5 | actuator | actuator | discrete | no | yes |
| 44 | `P502` | P5 | actuator | actuator | constant | yes | no |
| 45 | `PIT501` | P5 | sensor | sensor | continuous | no | no |
| 46 | `PIT502` | P5 | sensor | sensor | continuous | no | no |
| 47 | `PIT503` | P5 | sensor | sensor | continuous | no | no |
| 48 | `FIT601` | P6 | sensor | sensor | continuous | no | no |
| 49 | `P601` | P6 | actuator | actuator | constant | yes | no |
| 50 | `P602` | P6 | actuator | actuator | discrete | no | yes |
| 51 | `P603` | P6 | actuator | actuator | constant | yes | no |

机器可读全量统计、原始/规范化列名、每通道 min/max/mean/std/change rate 和异常样例字段位于 `outputs/thesis_t2_swat2015_audit/schema_profile.json`（Git ignored）。

## 6. 攻击清单与数据内标签核对

### 6.1 XLSX / PDF

`List_of_attacks_Final.xlsx` 有 41 个编号项。ID 5、9、12、15、18 明确标成 `No Physical Impact Attack` 且没有 end time，因此可形成 36 个物理攻击区间；这与论文“A total of 36 attacks were launched”一致。XLSX 和 PDF 的表头、41 个编号、时间、目标及这 5 个 marker 人工核对一致。PDF 没有提供可消除下列冲突的另一套时间。

36 个区间按 inclusive-second 仅用于描述时，持续时间：最短 101 秒（ID 34），中位数 443.5 秒，均值 1,496.83 秒，最长 34,209 秒（ID 28，约 9.5 小时）。本阶段不据此选择 64/128 或任何窗口长度。

原始清单按表格顺序有一个时间倒退：ID 36 后的 ID 37 被写为 `2015-01-02`。ID 37–41 在 XLSX 和 PDF 中都写为 2015-01-02，而 Attack Physical 对应时间只能落在 2016-01-02。审计输出另列“将年份改为 2016 的诊断性比较”，但原始字段保持 2015，且没有生成修正版标签。

清单不存在正常意义上的重叠区间；ID 10 的 end 与 ID 11 的 start 同为 2015-12-28 14:19:00。ID 27 的 end 与 ID 28 的 start 相邻 1 秒，二者之间没有可标成 Normal 的采样秒。最终 left-closed/right-open 或 inclusive 边界语义未冻结。

### 6.2 Physical 标签

数据内标签的 exact spelling 和行数是：

- `Normal`: 395,298
- `Attack`: 54,584
- `A ttack`: 37

若不改写任何原值，精确字符串 `Attack` 形成 35 个连续区间。按时间覆盖而不是区间序号核对 36 个清单攻击：

- 26 个原始日期和边界精确一致；
- ID 10/11、27/28 共 4 个清单区间被 Physical 的连续 `Attack` 标签合并覆盖；
- ID 24 的开始 37 秒写成 `A ttack`，之后才变为 `Attack`，属于实质 spelling mismatch；
- ID 37–41 的官方年份在 XLSX/PDF 中与 Physical 冲突；仅在明确标记的 2016 年诊断比较下，ID 37–40 边界精确一致；
- ID 41 在该诊断比较下开始一致，但 Physical `Attack` 延后 15 秒结束（13:41:11 vs 清单 13:40:56）；
- Physical 另有一个 `Attack` 区间 `2015-12-29 06:30:00–06:42:00`，没有清单区间与之重叠。

所以 Physical label 与 attack list **不完全一致**。这不是单纯统一开闭区间就能解释的 boundary-only mismatch；T2 不重写任何标签。机器可读逐攻击报告位于 `outputs/thesis_t2_swat2015_audit/label_alignment_report.csv`（Git ignored）。

## 7. Legacy 污染与隔离

官方根目录中没有 Legacy 文件；新脚本必须显式给出 official root，并拒绝名为 `Swat` 的模糊根目录以及包含 `normal.csv`/`merged.csv` 的候选根目录。不存在自动 fallback。

旧文件仍位于 `D:\workspace\python\mscad\Swat\`，未被修改：

| Legacy file | Size | SHA-256 | Comparison with official Physical |
|---|---:|---|---|
| `normal.csv` | 402,374,633 | `efa3dfd271fd33402d04ff4323d92ba739758bb2d4b9c6e9cc205980891f111f` | 哈希不同；53 列规范化表头一致；首行与 official Attack v0 逐字段匹配率 1.0 |
| `merged.csv` | 426,855,390 | `5472928b8faf51430f4490628c6fe99487d58d5826a41460ab26aee69dde22a4` | 哈希不同；53 列规范化表头一致；首行与 official Attack v0 逐字段匹配率 1.0 |

这说明 Legacy CSV 很可能是由同一 Physical 源加工出的历史输入，并非内容上完全无关；但它们不是未来协议允许读取的源路径。既有 4.3D 审计已证明 `merged.csv` 含完整 normal 前缀重叠和 attack-only 后缀，不能因为底层来源相关就恢复其 benchmark 有效性。

## 8. 候选正式协议（仅 proposed，尚未冻结）

在解决上节 label semantics、startup 和 gap policy 后，T3 可考虑：

- Train source：只从 Normal v1 的正常时序构造，是否再排除论文所述 5–6 小时 startup transient 必须先冻结。
- Validation：在 Normal v1 内按时间切出严格不重叠的后段；具体比例/边界留给 T3，不从 Attack test 抽样。
- Test source：完整保持 `SWaT_Dataset_Attack_v0.xlsx` 的官方时间顺序和文件边界；不拼接 Legacy attack-only suffix。
- Context/window：不得跨文件，且不得跨越 Attack v0 的 82 秒 gap；窗口长度不在 T2 决定。
- Label policy：保留原始 exact spelling 与 attack list；必须先对 `A ttack`、清单外区间、ID 37–41 年份和 ID 41 结束秒形成可引用的裁决记录。
- Scaler：只 fit Train；Validation/Test 仅 transform。
- Model selection：不得使用 Test label。
- Threshold：只由正常 Validation 的预先冻结规则确定，不用 Test label 调参。
- Primary metric：不使用传统 point adjustment；事件边界规则须在看测试结果前冻结。

该协议目前只是候选框架，不是已生成的数据划分，也没有任何训练或性能结果。

## 9. 未决问题

1. 是否能取得官方说明，确认 ID 37–41 应为 2016-01-02。
2. `2015-12-29 06:30:00–06:42:00` 的 Physical `Attack` 区间对应哪一条官方攻击定义。
3. ID 24 的 `A ttack` 是否可在协议层映射为攻击，以及如何保持原始值和映射审计轨迹。
4. ID 41 的 15 秒结束差异采用哪个权威边界。
5. ID 10/11 和 27/28 的相邻攻击在事件级评估中算一个连续事件还是两个清单事件。
6. Normal v1 中仍处于 startup transient 的时间如何处理。
7. Attack v0 的 82 秒 gap 是否直接作为 session boundary；禁止无证据插值。
8. 论文 Table 2 中 `LIT401` 的 type 文字冲突只影响描述性分组还是需要正式勘误说明。

在这些问题得到有证据的 protocol decision 之前，S15-01 不得用于正式性能 benchmark。

## 10. 可复核命令与输出

```powershell
C:\Users\jiusi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  scripts\thesis\00_audit\audit_swat2015_source.py `
  --official-root D:\workspace\python\mscad-thesis\dataset\Swat_2015 `
  --output-dir D:\workspace\python\mscad-thesis\outputs\thesis_t2_swat2015_audit `
  --legacy-candidate D:\workspace\python\mscad\Swat\normal.csv `
  --legacy-candidate D:\workspace\python\mscad\Swat\merged.csv
```

输出目录整体由 `.gitignore` 的 `outputs/` 规则排除。raw dataset、生成 CSV/JSON 和攻击标签不会进入 Git。

本次扫描使用 CPython 3.12.14 与 openpyxl 3.1.5 的只读模式；没有安装、升级或修改任何 Conda/Python 环境。审计测试使用既有 p310 的 pytest，并仅在测试进程内临时加入上述 openpyxl 路径。
