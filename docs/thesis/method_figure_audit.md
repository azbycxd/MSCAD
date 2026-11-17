# Chapter 4 / Chapter 5 正式方法图审计

## 1. 前置门禁

| 检查项 | 结果 | 证据 |
|---|---|---|
| Chapter 4 correctness frozen | PASS | `docs/thesis/chapter4_final_correctness_review.md`: `THESIS_CHAPTER4_CORRECTNESS_FROZEN` |
| Chapter 5 correctness frozen | PASS | `docs/thesis/chapter5_final_correctness_review.md`: `THESIS_CHAPTER5_CORRECTNESS_FROZEN` |
| Experimental phase closed | PASS | `docs/thesis/experimental_phase_freeze.md`: `THESIS_EXPERIMENTAL_PHASE_CLOSED` |

本轮仅调整 Mermaid 的布局方向、中文标签、换行、低饱和配色、线宽、阶段边框和阶段编号。没有新增或删除方法节点，没有改变任何有向方法连接、公式、算法或实验结论。`F ~~~ I` 与 `CAL ~~~ TEST` 是 Mermaid 的不可见布局约束，不表示方法数据流。

## 2. 正式图题与文件

### 图 4-1

- 中文图题：图 4-1 局部滞后双视角协同异常检测流程
- English caption: Fig. 4-1 Workflow of the Local Lagged Dual-View Collaboration Consistency Detection Method
- Source: `docs/thesis/figures_src/dual_view_method_flow.mmd`
- Outputs: `fig_ch4_dual_view_method.svg`、`fig_ch4_dual_view_method.pdf`、`fig_ch4_dual_view_method.png`

### 图 5-2

- 中文图题：图 5-2 基于正常尾概率校准的异构双证据融合流程
- English caption: Fig. 5-2 Workflow of Normal-Tail Calibrated Heterogeneous Dual-Evidence Fusion
- Source: `docs/thesis/figures_src/tail_calibration_method_flow.mmd`
- Outputs: `fig_ch5_tail_calibration_method.svg`、`fig_ch5_tail_calibration_method.pdf`、`fig_ch5_tail_calibration_method.png`

Chapter 5 保留正文已经冻结的“图 5-2”编号，没有重编号。

## 3. Chapter 4 逻辑审计

| 检查项 | 结果 | 图中表达 |
|---|---|---|
| Self lags | PASS | target 历史，`L_self = 1, 5, 10, 20, 30` |
| Cross lags | PASS | `L_cross = 0, 1, 5, 10, 20, 30` |
| Target exclusion | PASS | Cross 输入明确为同一 process 的其他非恒定变量，并标注 `target excluded` |
| Normal-only Reliability Gate | PASS | Gate 完全位于阶段 1，仅由 Normal Train / Normal Validation 路径输入 |
| Disagreement semantics | PASS | Self 与 Cross 预测进入 `|ŷ_self - ŷ_cross|`；不是 observed-target residual |
| Frozen reference | PASS | Normal Validation 建立 median/MAD 正常参考，冻结后单向进入 Test 证据标准化 |
| Max aggregation | PASS | 逐 target 证据只在可靠 target 集合上取 `max`，输出 DUAL score |
| No Test calibration | PASS | 阶段 2 明确标注“不再拟合或校准”，没有 Test 到 Gate 或正常参考的回流 |

## 4. Chapter 5 逻辑审计

| 检查项 | 结果 | 图中表达 |
|---|---|---|
| Common endpoint alignment | PASS | REC/DUAL Validation 先按连续 segment 对齐到共同合法 endpoint |
| A/B split | PASS | 每段按时间顺序 50/50；前半 Calibration-A，后半 Threshold-B |
| ECDF upper tail | PASS | 两分支分别建立经验正常上尾映射，保留 `count(s >= x)` 与有限样本 `+1` 修正 |
| `-log10` evidence | PASS | Threshold-B 与 Test 均通过冻结映射得到 REC/DUAL 尾部证据 |
| Max fusion | PASS | 唯一融合为 `max(E_REC, E_DUAL)`；图中无权重、平均或 Robust-Z |
| Threshold-B q99 | PASS | q99 只由 Threshold-B 融合分数冻结 |
| Frozen Test inference | PASS | Test 仅使用冻结的 REC/DUAL mapping 与冻结 q99；无重新拟合或阈值选择 |
| Binary rule | PASS | 严格判定 `S_TAIL(t) > tau_TAIL` |

## 5. 视觉与格式审计

| 检查项 | 图 4-1 | 图 5-2 |
|---|---|---|
| 白色背景 | PASS | PASS |
| 字体与字号统一 | PASS | PASS |
| 节点边框与箭头统一 | PASS | PASS |
| Normal-only / Calibration 阶段使用实线分组框 | PASS | PASS |
| Frozen Test 阶段使用虚线分组框 | PASS | PASS |
| 核心 score / decision 使用加粗黑色边框 | PASS | PASS |
| 黑白打印可区分阶段 | PASS | PASS |
| 无渐变、3D 或装饰性阴影 | PASS | PASS |
| 文字截断、重叠或缺字 | PASS | PASS |
| 箭头与数据流可追踪 | PASS | PASS |

PNG 最终尺寸：图 4-1 为 `2906 x 2170`，图 5-2 为 `3332 x 2304`，满足论文插图的 300 dpi 等效清晰度。两份 PDF 均为单页、由 Chromium/Skia 生成的矢量 PDF；两份 SVG 均保留矢量节点、路径和文本，没有嵌入栅格图片。

## 6. 正文引用核验

| 正文 | 结果 | 当前引用 |
|---|---|---|
| `docs/thesis/chapters/chapter4_dual_view.md` | PASS | 方法总流程引用“图 4-1”，源文件为 `dual_view_method_flow.mmd` |
| `docs/thesis/chapters/chapter5_tail_calibration.md` | PASS | 校准/推理隔离流程引用“图 5-2”，源文件为 `tail_calibration_method_flow.mmd` |

图号、引用位置和源文件均一致，因此没有修改 Chapter 4/5 方法正文。

## 7. 最终状态

`THESIS_METHOD_FIGURES_READY`

本轮未训练、未推理、未运行实验，未修改方法、公式、算法或结论，未执行 Git add、commit 或 push。
