# Chapter 4 草稿审计

## 1. 前置状态

| CHECK | PASS/FAIL | evidence source | notes |
|---|---|---|---|
| 实验阶段已关闭 | PASS | `docs/thesis/experimental_phase_freeze.md` | `THESIS_EXPERIMENTAL_PHASE_CLOSED` |
| 贡献结构已冻结 | PASS | `docs/thesis/final_contribution_structure.md` | `THESIS_CONTRIBUTION_STRUCTURE_FROZEN` |
| 最终主张可辩护 | PASS | `docs/thesis/final_claim_audit.md` | `FINAL_CLAIMS_DEFENSIBLE` |
| 论文叙事结构已冻结 | PASS | `docs/thesis/final_thesis_outline.md` | `THESIS_NARRATIVE_STRUCTURE_READY`；采用冻结的 4.1–4.8 结构 |

## 2. 方法定义自审查

| CHECK | PASS/FAIL | evidence source | notes |
|---|---|---|---|
| A. 未把 DUAL 写成 observed-target residual | PASS | `run_hai21_final_unified.py:83-88`; `run_swat2015_final_unified.py:96-101` | 核心分数为 `abs(pred_self-pred_cross)`；observed target 只用于正常 skill 审计 |
| B. Self/Cross lag 正确 | PASS | `run_hai21_final_unified.py:27-29`; `run_swat2015_final_unified.py:23-25` | Self=`{1,5,10,20,30}`；Cross=`{0,1,5,10,20,30}` |
| C. Ridge alpha 正确 | PASS | 同上；`run_hai21_b1_real.py:113-115` | `alpha=1`，无搜索或改写 |
| D. Gate 为 Normal-only | PASS | `run_hai21_final_unified.py:155-214`; `run_swat2015_final_unified.py:134-194` | 只用 Normal discovery、Normal Validation full/first/second |
| E. 未写入 Test-driven selection | PASS | 两个 final manifest 的 `target_freeze_before_test` 与 `no_test_driven_selection` | Test 标签不参与训练、skill、稳定性门禁或 target 选择 |
| F. 未声称普适性 | PASS | `final_claim_audit.md` C1–C3、C7 | 明确 lag dataset dependence 和 relation-anomaly 适用范围 |
| G. 未隐藏 SWaT 负向边界 | PASS | `final_claim_audit.md` C7；Chapter 4.8 | 明确预告 SWaT official DUAL 结果有限，数值留第 6 章 |
| H. 未把轻量 REC 写成第 4 章创新 | PASS | `final_contribution_structure.md`; `final_thesis_outline.md:254-343` | 第 4 章只写 DUAL；REC 属于第 3 章轻量化设计贡献 |
| I. MAD 标定常数与下限正确 | PASS | `run_swat2015_final_unified.py:51-53`; `run_hai21_b1_real.py:133-136` | `scale=max(1.4826*MAD,1e-6)` |
| J. 对称稳定性比值未简化 | PASS | `run_hai21_final_unified.py:176-190`; `run_swat2015_final_unified.py:159-168` | 双向比例取 max，分母保护 `1e-12`；median≤2，q95≤3 |
| K. Cross target exclusion 明确 | PASS | `run_hai21_final_unified.py:191-206`; `run_swat2015_final_unified.py:169-176` | Cross source 为同过程其他非恒定变量，目标排除；lag 0 仅适用于其他变量 |
| L. 参数量语义正确 | PASS | `docs/thesis/efficiency_e1_final_method.md` | 使用 learned coefficients；SWaT 771、HAI 7,038 |

## 3. 输出完整性

| CHECK | PASS/FAIL | evidence source | notes |
|---|---|---|---|
| Chapter 4 正文 | PASS | `docs/thesis/chapters/chapter4_dual_view.md` | 含冻结 4.1–4.8、表 4-1/4-2、算法 4-1、公式（4-1）至（4-14） |
| 论文图源 | PASS | `docs/thesis/figures_src/dual_view_method_flow.mmd` | Self/Cross、分歧、校准、Gate、max 与 DUAL 流程齐全 |
| 公式注册表 | PASS | `docs/thesis/chapter4_formula_registry.csv` | 12 条，字段为 formula_id/section/symbol/definition/source/status |
| 引文处理 | PASS | Chapter 4.1、4.2 | 仅使用 `[CITATION:Ridge]`、`[CITATION:dependency-MTSAD]` 占位；未虚构文献编号 |
| 实验内容控制 | PASS | Chapter 4 全文 | 无实验表或新增结果；只指向第 6 章并保留适用边界 |

## 4. 冻结结论

未发现章节草稿与正式代码、manifest、贡献结构或主张边界之间的方法定义冲突。

Final status：`THESIS_CHAPTER4_DRAFT_READY`。

NO TRAINING；NO INFERENCE；NO NEW EXPERIMENT；NO METHOD CHANGE；NO NEW CLAIM；NO NEW LITERATURE SEARCH；NO COMMIT；NO PUSH。

