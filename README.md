# MSCAD 硕士论文研究项目

**轻量化多变量时序协同异常检测方法研究**

本仓库保存论文研究代码、冻结协议、实验审计与章节草稿。正式研究工作区为 `mscad-thesis`；本 GitHub 仓库于 2026-11-17 从其当前工作文件建立独立快照，包含此前尚未提交的论文材料。原始 Git 历史、工作区和实验资产保留在本地，来源见 [publication_manifest.json](docs/publication/publication_manifest.json)。

## 阅读入口

- [最终贡献结构](docs/thesis/final_contribution_structure.md)：局部滞后双视角协同检测、正常尾概率融合与轻量化框架。
- [论文结构](docs/thesis/final_thesis_outline.md) 与 [实验阶段冻结声明](docs/thesis/experimental_phase_freeze.md)。
- [第4章：双视角协同方法](docs/thesis/chapters/chapter4_dual_view.md)。
- [第5章：正常尾概率校准](docs/thesis/chapters/chapter5_tail_calibration.md)。
- 第6章：[数据与协议](docs/thesis/chapters/chapter6_experiments_part1.md)、[总体性能](docs/thesis/chapters/chapter6_experiments_part2.md)、[效率](docs/thesis/chapters/chapter6_experiments_part3.md)、[机制验证](docs/thesis/chapters/chapter6_experiments_part4.md)、[消融](docs/thesis/chapters/chapter6_experiments_part5.md)、[稳定性及文献对照](docs/thesis/chapters/chapter6_experiments_part6.md)。
- [整理后的冻结证据](docs/thesis/evidence/)：小型结果表、注册表、汇总与论文插图；逐文件原始位置和 SHA256 见发布清单。

历史文件名和早期审计文档保持原样。`docs/thesis/README.md`、`current_status.md` 等早期文档反映当时阶段；当前论文定位以最终贡献结构、实验冻结声明及章节审计为准。Legacy SWaT 重叠协议结果只作历史记录，不可作为最终 benchmark 证据。Oracle/PA 是补充诊断，不能替代 Strict 主表；2026 文献记录保留原指标语义与可比性限制。

## 目录

```text
models/ data/ losses/ utils/    模型、数据接口与公共实现
configs/                       冻结及历史配置
scripts/thesis/                 正式论文协议、方法、基线和审计脚本
tests/                         协议与实现测试
docs/thesis/chapters/           已完成的论文章节草稿
docs/thesis/evidence/           审阅后收录的小型冻结证据
docs/publication/              GitHub 快照来源、外部仓库与验证记录
```

根目录旧脚本及非 thesis 脚本为研究历史的一部分，不能仅根据文件名认为其属于最终方法。

## 环境与复现边界

Python 环境记录保存在 [environment.yml](environment.yml)、[requirements.txt](requirements.txt) 和 [环境审计](docs/thesis/environment_audit.md)。这些文件按原研究快照保留，本次上传没有安装依赖或更新版本，也没有重新验证完整训练环境。

```powershell
conda env create -f environment.yml
conda activate mscad-thesis
python -m pip install -r requirements.txt
```

执行前需要按环境审计确认 CUDA/PyTorch 与本机硬件兼容。部分历史脚本、基线和制图脚本需要额外依赖。第三方仓库不直接打包，来源与提交见 [external_repositories.json](docs/publication/external_repositories.json) 及各 baseline protocol 文档。

原始数据需要按许可自行获取。数据路径规范与审计保存在 `docs/thesis/`；支持该接口的脚本通过 `MSCAD_DATA_ROOT` 指向本地数据根目录。部分历史脚本依赖原始目录布局，应先检查对应协议与路径解析，不能直接把所有脚本视为一键可复现入口。

冻结 checkpoint、优化器状态、逐点 raw scores 和大体积输出未上传。依赖这些资产的测试或结果恢复步骤需要本地资产；本次只验证源文件语法、发布文件完整性与选定的数据无关测试，不宣称完整实验可在空白机器上立即重现。

证据文件保持原始字节，位于 `docs/thesis/evidence/`，其相对结构对应原 `outputs/`。如需在本地查看章节中的原始输出图片链接，可复制回 ignored 目录：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
Copy-Item docs/thesis/evidence/* outputs/ -Recurse
```

此操作只恢复已收录的小型文档证据，不生成模型结果。完整数据、checkpoint 和 raw scores 仍需单独保留。

## Git 与研究边界

数据集、PCAP、checkpoint、缓存、环境、密钥、完整 outputs 与外部克隆仓库均被排除。实验阶段已冻结；上传不改变模型、算法、评分、数据划分或科学结论。未新增许可证授权，第三方代码与数据遵循各自许可。
