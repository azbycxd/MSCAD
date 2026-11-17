from __future__ import annotations

from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


OUT = Path("docs") / "研究内容_研究方案_研究进展_综合整理_论文排版精简版.docx"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def e(text: str) -> str:
    return escape(text, {'"': "&quot;"})


def rpr(size: int = 24, bold: bool = False, font: str = "宋体") -> str:
    return (
        "<w:rPr>"
        + (f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="{font}"/>' )
        + ("<w:b/>" if bold else "")
        + f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        + "</w:rPr>"
    )


def ppr(
    style: str | None = None,
    align: str | None = None,
    first_line: int | None = None,
    before: int = 0,
    after: int = 120,
    line: int = 360,
) -> str:
    parts = ["<w:pPr>"]
    if style:
        parts.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        parts.append(f'<w:jc w:val="{align}"/>')
    parts.append(f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>')
    if first_line:
        parts.append(f'<w:ind w:firstLine="{first_line}"/>')
    parts.append("</w:pPr>")
    return "".join(parts)


def paragraph(text: str = "", style: str | None = "Body", align: str | None = None) -> str:
    first = 480 if style == "Body" else None
    size = {"Title": 36, "Subtitle": 24, "Heading1": 32, "Heading2": 28, "Body": 24, "Caption": 21, "Code": 21}.get(style or "Body", 24)
    bold = style in {"Title", "Heading1", "Heading2"}
    font = "黑体" if style in {"Title", "Heading1", "Heading2"} else "宋体"
    if style == "Code":
        first = None
        font = "Courier New"
    return f"<w:p>{ppr(style, align, first)}<w:r>{rpr(size, bold, font)}<w:t>{e(text)}</w:t></w:r></w:p>"


def list_item(text: str) -> str:
    return (
        f"<w:p>{ppr('Body', None, None)}"
        f"<w:r>{rpr(24)}<w:t>{e(text)}</w:t></w:r></w:p>"
    )


def table(rows: list[list[str]], widths: list[int] | None = None) -> str:
    cols = len(rows[0])
    widths = widths or [int(9000 / cols)] * cols
    out = [
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="666666"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="666666"/></w:tblBorders>'
        '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr>'
    ]
    for ri, row in enumerate(rows):
        out.append("<w:tr>")
        for ci, cell in enumerate(row):
            shade = '<w:shd w:fill="D9EAF7"/>' if ri == 0 else ""
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shade}</w:tcPr>'
                f'<w:p>{ppr(None, "center" if ri == 0 else None, None, after=60, line=300)}'
                f'<w:r>{rpr(21, ri == 0, "宋体")}<w:t>{e(cell)}</w:t></w:r></w:p></w:tc>'
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS["w"]}">
  <w:docDefaults><w:rPrDefault>{rpr(24)}</w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Body"><w:name w:val="正文"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="题名"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="副标题"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="一级标题"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="二级标题"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="表题"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="公式"/><w:qFormat/></w:style>
</w:styles>'''


def document_xml(body: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS["w"]}" xmlns:r="{NS["r"]}">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" w:header="720" w:footer="720" w:gutter="0"/>
      <w:cols w:space="425"/>
      <w:docGrid w:type="lines" w:linePitch="312"/>
    </w:sectPr>
  </w:body>
</w:document>'''


def content_types() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''


def rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''


def doc_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''


def build_body() -> str:
    parts: list[str] = []
    parts.append(paragraph("基于多尺度结构感知的轻量级多变量时间序列异常检测研究", "Title", "center"))
    parts.append(paragraph("研究内容、研究方案与阶段进展综合整理", "Subtitle", "center"))
    parts.append(paragraph(f"整理日期：{date.today().isoformat()}    课题简称：MSCAD", "Caption", "center"))

    parts.append(paragraph("摘要", "Heading1"))
    parts.append(paragraph(
        "本文面向工业控制、航天器遥测和发动机退化监测等多变量时间序列异常检测场景，研究仅使用正常样本训练条件下的异常识别问题。针对传统重构式方法较难刻画通道协同关系变化的问题，提出 MSCAD（Multi-Scale Structure-aware Correlation Anomaly Detection）方法：在轻量级一维卷积自编码器中引入短期和长期通道结构约束，并在测试阶段融合数值重构误差与结构偏差分数。现有实验表明，MSCAD 在 SWaT 数据集上以较低参数量和较快推理速度取得具有竞争力的检测效果；结构微调和分数融合能够提升 AUPR/AUROC，说明通道结构信息有助于改善异常排序能力。"
    ))
    parts.append(paragraph("关键词：多变量时间序列；异常检测；结构感知；自编码器；SWaT", "Body"))

    parts.append(paragraph("1 研究背景与问题定义", "Heading1"))
    parts.append(paragraph(
        "多变量传感器数据通常具有通道数量多、变量耦合强、异常标注稀缺和实时部署要求高等特点。异常不仅可能表现为单一通道数值突变，也可能表现为多个传感器之间原有协同关系被破坏。若只依赖重构误差，模型对关系型异常的敏感性可能不足。"
    ))
    parts.append(paragraph("设输入窗口为：", "Body"))
    parts.append(paragraph("X in R^{T x C}", "Code"))
    parts.append(paragraph(
        "其中 T 表示窗口长度，C 表示通道数。模型仅用正常样本学习正常工况下的数值模式和通道结构模式，测试时对每个窗口输出异常分数："
    ))
    parts.append(paragraph("Score(X) = f(reconstruction_error, short_structure_error, long_structure_error)", "Code"))
    parts.append(paragraph("本研究重点解决三个问题：如何在轻量级模型中加入通道结构建模能力，如何同时利用短期与长期结构变化，以及如何在检测性能、模型开销和解释能力之间取得平衡。"))

    parts.append(paragraph("2 研究目标与创新点", "Heading1"))
    for item in [
        "（1）设计轻量级多变量时间序列异常检测方法，在重构误差之外引入多尺度结构异常分数。",
        "（2）在 SWaT 等公开数据集上验证方法有效性，并与 AE、LSTM-AE、PCA、Isolation Forest、MTAD、CATCH 等方法对比。",
        "（3）通过消融实验分析短期结构项、长期结构项、结构度量、阈值策略和分数融合策略的影响。",
        "（4）形成稳妥论文主线：强调轻量化成本下的竞争性结果、结构异常分数带来的排序能力提升和一定解释潜力。"
    ]:
        parts.append(list_item(item))
    parts.append(paragraph("主要创新点包括：从“重构数值”扩展到“重构关系”；引入短期与长期两类结构尺度；通过相关通道对筛选降低结构噪声；保持约 11.7K 参数的轻量级模型规模，便于在线监控和资源受限场景部署。"))

    parts.append(paragraph("3 方法设计", "Heading1"))
    parts.append(paragraph("3.1 模型主干", "Heading2"))
    parts.append(paragraph("MSCAD 采用一维卷积自编码器作为主干。输入窗口形状为 [batch, window_size, channels]，在模型内部转置为卷积友好的格式后，由编码器提取隐空间表示，再由解码器重构原始多变量时间序列窗口。"))
    parts.append(paragraph("表1 主要模型参数", "Caption", "center"))
    parts.append(table([
        ["参数", "取值"],
        ["输入通道数 in_channels", "51"],
        ["窗口长度 window_size", "64"],
        ["短窗口长度 short_window", "16"],
        ["隐藏通道数 hidden_channels", "64"],
        ["潜变量通道数 latent_channels", "32"],
        ["dropout", "0.1"],
        ["参数量", "11,730"],
    ], [5000, 3000]))
    parts.append(paragraph("3.2 训练损失与异常分数", "Heading2"))
    parts.append(paragraph("训练损失由重构损失、短期结构损失和长期结构损失组成："))
    parts.append(paragraph("Loss = lambda_rec * L_rec + lambda_s * L_short + lambda_l * L_long", "Code"))
    parts.append(paragraph("其中，L_rec 约束重构序列与原始序列在数值上接近；L_short 在窗口末尾短片段上计算通道关系矩阵差异，用于捕捉突发关系变化；L_long 在完整窗口上计算结构差异，用于捕捉较稳定的协同模式偏移。测试阶段分别生成 rec、short、long 三类异常分数，并在训练分数统计基础上标准化融合："))
    parts.append(paragraph("Score = alpha * z(rec) + beta * z(short) + gamma * z(long)", "Code"))
    parts.append(paragraph("3.3 阈值与结构度量", "Heading2"))
    parts.append(paragraph("当前主要采用 POT 阈值和 mean + 3std 阈值。结构关系度量支持 corr、cov 和 shrink_cov，主结果推荐使用 corr。已有实验显示，结构分数有价值，但不宜简单增大结构权重；更稳定的策略是弱结构融合或两阶段结构微调。"))

    parts.append(paragraph("4 实验设计", "Heading1"))
    parts.append(paragraph("核心数据集为 SWaT，包含工业水处理系统正常运行数据和攻击数据，输入通道数为 51，窗口长度为 64，短窗口长度为 16。SMAP 和 C-MAPSS/N-CMAPSS 目前作为补充实验，用于分析方法适用边界。"))
    parts.append(paragraph("对比方法包括 AE、LSTM-AE、PCA、Isolation Forest、MTAD 和 CATCH。评价指标采用 Precision、Recall、F1、AUPR、AUROC、参数量、推理时间和显存占用。其中 F1 依赖阈值，AUPR/AUROC 更能反映异常分数的排序质量。"))

    parts.append(paragraph("5 实验结果与分析", "Heading1"))
    parts.append(paragraph("表2 SWaT 主实验结果", "Caption", "center"))
    parts.append(table([
        ["方法", "F1", "AUPR", "AUROC", "参数量", "推理 ms/batch"],
        ["MSCAD main", "0.7604", "0.6698", "0.8786", "11,730", "0.8217"],
        ["AE", "0.7936", "0.8122", "0.9529", "11,730", "0.4520"],
        ["LSTM-AE", "0.8513", "0.8396", "0.9633", "71,603", "0.5754"],
        ["PCA", "0.8076", "0.7737", "0.9508", "-", "8.0871"],
        ["Isolation Forest", "0.6886", "0.5455", "0.9014", "-", "13.0003"],
        ["CATCH-FULL+FUSION", "0.7599", "0.6991", "0.9243", "63,761", "3.7210"],
        ["MTAD-FULL+FUSION", "0.8028", "0.8021", "0.9561", "83,801", "0.4619"],
    ], [2700, 1200, 1200, 1200, 1500, 1700]))
    parts.append(paragraph("主实验说明，MSCAD main 与 CATCH 类方法在 F1 上接近，但参数量更小、推理更快；LSTM-AE 和 MTAD 在部分指标上更强，因此论文不宜宣称方法全面领先。更合理的论证重点是：MSCAD 在轻量化条件下取得竞争性结果，并提供结构感知与解释潜力。"))
    parts.append(paragraph("结构微调与融合搜索显示，较高重构权重和较弱结构权重更稳定。例如 R5 配置 F1 为 0.7997、AUPR 为 0.8270、AUROC 为 0.9591；R6 配置 AUPR 提升至 0.8385、AUROC 提升至 0.9611。这表明结构信息对排序指标有帮助，但其收益受训练策略、融合权重和阈值选择影响。"))

    parts.append(paragraph("6 阶段进展与不足", "Heading1"))
    parts.append(paragraph("表3 当前研究进展", "Caption", "center"))
    parts.append(table([
        ["模块", "完成情况", "完成度"],
        ["研究问题定义", "已明确数值异常与关系异常并存的问题", "85%"],
        ["MSCAD 方法设计", "已完成主干、结构损失、分数融合和阈值流程", "90%"],
        ["代码实现", "已完成训练、评估、baseline、消融和跨数据集脚本", "90%"],
        ["SWaT 主实验", "已完成主方法、变体和主要 baseline 对比", "85%"],
        ["消融实验", "已完成结构项、结构度量和阈值方式消融", "75%"],
        ["跨数据集实验", "已有 SMAP、C-MAPSS/N-CMAPSS 尝试", "45%"],
        ["可解释性分析", "已有思路，缺少正式案例图和通道对表", "35%"],
        ["正式论文正文", "尚未正式成稿", "0%"],
    ], [2800, 5200, 1200]))
    parts.append(paragraph("综合来看，当前研究完成度约为 70%。实验主干已经完成，论文级论证还需要补强多随机种子稳定性、融合权重敏感性和可解释性案例。跨数据集结果目前不稳定，应作为适用边界分析，避免削弱主线。"))

    parts.append(paragraph("7 后续工作", "Heading1"))
    for item in [
        "（1）补充 3 到 5 个随机种子实验，优先覆盖 rec-only 和 struct-finetune，报告均值与标准差。",
        "（2）固定 checkpoint 后系统搜索 alpha/beta/gamma，绘制 F1、AUPR、AUROC 随结构权重变化的曲线。",
        "（3）选择 1 到 3 个 SWaT 异常片段，展示 rec、short、long 三类分数随时间变化，并列出贡献最大的通道对。",
        "（4）优先撰写绪论、相关工作、方法和 SWaT 主实验章节，跨数据集与可解释性内容在补实验后完善。"
    ]:
        parts.append(list_item(item))

    parts.append(paragraph("8 结论", "Heading1"))
    parts.append(paragraph("本文提出的 MSCAD 方法以轻量级卷积自编码器为基础，引入短期和长期通道结构约束，将异常检测从单纯数值重构扩展到数值模式与通道关系的联合建模。已有结果表明，结构分数能够改善 AUPR/AUROC 等排序指标，模型也具有较明显的轻量化优势。与此同时，实验也说明结构约束并非越强越好，F1 对阈值和融合权重较敏感。后续论文写作应采用稳妥论证策略：突出轻量化、结构感知和可解释潜力，并将部分不理想结果转化为训练策略和适用边界分析。"))
    return "\n".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types())
        z.writestr("_rels/.rels", rels())
        z.writestr("word/_rels/document.xml.rels", doc_rels())
        z.writestr("word/document.xml", document_xml(build_body()))
        z.writestr("word/styles.xml", styles_xml())
    print(OUT)


if __name__ == "__main__":
    main()
