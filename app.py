from hashlib import sha256
from html import escape
from pathlib import Path

import streamlit as st

from core.charts import scatter_with_trend
from core.data import DataValidationError, excel_sheets, load_dataset, numeric_columns
from core.evidence import (DIRECTION_LABELS, DETAIL_STATUS_LABELS, EFFECT_SIZE_NOTE,
                           format_p, verify_claim, add_stability_analysis)
from generate_demo_data import generate_demo_data
from generate_stability_demo import generate_stability_demo
from ui.stability import render_stability
from ui.audit import render_audit
from ui.claim_parsing import render_v4
from generate_audit_demo import generate_quality_flip_demo, generate_method_audit_demo

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="DataProof 数证链", page_icon="◈", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem;}
h1,h2,h3 {letter-spacing: -.025em;}
.hero {background:#192B40; border-radius:16px; padding:30px 34px; color:white; margin-bottom:26px;}
.hero .eyebrow {font-size:12px; letter-spacing:2px; color:#8CD3CE; margin-bottom:12px;}
.hero h1 {color:white; font-size:36px; padding:0; margin-bottom:8px;}
.hero .slogan {color:#D6E4EE; font-size:19px; margin:0 0 20px;}
.hero .flow {font-size:13px; color:#A9BDCE; border-top:1px solid #3A4B5F; padding-top:17px;}
.evidence-card {background:white; border:1px solid #D9E3EC; border-radius:14px; overflow:hidden;}
.card-head {padding:20px 24px; background:#EDF5F4; border-bottom:1px solid #D9E3EC;}
.card-head b {font-size:19px;}
.card-head small {display:block; color:#597080; margin-top:6px; word-break:break-all;}
.chain-step {display:grid; grid-template-columns:36px 175px 1fr; gap:12px; padding:16px 24px; border-bottom:1px solid #EDF0F4; align-items:start;}
.step-number {color:#157A78; font-weight:700;}
.step-label {font-size:12px; color:#617488; line-height:1.65;}
.step-label strong {display:block; color:#253C52; font-size:14px;}
.step-value {font-size:14px; line-height:1.7; overflow-wrap:anywhere; white-space:pre-wrap;}
.chain-step:last-child {border-bottom:none; background:#F5F9FC;}
@media(max-width:760px){.chain-step{grid-template-columns:25px 1fr;padding:14px;}.step-value{grid-column:2;}.hero{padding:22px;}.hero h1{font-size:28px;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><div class="eyebrow">DATAPROOF / RESEARCH EVIDENCE AUDIT</div>
<h1>DataProof 数证链</h1><p class="slogan">让每一个结论，都有数据作证。</p>
<div class="flow">自然语言主张 → 用户确认 → 数据与方法审计 → 真实计算 → D2C Evidence Graph &nbsp; · &nbsp; V4</div></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 数据工作台")
    source_mode = st.radio("选择数据来源", ["体验模拟数据", "上传我的数据", "合成脆弱性演示", "Demo D · 数据质量翻转", "Demo E · 方法审计"], key="source_mode")
    st.caption("统计始终本地计算 · V4 可选 LLM 仅解析语言")
    st.divider()
    st.markdown("**本阶段支持**")
    st.write("CSV / Excel 数据读取\n\nPearson 相关分析\n\n证据卡片与 JSON 导出\n\n四类稳定性压力测试")
    st.write("V3 数据质量审计 / 多方法核验 / 质量敏感性对照")
    st.write("V4 自然语言主张 / D2C证据图 / HTML审计报告")
    st.divider()
    st.caption("研究提示：数值编码并不代表连续变量；学号、组别编码通常不适合 Pearson 分析。")

content = None
sheet = None
if source_mode in ("Demo D · 数据质量翻转", "Demo E · 方法审计"):
    is_d = source_mode.startswith("Demo D")
    path = ROOT / "data" / ("quality_flip_demo.csv" if is_d else "method_audit_demo.csv")
    generator = generate_quality_flip_demo if is_d else generate_method_audit_demo
    content = path.read_bytes() if path.exists() else generator().to_csv(index=False).encode("utf-8-sig")
    file_name = path.name
    st.info("Synthetic Demo，仅用于功能演示。数据与任何真实研究无关。")
elif source_mode in ("体验模拟数据", "合成脆弱性演示"):
    is_stability_demo = source_mode == "合成脆弱性演示"
    path = ROOT / "data" / ("stability_demo.csv" if is_stability_demo else "demo_students.csv")
    generator = generate_stability_demo if is_stability_demo else generate_demo_data
    content = path.read_bytes() if path.exists() else generator().to_csv(index=False).encode("utf-8-sig")
    file_name = path.name
    st.info("Synthetic Stability Demo · 合成演示数据，仅用于展示稳定性分析功能。80 条记录；建议以默认 α=0.05 体验。" if is_stability_demo else "演示模式 · 500 名大学生合成数据，用于展示核验流程，不代表真实调查结论。")
else:
    uploaded = st.file_uploader("上传研究数据（首行为列名，每行一个样本）", type=["csv", "xlsx", "xls"], key="upload")
    if uploaded is None:
        st.info("请上传 CSV / Excel，或从左侧选择模拟数据开始体验。")
        st.stop()
    file_name = uploaded.name
    content = uploaded.getvalue()
    if Path(file_name).suffix.lower() in (".xlsx", ".xls"):
        try:
            sheets = excel_sheets(content, file_name)
            sheet = st.selectbox("选择 Excel 工作表", sheets, key=f"sheet_{sha256(content).hexdigest()}")
        except DataValidationError as exc:
            st.error(str(exc))
            st.stop()
try:
    dataset = load_dataset(content, file_name, sheet)
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

frame = dataset.frame
numeric = numeric_columns(frame)
st.subheader("01 / 数据概览")
for column, label, value in zip(st.columns(4), ["样本数量", "变量数量", "数值变量", "缺失值（单元格）"],
                                [len(frame), len(frame.columns), len(numeric), int(frame.isna().sum().sum())]):
    column.metric(label, f"{value:,}")
with st.expander("查看原始数据预览与文件指纹", expanded=False):
    st.dataframe(frame.head(50), hide_index=True, width="stretch")
    st.caption(f"仅预览前 50 行 · {dataset.file_name} · 工作表：{dataset.sheet_name or '不适用'}")
    st.code(f"SHA-256: {dataset.sha256}", language=None)
if source_mode != "上传我的数据":
    st.download_button("下载模拟 CSV", content, file_name, "text/csv", key="download_demo")
if source_mode.startswith("Demo D") or source_mode.startswith("Demo E"):
    render_audit(dataset)
    st.stop()
workspace = st.radio("分析工作区", ["V1/V2 相关核验与稳定性", "V3 研究可信性审计", "V4 AI主张与证据图"], horizontal=True, key="audit_workspace")
if workspace == "V4 AI主张与证据图":
    render_v4(dataset)
    st.stop()
if workspace == "V3 研究可信性审计":
    render_audit(dataset)
    st.stop()
if len(numeric) < 2:
    st.warning("至少需要两个数值变量才能进行相关分析。带单位、文本或混合内容的列，请先整理为纯数值。")
    st.stop()

if st.session_state.get("source_id") != dataset.source_id:
    for key in ("x", "y", "claim_type", "claim_text", "last_evidence", "last_signature", "preset", "previous_preset"):
        st.session_state.pop(key, None)
    st.session_state.source_id = dataset.source_id
    if source_mode == "合成脆弱性演示":
        st.session_state.update(x="synthetic_x", y="synthetic_y", claim_type="正相关",
                                claim_text="合成变量 X 与 Y 呈正相关（仅作功能演示）。")

st.subheader("02 / 研究主张核验")
left, right = st.columns([1.5, 1], gap="large")
with left:
    if source_mode == "体验模拟数据":
        preset = st.selectbox("快速体验研究主张", ["Test A · 学习时间与成绩", "Test B · 短视频与成绩", "Test C · 睡眠与成绩", "自定义主张"], key="preset")
        if st.session_state.get("previous_preset") != preset:
            scenarios = {
                "Test A · 学习时间与成绩": ("study_hours", "正相关", "每天学习时间越长，考试成绩越高。"),
                "Test B · 短视频与成绩": ("short_video_hours", "负相关", "每天短视频使用时间越长，考试成绩显著越低。"),
                "Test C · 睡眠与成绩": ("sleep_hours", "负相关", "每天睡眠时间越长，考试成绩越低。"),
            }
            if preset in scenarios:
                x_value, type_value, text_value = scenarios[preset]
                st.session_state.update(x=x_value, y="exam_score", claim_type=type_value, claim_text=text_value)
            st.session_state.previous_preset = preset
    selectors = st.columns(2)
    x = selectors[0].selectbox("自变量 X", numeric, key="x")
    y = selectors[1].selectbox("因变量 Y", numeric, index=1, key="y")
    settings = st.columns([2, 1])
    claim_type = settings[0].selectbox("主张类型", ["正相关", "负相关", "有显著差异", "无显著差异"], key="claim_type")
    alpha = settings[1].number_input("显著性水平 α", min_value=0.001, max_value=0.2, value=0.05, step=0.01, format="%.3f", key="alpha")
    st.caption(f"本次待检验命题：{x} 与 {y} 呈{claim_type}。" if claim_type in ("正相关", "负相关") else "差异主张需要另行设计分组与差异检验。")
    claim_text = st.text_area("原始主张（可选，仅记录，不做语义解析）", key="claim_text", placeholder="例如：每天学习时间越长，考试成绩越高。", max_chars=2000)
    unsupported = claim_type not in ("正相关", "负相关")
    if unsupported:
        st.warning("第一阶段仅实现 Pearson 相关检验，暂不能核验“有／无显著差异”。不显著也不能证明没有差异。")
    if x == y:
        st.warning("请选择两个不同的数值变量。")
    execute = st.button("执行统计核验", type="primary", disabled=unsupported or x == y, key="verify", width="stretch")
with right:
    with st.container(border=True):
        st.markdown("#### 一条可追溯的证据链")
        st.markdown("**01** 记录主张与变量映射\n\n**02** 确定实际参与分析的样本\n\n**03** 调用 SciPy 真实计算\n\n**04** 给出证据判断与建议表述")
        st.divider()
        st.caption("采用双侧检验，p < α 时达到统计显著。缺失值与无穷值按 X/Y 成对剔除；其他变量的缺失不影响本次分析。")
        st.caption("系统仅核验所选变量的相关方向与显著性。原始文本中的因果或其他限定，不属于本次核验范围。")

signature = (dataset.source_id, x, y, claim_type, alpha, claim_text)
if st.session_state.get("last_signature") != signature:
    st.session_state.pop("last_evidence", None)
if execute:
    try:
        with st.spinner("正在计算 Pearson 相关并建立证据链…"):
            node, calculation = verify_claim(dataset, x, y, claim_type, alpha, claim_text)
            figure = scatter_with_trend(calculation.used_data, x, y)
            st.session_state.last_evidence = (node, calculation, figure)
            st.session_state.last_signature = signature
    except DataValidationError as exc:
        st.error(str(exc))

if "last_evidence" in st.session_state and getattr(st.session_state.last_evidence[0], "evidence_detail_status", None) is None:
    st.session_state.pop("last_evidence", None)
if "last_evidence" not in st.session_state:
    st.caption("选择变量并执行核验后，这里将生成统计结果、散点图和 Evidence Card。")
    st.stop()

node, calculation, figure = st.session_state.last_evidence
st.divider()
st.subheader("03 / 核验结果")
message = f"{DETAIL_STATUS_LABELS[node.evidence_detail_status]} {node.judgment_reason}"
if node.evidence_status == "Supported":
    st.success(message)
elif node.evidence_status == "Direction Conflict" and node.significant:
    st.error(message)
else:
    st.warning(message)
st.caption(f"判断范围：{node.x_variable} 与 {node.y_variable} 的{node.claim_type}主张；原始文本仅存档。")
st.write(f"关系强度：**{node.effect_size['label']}**（|r| = {abs(node.statistic):.4f}）。")
st.caption(EFFECT_SIZE_NOTE)
for column, label, value in zip(st.columns(4), ["有效样本 n", "Pearson r", "双侧 p-value", "统计显著"],
                                [str(node.sample_size), f"{node.statistic:.4f}", format_p(node.p_value), "是" if node.significant else "否"]):
    column.metric(label, value)
st.write(f"关系方向：**{DIRECTION_LABELS[node.direction]}**。在这 {node.sample_size} 个有效样本中，"
         + ("X 较大的观测通常伴随较高的 Y。" if node.statistic > 0 else "X 较大的观测通常伴随较低的 Y。" if node.statistic < 0 else "没有观察到线性相关方向。"))
st.caption("“通常”仅描述当前样本趋势，不代表每个个体均如此。显著性与相关强度是两回事。")
st.info("相关不等于因果。当前核验不能证明 X 导致 Y，也不能将不显著解释为没有关系。")
st.plotly_chart(figure, width="stretch")
st.caption(f"图表与检验使用同一组有效样本 · 已排除 {len(calculation.excluded_row_ids)} 行 · 趋势线为带截距 OLS 拟合，不用于因果推断。")

st.subheader("04 / D2C Evidence Card")
steps = [
    ("Claim", "研究主张", node.claim_text),
    ("Variable Mapping", "变量映射", f"{node.x_variable} → {node.y_variable}\n结构化主张：{node.claim_type}；来源：{dataset.file_name}"),
    ("Used Sample", "使用样本", f"n = {node.sample_size} / {len(frame)}；剔除 {len(calculation.excluded_row_ids)} 行\n缺失配对：{calculation.missing_pair_count}；其余非有限配对：{calculation.nonfinite_pair_count}"),
    ("Statistical Method", "统计方法", f"Pearson Correlation · 双侧检验 · α = {node.alpha:g}"),
    ("Statistical Result", "统计结果", f"r = {node.statistic:.4f}；p = {format_p(node.p_value)}\n{DIRECTION_LABELS[node.direction]}；{'达到' if node.significant else '未达到'}统计显著"),
    ("Evidence Judgment", "证据判断", f"{node.evidence_detail_status}\n{node.judgment_reason}\n仅针对结构化方向主张，不审核原始文本语义。"),
    ("Suggested Wording", "建议表述", node.suggested_wording + " 相关不等于因果。"),
]
if node.stability_analysis is not None:
    steps.insert(-1, ("Stability Analysis", "稳定性分析", "已计算：方法敏感性 / Bootstrap / Jackknife / IQR 敏感性\n五维稳定性画像及各路径细节见下方，完整结果包含在 JSON 中。"))
rows = "".join(f'<div class="chain-step"><div class="step-number">{i:02d}</div><div class="step-label"><strong>{escape(cn)}</strong>{escape(en)}</div><div class="step-value">{escape(value)}</div></div>'
               for i, (en, cn, value) in enumerate(steps, 1))
st.markdown(f'<div class="evidence-card"><div class="card-head"><b>Evidence Chain / 研究证据链</b><small>记录 {node.evidence_id} · {node.created_at}</small></div>{rows}</div>', unsafe_allow_html=True)
st.write("")
export_left, export_right = st.columns(2)
export_left.download_button("导出完整证据链 JSON", node.to_json(), f"evidence_{node.evidence_id}.json", "application/json", key="download_evidence", width="stretch")
used_export = calculation.used_data.copy()
row_id_name = "_dataproof_source_row_id"
while row_id_name in used_export.columns:
    row_id_name = "_" + row_id_name
used_export.insert(0, row_id_name, calculation.used_row_ids)
export_right.download_button("导出本次有效样本 CSV", used_export.to_csv(index=False).encode("utf-8-sig"), f"used_sample_{node.evidence_id}.csv", "text/csv", key="download_sample", width="stretch")
with st.expander("查看样本处理、文件指纹与结构化记录"):
    st.write(node.sample["processing"])
    st.caption(node.sample["row_id_definition"])
    st.json(node.to_dict(), expanded=False)
with st.expander("方法适用条件与解释边界"):
    for limitation in node.limitations:
        st.write("• " + limitation)
st.divider()
st.caption("按需运行：1000 次成对 Bootstrap（固定种子）与全部有效样本的逐一删除。修改变量、主张、α 或数据后需重新核验。")
if node.sample_size > 5000:
    st.warning("样本较大，完整逐一删除计算可能较慢；不会自动减少样本或删除场景。")
if st.button("进行证据稳定性压力测试", key="run_stability", type="primary"):
    try:
        with st.spinner("正在执行 Spearman、1000 次 Bootstrap、逐一删除与 IQR 敏感性分析…"):
            updated_node = add_stability_analysis(node, dataset)
            st.session_state.last_evidence = (updated_node, calculation, figure)
        st.rerun()
    except DataValidationError as exc:
        st.error(str(exc))
if node.stability_analysis is not None:
    render_stability(node)
st.caption("DataProof 数证链 · V2 · 保存原始数据文件与 JSON，才能重新执行与复核分析；文件指纹用于校验一致性，不证明数据本身真实。")
