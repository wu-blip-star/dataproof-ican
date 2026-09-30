import pandas as pd
import streamlit as st

from core.charts import bootstrap_distribution_chart
from core.evidence import DIRECTION_LABELS, EFFECT_SIZE_NOTE, format_p


def _number(value, digits=4):
    return "不可计算" if value is None else f"{value:.{digits}f}"


def _yes(value):
    return "不可判断" if value is None else "是" if value else "否"


def render_stability(node):
    analysis = node.stability_analysis
    original = analysis["original"]
    method = analysis["method_sensitivity"]
    boot = analysis["bootstrap"]
    jack = analysis["jackknife"]
    out = analysis["outlier_sensitivity"]
    st.subheader("05 / 证据稳定性概览")
    st.caption("Evidence Stability · 分别展示不同分析路径，不合成为任意评分，也不把一致不显著解释为证明无关系。")
    columns = st.columns(4)
    columns[0].metric("方法判断一致", _yes(method["consistent"]))
    columns[1].metric("Bootstrap 区间含 0", _yes(boot["ci_crosses_zero"]))
    columns[2].metric("逐一删除：主张状态变化", str(jack["claim_status_flip_count"]) if jack["available"] else "不可计算")
    columns[3].metric("IQR 场景：主张状态变化", _yes(out["claim_status_changed"]))

    st.markdown("#### 方法敏感性与分析路径对照")
    comparison = []
    for label, result in [("原始 Pearson", original), ("Spearman", method["spearman"]),
                           ("排除 IQR 标记后的 Pearson", out["scenario"])]:
        comparison.append({"分析路径": label, "n": result["sample_size"],
                           "r / ρ": _number(result["statistic"]),
                           "p（双侧）": format_p(result["p_value"]) if result["p_value"] is not None else "不可计算",
                           "方向": DIRECTION_LABELS.get(result["direction"], "不可计算"),
                           "关系强度": result.get("effect_size", {}).get("label", "不可计算"),
                           "Claim 判断": result["claim_status_detail"] or "不可计算"})
    st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
    st.write(f"Pearson / Spearman：方向一致：{_yes(method['direction_consistent'])}；显著性一致：{_yes(method['significance_consistent'])}；主张判断一致：{_yes(method['claim_status_consistent'])}。")
    st.caption(method["spearman"]["limitation"])
    st.caption(EFFECT_SIZE_NOTE)
    if not out["available"]:
        st.warning("IQR 排除场景无法计算：" + out["scenario"]["reason"])

    st.markdown("#### Bootstrap / 成对重抽样")
    st.caption(f"请求 {boot['n_bootstrap']} 次；有效 {boot['valid_count']} 次；无效 {boot['invalid_count']} 次（不补抽）。固定种子 {boot['random_seed']}；95% CI 固定，不随主张 α 改变。")
    if boot["available"]:
        st.write(f"95% percentile CI：**[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]**；{'包含 0，方向存在不确定性' if boot['ci_crosses_zero'] else '不包含 0，但这不证明因果或总体必然存在该方向'}。")
        st.write(f"{boot['n_bootstrap']} 次尝试中的 {boot['valid_count']} 次有效重抽样里，**{boot['original_direction_consistency_ratio']:.1%}** 保持与原始结果相同的方向。")
        st.caption(f"正向比例 {boot['positive_ratio']:.1%} · 负向比例 {boot['negative_ratio']:.1%} · 零值比例 {boot['zero_ratio']:.1%} · 平均 r = {boot['bootstrap_mean_r']:.4f} · 中位数 r = {boot['bootstrap_median_r']:.4f}")
    else:
        st.warning(boot["reason"])
    if boot["invalid_count"] or boot["distribution_degenerate"] or boot["sample_size"] < 10:
        st.warning("存在无效重抽样、退化分布或极小样本；百分位区间及方向比例需特别谨慎，不能直接视作稳定性保证。")
    st.plotly_chart(bootstrap_distribution_chart(boot), width="stretch", key="bootstrap_distribution")
    st.caption(boot["limitation"])
    draws = pd.DataFrame({"replicate": range(1, boot["n_bootstrap"] + 1), "pearson_r": boot["bootstrap_r"],
                          "valid": [value is not None for value in boot["bootstrap_r"]]})
    st.download_button("下载 Bootstrap 分布 CSV", draws.to_csv(index=False).encode("utf-8-sig"),
                       f"bootstrap_{node.evidence_id}.csv", "text/csv", key="download_bootstrap")

    st.markdown("#### Jackknife / 逐一删除敏感性")
    st.write(f"共尝试 {jack['n_jackknife']} 次；有效 {jack['valid_count']} 次，无效 {jack['invalid_count']} 次。所有变化计数仅针对有效场景。")
    if jack["available"]:
        st.write(f"r 范围：[{jack['r_min']:.4f}, {jack['r_max']:.4f}]；p 范围：[{format_p(jack['p_min'])}, {format_p(jack['p_max'])}]。")
        st.write(f"方向严格翻转 **{jack['direction_flip_count']}** 次；显著性变化 **{jack['significance_flip_count']}** 次；V1 主张状态变化 **{jack['claim_status_flip_count']}** 次。")
        top = pd.DataFrame(jack["influential_rows"])[["row_id", "student_id", "r_without_row", "p_without_row", "delta_r", "claim_status_detail_without_row"]]
        top.columns = ["原始记录号", "student_id", "删除后 r", "删除后 p", "Δr（删除后−原始）", "删除后 Claim 判断"]
        st.markdown("**影响较大的样本 Top 5（按 |Δr| 降序）**")
        st.dataframe(top, hide_index=True, width="stretch")
    else:
        st.warning(jack["reason"])
    st.caption(jack["limitation"])
    st.caption("主张状态变化沿用 V1 的四类判断；方向翻转只计算正负互换，涉及零的方向变化另存 JSON。未能计算的删除场景不能视为不变。")
    st.download_button("下载全部 Jackknife 场景 CSV", pd.DataFrame(jack["trials"]).to_csv(index=False).encode("utf-8-sig"),
                       f"jackknife_{node.evidence_id}.csv", "text/csv", key="download_jackknife")

    st.markdown("#### 潜在极端值敏感性 / IQR")
    st.write(f"按 X 或 Y 超出 Q1 − {out['factor']:g}×IQR、Q3 + {out['factor']:g}×IQR 的严格阈值，识别出 **{out['flagged_count']}** 条记录；敏感性场景保留 **{out['scenario']['sample_size']}** 条。")
    st.write(f"方向变化：{_yes(out['direction_changed'])}；显著性变化：{_yes(out['significance_changed'])}；主张状态变化：{_yes(out['claim_status_changed'])}。")
    st.caption(out["limitation"])
    with st.expander("查看 IQR 阈值与原始记录号"):
        st.dataframe(pd.DataFrame(out["bounds"]).T.rename_axis("变量").reset_index(), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(out["flagged_rows"], columns=["row_id", "student_id"]), hide_index=True, width="stretch")
        st.caption(node.sample["row_id_definition"])

    st.markdown("#### Evidence Stability Profile / 证据稳定性画像")
    profile_rows = [
        ("Original Evidence / 原始证据", f"{node.evidence_detail_status}；关系强度：{node.effect_size['label']}"),
        ("Method Consistency / 方法一致性", f"方向：{_yes(method['direction_consistent'])}；显著性：{_yes(method['significance_consistent'])}；判断：{_yes(method['claim_status_consistent'])}"),
        ("Bootstrap Robustness / 重抽样", f"95% CI [{_number(boot['ci_lower'])}, {_number(boot['ci_upper'])}]；含0：{_yes(boot['ci_crosses_zero'])}；方向一致：{boot['original_direction_consistency_ratio']:.1%}" if boot["available"] else "有效结果不足"),
        ("Sample Fragility / 样本脆弱性", f"主张状态变化：{jack['claim_status_flip_count']}；显著性变化：{jack['significance_flip_count']}；无效场景：{jack['invalid_count']}" if jack["available"] else "所有删除场景均无法计算"),
        ("Outlier Sensitivity / 极端值敏感性", f"排除后 r={_number(out['scenario']['statistic'])}，p={_number(out['scenario']['p_value'])}；主张变化：{_yes(out['claim_status_changed'])}"),
    ]
    st.dataframe(pd.DataFrame(profile_rows, columns=["维度", "实际结果"]), hide_index=True, width="stretch")
    st.info(node.suggested_wording)
    st.download_button("下载含稳定性分析的完整 Evidence Chain JSON", node.to_json(),
                       f"evidence_v2_{node.evidence_id}.json", "application/json", key="download_stability_evidence")
    with st.expander("稳定性方法与解释边界"):
        for text in analysis["limitations"]:
            st.write("• " + text)
