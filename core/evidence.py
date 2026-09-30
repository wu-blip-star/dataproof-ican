"""可序列化证据节点；节点与有向边保留向 D2C Evidence Graph 扩展的入口。"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from hashlib import sha256
import json
from uuid import uuid4

from core.data import Dataset, DataValidationError
from core.statistics import (PearsonResult, run_pearson, run_spearman, bootstrap_correlation,
                             jackknife_correlation, iqr_sensitivity,
                             DEFAULT_BOOTSTRAP_SEED, DEFAULT_N_BOOTSTRAP)

STATUS_LABELS = {
    "Supported": "当前数据支持该结论。",
    "Evidence Insufficient": "方向一致，但当前证据不足以支持显著关系。",
    "Direction Conflict": "样本方向与主张不一致，需结合显著性解释。",
    "Unsupported": "当前数据未支持所主张的方向。",
}
DIRECTION_LABELS = {"positive": "正相关", "negative": "负相关", "zero": "无线性相关方向"}
DETAIL_STATUS_LABELS = {
    **STATUS_LABELS,
    "Significant Direction Conflict": "当前数据提供了与原主张显著相反的相关证据。",
    "Sample Direction Conflict / Evidence Insufficient": "样本方向与主张不一致，但反向关系本身也缺乏统计显著证据。",
}
EFFECT_SIZE_BANDS = ((0.1, "极弱（系数接近0）"), (0.3, "弱"), (0.5, "中等"), (None, "较强"))
EFFECT_SIZE_NOTE = "该分级仅用于描述性解释，不代表领域统一标准。统计显著不等于关系强，接近零也不排除其他形式的关系。"


def detailed_status(status: str, significant: bool) -> str:
    if status == "Direction Conflict":
        return "Significant Direction Conflict" if significant else "Sample Direction Conflict / Evidence Insufficient"
    return status


def effect_size_description(coefficient: float) -> dict:
    magnitude = abs(coefficient)
    for upper, label in EFFECT_SIZE_BANDS:
        if upper is None or magnitude < upper:
            return {"absolute_coefficient": magnitude, "label": label,
                    "bands": [{"upper_exclusive": bound, "label": text} for bound, text in EFFECT_SIZE_BANDS],
                    "note": EFFECT_SIZE_NOTE}
LIMITATIONS = [
    "相关不等于因果，不能据此断言 X 导致 Y。",
    "本结果仅核验结构化方向主张；不解析自由文本中的因果、分组、控制变量或其他限定。",
    "双侧 Pearson 检验假设观测独立；常规精确 p 值依赖零相关假设下的分布条件。",
    "原始 Pearson 未自动诊断线性关系、分布条件及异常值影响；V2 压力测试也只覆盖指定敏感性场景。显著不等于效应强或实际重要。",
    "未达到显著性不代表没有关系，也不能证明两组相同；重复探索未做多重比较校正。",
]


def format_p(p: float) -> str:
    return f"{p:.3e}" if p < 0.001 else f"{p:.6f}"


def judge(r: float, p: float, claim_type: str, alpha: float) -> tuple[str, str]:
    if claim_type not in ("正相关", "负相关"):
        raise DataValidationError("差异主张需要组间差异检验；第一阶段仅支持正相关和负相关主张。")
    expected = 1 if claim_type == "正相关" else -1
    if r * expected < 0:
        status = "Direction Conflict"
        reason = f"样本相关方向与主张相反（r = {r:.4f}）。"
        if p >= alpha:
            reason += "但该相关未达显著，不能据此认定总体存在相反关系。"
        else:
            reason += f"双侧 p = {format_p(p)} < α = {alpha:g}，当前数据提供显著反向相关证据。"
    elif r == 0:
        status = "Unsupported"
        reason = "样本 Pearson r 为 0，未观察到所主张的线性相关方向。"
    elif p < alpha:
        status = "Supported"
        reason = f"相关方向与主张一致，且双侧 p = {format_p(p)} < α = {alpha:g}。"
    else:
        status = "Evidence Insufficient"
        reason = f"相关方向与主张一致，但双侧 p = {format_p(p)} ≥ α = {alpha:g}。"
    return status, reason


@dataclass
class EvidenceNode:
    schema_version: str
    evidence_id: str
    created_at: str
    claim_text: str
    claim_type: str
    verification_scope: str
    x_variable: str
    y_variable: str
    sample_size: int
    method_name: str
    statistic: float | None
    p_value: float
    direction: str
    evidence_status: str
    judgment_reason: str
    suggested_wording: str
    alpha: float
    significant: bool
    source: dict
    sample: dict
    method_parameters: dict
    limitations: list[str]
    chain: list[dict]
    edges: list[dict]
    evidence_detail_status: str | None = None
    effect_size: dict | None = None
    stability_analysis: dict | None = None
    structured_claim: dict | None = None
    original_text: str | None = None
    parsed_claim: dict | None = None
    quality_evidence: dict | None = None
    method_audit: dict | None = None
    quality_sensitivity: dict | None = None
    statistical_result: dict | None = None
    final_judgment: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)


def verify_claim(dataset: Dataset, x: str, y: str, claim_type: str,
                 alpha: float = 0.05, claim_text: str = "") -> tuple[EvidenceNode, PearsonResult]:
    if claim_type not in ("正相关", "负相关"):
        raise DataValidationError("第一阶段仅实现相关主张核验；有／无显著差异不能用 Pearson 检验判断。")
    result = run_pearson(dataset.frame, x, y, alpha)
    status, reason = judge(result.statistic, result.p_value, claim_type, alpha)
    direction_cn = DIRECTION_LABELS[result.direction]
    if result.significant:
        suggested = f"在当前样本中，{x} 与 {y} 呈显著{direction_cn}（n = {result.sample_size}，r = {result.statistic:.4f}，p = {format_p(result.p_value)}，双侧检验）。"
    else:
        suggested = f"在当前样本中，尚未发现 {x} 与 {y} 的统计显著线性相关（n = {result.sample_size}，r = {result.statistic:.4f}，p = {format_p(result.p_value)}，α = {alpha:g}）。"
    evidence_id = str(uuid4())
    claim = claim_text.strip() or f"{x} 与 {y} 呈{claim_type}。"
    source = {
        "file_name": dataset.file_name, "file_sha256": dataset.sha256,
        "sheet_name": dataset.sheet_name, "parser": dataset.parser,
        "total_rows": len(dataset.frame), "total_columns": len(dataset.frame.columns),
    }
    used_csv = result.used_data.to_csv(index=False, lineterminator="\n")
    sample = {
        "used_row_ids": result.used_row_ids, "excluded_row_ids": result.excluded_row_ids,
        "row_id_definition": "解析后的数据记录从 1 开始编号，不含表头；Excel 对应工作表行号减 1；CSV 含多行字段时不等同于物理行号。",
        "missing_pair_count": result.missing_pair_count,
        "nonfinite_pair_count": result.nonfinite_pair_count,
        "used_values_sha256": sha256(used_csv.encode("utf-8")).hexdigest(),
        "processing": "仅对 X/Y 成对剔除缺失值和正负无穷值；不插补、不去重、不剔除异常值、不标准化；列名去除首尾空格。",
    }
    method_parameters = {
        "alternative": "two-sided", "null_hypothesis": "population Pearson rho = 0",
        "significance_rule": "p_value < alpha (strict)", "alpha": alpha,
        "implementation": "scipy.stats.pearsonr",
        "library_versions": {name: version(name) for name in ("scipy", "numpy", "pandas", "statsmodels")},
    }
    contents = [
        ("claim", "Claim", {"text": claim, "type": claim_type}),
        ("variables", "Variable Mapping", {"x": x, "y": y, "source": source}),
        ("sample", "Used Sample", {"n": result.sample_size, **sample}),
        ("method", "Statistical Method", {"name": "Pearson Correlation", **method_parameters}),
        ("result", "Statistical Result", {"r": result.statistic, "p_value": result.p_value, "direction": result.direction, "significant": result.significant, "effect_size": effect_size_description(result.statistic)}),
        ("judgment", "Evidence Judgment", {"status": status, "status_detail": detailed_status(status, result.significant), "reason": reason}),
        ("wording", "Suggested Wording", {"text": suggested}),
    ]
    chain = [{"node_id": f"{evidence_id}:{key}", "node_type": key, "label": label, "data": data}
             for key, label, data in contents]
    edges = [{"from": a["node_id"], "to": b["node_id"], "relation": "next_evidence_step"}
             for a, b in zip(chain, chain[1:])]
    node = EvidenceNode(
        "1.1", evidence_id, datetime.now(timezone.utc).isoformat(), claim, claim_type,
        "structured_direction_only", x, y, result.sample_size, "Pearson Correlation",
        result.statistic, result.p_value, result.direction, status, reason, suggested,
        float(alpha), result.significant, source, sample, method_parameters,
        LIMITATIONS.copy(), chain, edges,
    )
    node.evidence_detail_status = detailed_status(status, result.significant)
    node.effect_size = effect_size_description(result.statistic)
    return node, result


def add_stability_analysis(node: EvidenceNode, dataset: Dataset,
                           n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                           random_seed: int = DEFAULT_BOOTSTRAP_SEED) -> EvidenceNode:
    """扩展同一个 EvidenceNode；返回副本，保留原始分析和来源，不改写原数据。"""
    x, y, alpha = node.x_variable, node.y_variable, node.alpha
    original_check = run_pearson(dataset.frame, x, y, alpha)
    used_hash = sha256(original_check.used_data.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    if (dataset.sha256 != node.source["file_sha256"] or dataset.sheet_name != node.source["sheet_name"]
            or dataset.file_name != node.source["file_name"] or used_hash != node.sample["used_values_sha256"]
            or original_check.used_row_ids != node.sample["used_row_ids"]
            or original_check.statistic != node.statistic or original_check.p_value != node.p_value):
        raise DataValidationError("当前数据与原始证据记录不一致，请先重新执行原始核验。")
    original = {"method_name": node.method_name, "sample_size": node.sample_size,
                "statistic": node.statistic, "p_value": node.p_value,
                "direction": node.direction, "significant": node.significant, "alpha": alpha,
                "claim_status": node.evidence_status, "claim_status_detail": node.evidence_detail_status,
                "effect_size": node.effect_size}
    spearman = run_spearman(dataset.frame, x, y, alpha)
    sp_status, sp_reason = judge(spearman["rho"], spearman["p_value"], node.claim_type, alpha)
    spearman.update(claim_status=sp_status, claim_status_detail=detailed_status(sp_status, spearman["significant"]),
                    judgment_reason=sp_reason, effect_size=effect_size_description(spearman["rho"]))
    comparisons = {"direction_consistent": spearman["direction"] == node.direction,
                   "significance_consistent": spearman["significant"] == node.significant,
                   "claim_status_consistent": sp_status == node.evidence_status,
                   "claim_detail_consistent": spearman["claim_status_detail"] == node.evidence_detail_status}
    method = {"pearson": original, "spearman": spearman, **comparisons,
              "consistent": all(comparisons.values())}
    boot = bootstrap_correlation(dataset.frame, x, y, alpha, n_bootstrap, random_seed)
    jack = jackknife_correlation(dataset.frame, x, y, node.claim_type, alpha)
    outlier = iqr_sensitivity(dataset.frame, x, y, node.claim_type, alpha)
    if outlier["scenario"]["available"]:
        outlier["scenario"]["effect_size"] = effect_size_description(outlier["scenario"]["statistic"])
    profile = {
        "original_evidence": {"supported": node.evidence_status == "Supported",
                              "status_detail": node.evidence_detail_status, "effect_size": node.effect_size["label"]},
        "method_consistency": comparisons,
        "bootstrap_robustness": {key: boot[key] for key in ("available", "valid_count", "invalid_count", "ci_lower", "ci_upper", "ci_crosses_zero", "original_direction_consistency_ratio", "distribution_degenerate")},
        "sample_fragility": {key: jack[key] for key in ("available", "valid_count", "invalid_count", "claim_status_flip_count", "significance_flip_count", "direction_flip_count")},
        "outlier_sensitivity": {"available": outlier["available"], "flagged_count": outlier["flagged_count"],
                                "statistic": outlier["scenario"]["statistic"], "p_value": outlier["scenario"]["p_value"],
                                "claim_status_changed": outlier["claim_status_changed"]},
    }
    stability = {"analysis_version": "2.0", "computed_at": datetime.now(timezone.utc).isoformat(),
                 "original": original, "method_sensitivity": method, "bootstrap": boot,
                 "jackknife": jack, "outlier_sensitivity": outlier, "profile": profile,
                 "parameters": {"n_bootstrap": int(n_bootstrap), "random_seed": int(random_seed),
                                "confidence_level": .95, "alpha": alpha, "iqr_factor": outlier["factor"]},
                 "limitations": [spearman["limitation"], boot["limitation"], jack["limitation"], outlier["limitation"],
                                 "重抽样假设观测独立同分布，不适用于直接处理时序、聚类或重复测量。未做多重比较校正，路径比较不能用于挑选最小 p 值。",
                                 "本画像不产生总评分或总稳定标签；已支持主张也可能不稳定，一致不显著也不等于证实无关系。"]}
    updated = deepcopy(node)
    updated.schema_version = "2.0"
    updated.stability_analysis = stability
    updated.chain = [step for step in updated.chain if step["node_type"] != "stability"]
    subnodes = [{"node_id": f"{node.evidence_id}:stability:{key}", "node_type": key,
                 "data_ref": f"stability_analysis.{key}"}
                for key in ("method_sensitivity", "bootstrap", "jackknife", "outlier_sensitivity")]
    stability_step = {"node_id": f"{node.evidence_id}:stability", "node_type": "stability",
                      "label": "Stability Analysis", "data": {"profile": profile, "data_ref": "stability_analysis"},
                      "subnodes": subnodes}
    wording_position = next(i for i, step in enumerate(updated.chain) if step["node_type"] == "wording")
    updated.chain.insert(wording_position, stability_step)
    base_wording = next(step["data"].get("original_text", step["data"]["text"]) for step in updated.chain if step["node_type"] == "wording")
    boot_text = (f"Bootstrap 95% 百分位区间为 [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]，{'包含' if boot['ci_crosses_zero'] else '不包含'}0"
                 if boot["available"] else "Bootstrap 有效结果不足，无法解释区间")
    jack_text = (f"逐一删除的 {jack['valid_count']} 个有效场景中，{jack['claim_status_flip_count']} 次 V1 主张状态变化"
                 if jack["available"] else "逐一删除场景均无法计算")
    out_text = (f"排除 {outlier['flagged_count']} 个 IQR 标记样本后主张状态{'变化' if outlier['claim_status_changed'] else '未变化'}"
                if outlier["available"] else "IQR 排除场景无法计算")
    updated.suggested_wording = (base_wording + f" 稳定性检查：Spearman 与原始方法的方向、显著性及判断{'一致' if method['consistent'] else '不完全一致'}；"
                                 + boot_text + "；" + jack_text + "；" + out_text + "。这些结果仅描述分析路径敏感性，不证明因果关系。")
    for step in updated.chain:
        if step["node_type"] == "wording":
            step["data"] = {"text": updated.suggested_wording, "original_text": base_wording}
    updated.edges = [{"from": a["node_id"], "to": b["node_id"], "relation": "next_evidence_step"}
                     for a, b in zip(updated.chain, updated.chain[1:])]
    updated.edges.extend({"from": stability_step["node_id"], "to": child["node_id"], "relation": "has_analysis_path"}
                         for child in subnodes)
    return updated


def verify_research_claim(dataset, claim, method=None, alpha=.05, quality_config=None,
                          sensitivity_rules=(), effect_change_threshold=.1, include_stability=False):
    """V3 public entrypoint, extends the existing EvidenceNode (no parallel report schema)."""
    from core.research_audit import build_research_evidence
    return build_research_evidence(dataset, claim, method, alpha, quality_config,
                                  sensitivity_rules, effect_change_threshold, include_stability)
