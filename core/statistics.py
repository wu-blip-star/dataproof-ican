"""可执行统计层：保留 V1/V2 路径，V3 扩展显式选择的基础统计方法。"""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning, NearConstantInputWarning

from core.data import DataValidationError, numeric_columns


@dataclass
class PearsonResult:
    sample_size: int
    statistic: float
    p_value: float
    direction: str
    significant: bool
    alpha: float
    used_row_ids: list[int]
    excluded_row_ids: list[int]
    missing_pair_count: int
    nonfinite_pair_count: int
    used_data: pd.DataFrame


def run_pearson(frame: pd.DataFrame, x: str, y: str, alpha: float = 0.05) -> PearsonResult:
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise DataValidationError("显著性水平必须在 0 和 1 之间。")
    if x == y:
        raise DataValidationError("X 和 Y 不能是同一个变量。")
    if x not in numeric_columns(frame) or y not in numeric_columns(frame):
        raise DataValidationError("Pearson 分析要求 X 和 Y 都是数值变量。")
    pair = frame[[x, y]].reset_index(drop=True)
    values = pair.to_numpy(dtype=float, na_value=np.nan)
    missing = np.isnan(values).any(axis=1)
    infinite = np.isinf(values).any(axis=1) & ~missing
    valid = np.isfinite(values).all(axis=1)
    used = pair.loc[valid].copy()
    if len(used) < 3:
        raise DataValidationError("有效样本不足：去除缺失值和无穷值后至少需要 3 对观测。")
    if used[x].nunique() < 2 or used[y].nunique() < 2:
        raise DataValidationError("存在常量列（所有值相同），无法计算 Pearson 相关。")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConstantInputWarning)
        warnings.simplefilter("error", NearConstantInputWarning)
        try:
            result = pearsonr(values[valid, 0], values[valid, 1], alternative="two-sided")
        except (ConstantInputWarning, NearConstantInputWarning) as exc:
            raise DataValidationError("变量为常量或近似常量，相关结果不可靠，请检查数据精度。") from exc
    r, p = float(result.statistic), float(result.pvalue)
    if not np.isfinite([r, p]).all():
        raise DataValidationError("计算出现非有限结果，请检查变量范围和数据精度。")
    direction = "positive" if r > 0 else "negative" if r < 0 else "zero"
    return PearsonResult(len(used), r, p, direction, p < alpha, float(alpha),
                         (np.flatnonzero(valid) + 1).tolist(),
                         (np.flatnonzero(~valid) + 1).tolist(),
                         int(missing.sum()), int(infinite.sum()), used)


# V2 扩展；上方 V1 run_pearson 保持不变。
DEFAULT_BOOTSTRAP_SEED = 20260928
DEFAULT_N_BOOTSTRAP = 1000
IQR_FACTOR = 1.5


def _direction(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _claim_fields(r, p, claim_type, alpha):
    # 运行时导入，复用 evidence.py 中唯一的 V1 Claim 规则，避免模块级循环。
    from core.evidence import judge, detailed_status
    status, reason = judge(r, p, claim_type, alpha)
    return {"claim_status": status, "claim_status_detail": detailed_status(status, p < alpha),
            "judgment_reason": reason}


def _summary(result: PearsonResult, claim_type: str) -> dict:
    return {"available": True, "sample_size": result.sample_size,
            "statistic": result.statistic, "p_value": result.p_value,
            "direction": result.direction, "significant": result.significant,
            "alpha": result.alpha, **_claim_fields(result.statistic, result.p_value, claim_type, result.alpha)}


def _row_identity(frame, row_id):
    value = frame.iloc[row_id - 1]["student_id"] if "student_id" in frame else None
    return {"row_id": int(row_id), "student_id": None if value is None or pd.isna(value) else str(value)}


def run_spearman(frame: pd.DataFrame, x: str, y: str, alpha: float = 0.05) -> dict:
    baseline = run_pearson(frame, x, y, alpha)
    result = spearmanr(baseline.used_data[x], baseline.used_data[y], alternative="two-sided")
    rho, p = float(result.statistic), float(result.pvalue)
    if not np.isfinite([rho, p]).all():
        raise DataValidationError("Spearman 结果无法计算，请检查秩是否有变化。")
    return {"available": True, "sample_size": baseline.sample_size, "rho": rho,
            "statistic": rho, "p_value": p, "direction": _direction(rho),
            "significant": p < alpha, "alpha": float(alpha),
            "used_row_ids": baseline.used_row_ids, "excluded_row_ids": baseline.excluded_row_ids,
            "method_name": "Spearman Rank Correlation", "alternative": "two-sided",
            "p_value_method": "scipy.stats.spearmanr asymptotic approximation",
            "limitation": "Spearman 是另一条方法路径，不比 Pearson 更正确；其 p 值采用渐近近似，小样本或大量并列秩时应谨慎解释，本版本不计算置换 p 值。"}


def bootstrap_correlation(frame: pd.DataFrame, x: str, y: str, alpha: float = 0.05,
                          n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                          random_seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict:
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise DataValidationError("Bootstrap 次数必须是至少为 2 的整数。")
    if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise DataValidationError("随机种子必须是非负整数。")
    baseline = run_pearson(frame, x, y, alpha)
    data = baseline.used_data.to_numpy(dtype=float)
    # 按列缩放以避免数值溢出；正尺度变换不改变相关系数。X/Y 始终成对抽样。
    data = data / np.max(np.abs(data), axis=0)
    n = len(data)
    rng = np.random.default_rng(random_seed)
    distribution = []
    invalid_constant = invalid_near_constant = 0
    batch_size = max(1, min(128, 1_000_000 // n))
    threshold = np.finfo(float).eps ** 0.75  # 与 SciPy Pearson 近常量阈值一致。
    for start in range(0, n_bootstrap, batch_size):
        indices = rng.integers(0, n, size=(min(batch_size, n_bootstrap - start), n))
        sampled = data[indices]
        means = sampled.mean(axis=1, keepdims=True)
        centered = sampled - means
        norms = np.linalg.norm(centered, axis=1)
        constant = (norms == 0).any(axis=1)
        near = (norms < threshold * np.abs(means[:, 0, :])).any(axis=1) & ~constant
        valid = ~constant & ~near
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = centered / norms[:, None, :]
            correlations = np.sum(normalized[:, :, 0] * normalized[:, :, 1], axis=1)
        valid &= np.isfinite(correlations)
        invalid_constant += int(constant.sum())
        invalid_near_constant += int(near.sum())
        distribution.extend(float(np.clip(r, -1, 1)) if ok else None
                            for r, ok in zip(correlations, valid))
    values = np.array([r for r in distribution if r is not None], dtype=float)
    enough = len(values) >= 2
    lower, upper = np.quantile(values, [0.025, 0.975], method="linear") if enough else (None, None)
    return {
        "available": enough, "sample_size": n, "n_bootstrap": int(n_bootstrap),
        "random_seed": int(random_seed), "original_r": baseline.statistic,
        "valid_count": len(values), "invalid_count": int(n_bootstrap - len(values)),
        "invalid_constant_count": invalid_constant, "invalid_near_constant_count": invalid_near_constant,
        "invalid_other_count": int(n_bootstrap - len(values) - invalid_constant - invalid_near_constant),
        "bootstrap_mean_r": float(values.mean()) if len(values) else None,
        "bootstrap_median_r": float(np.median(values)) if len(values) else None,
        "ci_lower": float(lower) if enough else None, "ci_upper": float(upper) if enough else None,
        "confidence_level": 0.95, "ci_method": "paired percentile; quantile interpolation=linear",
        "positive_ratio": float(np.mean(values > 0)) if len(values) else None,
        "negative_ratio": float(np.mean(values < 0)) if len(values) else None,
        "zero_ratio": float(np.mean(values == 0)) if len(values) else None,
        "original_direction_consistency_ratio": float(np.mean(np.sign(values) == np.sign(baseline.statistic))) if len(values) else None,
        "ci_crosses_zero": bool(lower <= 0 <= upper) if enough else None,
        "distribution_degenerate": bool(np.ptp(values) <= 1e-12) if len(values) else None,
        "bootstrap_r": distribution,
        "used_row_ids": baseline.used_row_ids, "excluded_row_ids": baseline.excluded_row_ids,
        "ratio_denominator": "valid_count; invalid resamples excluded, not redrawn",
        "limitation": "95% percentile 区间仅由有效重抽样形成；无效重抽样被记录而不补抽。小样本、退化分布或较多无效抽样时区间不可靠。方向比例不是主张为真的概率，不能解释为因果。",
        "reason": None if enough else "有效重抽样少于 2 次，无法报告百分位区间。",
    }


def jackknife_correlation(frame: pd.DataFrame, x: str, y: str, claim_type: str,
                          alpha: float = 0.05) -> dict:
    baseline = run_pearson(frame, x, y, alpha)
    original = _summary(baseline, claim_type)
    used = baseline.used_data.reset_index(drop=True)
    trials = []
    for position, row_id in enumerate(baseline.used_row_ids):
        identity = _row_identity(frame, row_id)
        try:
            result = run_pearson(used.drop(index=position), x, y, alpha)
            status = _claim_fields(result.statistic, result.p_value, claim_type, alpha)
            trials.append({**identity, "available": True, "sample_size": result.sample_size,
                           "r_without_row": result.statistic, "p_without_row": result.p_value,
                           "delta_r": result.statistic - baseline.statistic,
                           "claim_status_without_row": status["claim_status"],
                           "claim_status_detail_without_row": status["claim_status_detail"],
                           "direction_flipped": result.statistic * baseline.statistic < 0,
                           "direction_changed": result.direction != baseline.direction,
                           "significance_flipped": result.significant != baseline.significant,
                           "claim_status_flipped": status["claim_status"] != original["claim_status"],
                           "claim_detail_flipped": status["claim_status_detail"] != original["claim_status_detail"]})
        except DataValidationError as exc:
            trials.append({**identity, "available": False, "sample_size": len(used) - 1,
                           "r_without_row": None, "p_without_row": None, "delta_r": None,
                           "claim_status_without_row": None, "reason": str(exc)})
    valid = [row for row in trials if row["available"]]
    def extrema(field, function):
        return function(row[field] for row in valid) if valid else None
    def flips(field):
        return sum(row[field] for row in valid) if valid else None
    return {"available": bool(valid), "sample_size": baseline.sample_size,
            "original_r": baseline.statistic, "original_p": baseline.p_value,
            "alpha": float(alpha), "n_jackknife": len(trials),
            "valid_count": len(valid), "invalid_count": len(trials) - len(valid),
            "r_min": extrema("r_without_row", min), "r_max": extrema("r_without_row", max),
            "p_min": extrema("p_without_row", min), "p_max": extrema("p_without_row", max),
            "direction_flip_count": flips("direction_flipped"),
            "direction_changed_count": flips("direction_changed"),
            "significance_flip_count": flips("significance_flipped"),
            "claim_status_flip_count": flips("claim_status_flipped"),
            "claim_detail_flip_count": flips("claim_detail_flipped"),
            "influential_rows": sorted(valid, key=lambda row: (-abs(row["delta_r"]), row["row_id"]))[:5],
            "trials": trials,
            "used_row_ids": baseline.used_row_ids, "excluded_row_ids": baseline.excluded_row_ids,
            "flip_definition": "V1 coarse claim status; counts use valid deletions only; direction flip requires strict sign reversal, zero changes counted separately",
            "limitation": "这些样本对当前统计结论影响较大，建议进一步核查。影响较大不表示数据错误；逐一删除不能覆盖多个样本共同作用。",
            "reason": None if valid else "所有删除场景均无法计算（每次删除后至少需要 3 对非恒定观测）。"}


def iqr_sensitivity(frame: pd.DataFrame, x: str, y: str, claim_type: str,
                    alpha: float = 0.05, factor: float = IQR_FACTOR) -> dict:
    if isinstance(factor, bool) or not np.isfinite(factor) or factor <= 0:
        raise DataValidationError("IQR 倍数必须是正的有限数。")
    baseline = run_pearson(frame, x, y, alpha)
    original = _summary(baseline, claim_type)
    used = baseline.used_data.reset_index(drop=True)
    flagged = np.zeros(len(used), dtype=bool)
    bounds, by_variable = {}, {}
    for variable in (x, y):
        q1, q3 = np.quantile(used[variable].to_numpy(dtype=float), [.25, .75], method="linear")
        iqr = q3 - q1
        low, high = q1 - factor * iqr, q3 + factor * iqr
        if not np.isfinite([q1, q3, iqr, low, high]).all():
            raise DataValidationError("IQR 阈值超出有限数值范围，请检查变量数量级。")
        mask = (used[variable].to_numpy(dtype=float) < low) | (used[variable].to_numpy(dtype=float) > high)
        flagged |= mask
        bounds[variable] = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                            "lower_fence": float(low), "upper_fence": float(high)}
        by_variable[variable] = [baseline.used_row_ids[i] for i in np.flatnonzero(mask)]
    ids = [baseline.used_row_ids[i] for i in np.flatnonzero(flagged)]
    kept_ids = [baseline.used_row_ids[i] for i in np.flatnonzero(~flagged)]
    try:
        result = run_pearson(used.loc[~flagged], x, y, alpha)
        scenario = _summary(result, claim_type)
        changes = {"direction_changed": result.direction != baseline.direction,
                   "significance_changed": result.significant != baseline.significant,
                   "claim_status_changed": scenario["claim_status"] != original["claim_status"],
                   "claim_detail_changed": scenario["claim_status_detail"] != original["claim_status_detail"]}
    except DataValidationError as exc:
        scenario = {"available": False, "sample_size": len(kept_ids), "statistic": None,
                    "p_value": None, "direction": None, "significant": None, "claim_status": None,
                    "claim_status_detail": None, "alpha": float(alpha), "reason": str(exc)}
        changes = dict.fromkeys(["direction_changed", "significance_changed", "claim_status_changed", "claim_detail_changed"])
    return {"available": scenario["available"], "rule": "X or Y < Q1 - factor*IQR or > Q3 + factor*IQR; strict inequalities; linear quantiles on original valid pairs",
            "factor": float(factor), "bounds": bounds, "flagged_count": len(ids),
            "flagged_row_ids": ids, "flagged_rows": [_row_identity(frame, i) for i in ids],
            "flagged_by_variable": by_variable, "retained_row_ids": kept_ids,
            "original": original, "scenario": scenario, **changes,
            "original_used_row_ids": baseline.used_row_ids,
            "original_excluded_row_ids": baseline.excluded_row_ids,
            "limitation": "仅构建排除潜在极端值的敏感性场景，原始数据不变。IQR 标记不代表错误数据，也不能作为自动删除依据；IQR=0 时仍使用同一严格阈值规则。"}


def run_analysis(frame, claim, method=None, alpha=.05):
    """V3 unified result. A sensitivity scenario must pass the original method explicitly."""
    from scipy import stats
    import statsmodels.api as sm
    from core.method_audit import audit_method, prepare_analysis, group_arrays, METHOD_NAMES
    from core.claim_types import claim_judgment
    audit = audit_method(frame, claim, alpha)
    if audit['errors']:
        raise DataValidationError('；'.join(audit['errors']))
    method = method or audit['recommended_method']
    if method not in [audit['recommended_method']] + audit['alternative_methods']:
        raise DataValidationError('所选统计方法不适用于当前研究问题。')
    used, rows, excluded = prepare_analysis(frame, claim)
    n=len(used)
    extra={}
    direction='not_applicable'
    value=0.
    if method in ('pearson','spearman'):
        x,y=claim.variables
        if method=='pearson':
            result=run_pearson(frame,x,y,alpha)
            statistic,p=result.statistic,result.p_value
        else:
            result=stats.spearmanr(used[x],used[y],alternative='two-sided')
            statistic,p=float(result.statistic),float(result.pvalue)
        value=statistic
        effect={'name':'r' if method=='pearson' else 'rho', 'value':statistic}
        extra['p_value_method']='scipy pearsonr' if method=='pearson' else 'scipy spearmanr asymptotic (small samples/ties: caution)'
    elif method in ('student_t','welch_t','anova','kruskal'):
        labels,groups=group_arrays(used,claim)
        extra['group_statistics']=[{'group':str(label),'n':len(g),'mean':float(g.mean()),'sd':float(g.std(ddof=1))}
                                   for label,g in zip(labels,groups)]
        if method in ('student_t','welch_t'):
            a,b=groups
            result=stats.ttest_ind(a,b,equal_var=method=='student_t',alternative='two-sided')
            statistic,p=float(result.statistic),float(result.pvalue)
            ci=result.confidence_interval(confidence_level=.95)
            value=float(a.mean()-b.mean())
            pooled=((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2)
            if pooled<=0:
                raise DataValidationError('两组均无组内变异，t 检验及标准化效应量无法可靠估计。')
            d=value/np.sqrt(pooled)
            g=d*(1-3/(4*(len(a)+len(b)-2)-1))
            effect={'name':'Hedges g (pooled SD, approximate correction)', 'value':float(g)}
            extra.update(mean_difference=value, comparison_order=[str(l) for l in labels], cohen_d=float(d),
                         confidence_interval=[float(ci.low),float(ci.high)], confidence_level=.95, df=float(result.df),
                         effect_size_note='异方差下 pooled SD 标准化效应仅作描述；均值差及所选检验的95%区间为主要解释依据。')
        elif method=='anova':
            result=stats.f_oneway(*groups)
            statistic,p=float(result.statistic),float(result.pvalue)
            all_values=np.concatenate(groups)
            ss_total=np.sum((all_values-all_values.mean())**2)
            ss_between=sum(len(g)*(g.mean()-all_values.mean())**2 for g in groups)
            effect={'name':'eta_squared','value':float(ss_between/ss_total)}
            extra.update(df_between=len(groups)-1,df_within=n-len(groups))
        else:
            try:
                result=stats.kruskal(*groups)
            except ValueError as exc:
                raise DataValidationError('Kruskal 无法计算：请检查各组变异。') from exc
            statistic,p=float(result.statistic),float(result.pvalue)
            effect={'name':'epsilon_squared (clamped at zero)','value':float(max(0,(statistic-len(groups)+1)/(n-len(groups))))}
            extra['interpretation']='检验秩/分布差异；只有额外形状条件满足时才可解释为位置差异，不直接证明均值差异。'
    elif method in ('chi_square','fisher'):
        x,y=claim.variables
        table=pd.crosstab(used[x],used[y])
        chi,p_chi,df,expected=stats.chi2_contingency(table.to_numpy(),correction=False)
        effect={'name':"Cramer's V (uncorrected)",'value':float(np.sqrt(chi/(n*min(table.shape[0]-1,table.shape[1]-1))))}
        extra.update(contingency_table=table.to_numpy().tolist(), row_labels=[str(v) for v in table.index],
                     column_labels=[str(v) for v in table.columns], expected_frequencies=expected.tolist(),df=int(df),
                     chi_square=float(chi), continuity_correction=False)
        if method=='chi_square':
            statistic,p=float(chi),float(p_chi)
        else:
            result=stats.fisher_exact(table.to_numpy(),alternative='two-sided')
            statistic=float(result.statistic) if np.isfinite(result.statistic) else None
            p=float(result.pvalue)
            extra['statistic_note']='Fisher statistic 是优势比；无穷值导出为 null，并用 odds_ratio_infinite 标记。'
            extra['odds_ratio_infinite']=bool(not np.isfinite(result.statistic))
    else:
        design=sm.add_constant(used[claim.predictors].astype(float), has_constant='add')
        fitted=sm.OLS(used[claim.outcome].astype(float),design).fit()
        focus=claim.variables[0]
        statistic,p=float(fitted.tvalues[focus]),float(fitted.pvalues[focus])
        value=float(fitted.params[focus])
        effect={'name':'partial_r (focus coefficient)','value':float(statistic/np.sqrt(statistic**2+fitted.df_resid))}
        ci=fitted.conf_int(.05)
        extra.update(coefficients=[{'variable':str(c),'coefficient':float(fitted.params[c]),'standard_error':float(fitted.bse[c]),
                                    't':float(fitted.tvalues[c]),'p_value':float(fitted.pvalues[c]),
                                    'ci_lower':float(ci.loc[c,0]),'ci_upper':float(ci.loc[c,1])} for c in fitted.params.index],
                     r_squared=float(fitted.rsquared), adjusted_r_squared=float(fitted.rsquared_adj),
                     df_resid=float(fitted.df_resid), covariance_type='nonrobust', focus_variable=focus,
                     focus_coefficient=value, vif=audit['details'].get('vif',{}))
    if (statistic is not None and not np.isfinite(statistic)) or not np.isfinite(p) or not np.isfinite(effect['value']):
        raise DataValidationError('统计计算出现非有限结果，可能样本不足、常量或完全拟合。')
    if method=='ols' and any(not np.isfinite(c[k]) for c in extra['coefficients'] for k in ('coefficient','standard_error','t','p_value','ci_lower','ci_upper')):
        raise DataValidationError('OLS 系数推断非有限，请检查完全拟合或共线性。')
    if method in ('pearson','spearman','student_t','welch_t','ols'):
        direction=_direction(value)
    status,reason=claim_judgment(value,p,claim.direction,alpha)
    if method=='kruskal':
        reason += ' 此路径仅支持分布/秩差异表述，未核验均值差异。'
    return dict(available=True, method=method, method_name=METHOD_NAMES[method], sample_size=n,statistic=statistic,
                p_value=p, effect_size=effect,direction=direction,significant=p<alpha,alpha=float(alpha),
                claim_status=status,judgment_reason=reason,used_row_ids=rows,excluded_row_ids=excluded,
                details=extra,method_audit=audit)
