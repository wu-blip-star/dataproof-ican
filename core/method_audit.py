"""Transparent method selection for explicit research questions; no LLM."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from core.data import DataValidationError, numeric_columns

METHOD_NAMES = {'pearson':'Pearson Correlation', 'spearman':'Spearman Rank Correlation',
                'student_t':'Student independent t-test', 'welch_t':'Welch independent t-test',
                'anova':'One-way ANOVA', 'kruskal':'Kruskal-Wallis',
                'chi_square':'Chi-square independence', 'fisher':'Fisher exact (2×2)',
                'ols':'Multiple Linear Regression (OLS)'}


def prepare_analysis(frame, claim):
    claim.validate(frame)
    cols = claim.columns()
    work = frame[cols].reset_index(drop=True)
    valid = work.notna().all(axis=1).to_numpy()
    for c in cols:
        if c in numeric_columns(work):
            valid &= np.isfinite(work[c].to_numpy(dtype=float, na_value=np.nan))
    if claim.claim_type == 'group_difference' and claim.comparison_groups:
        valid &= work[claim.group_variable].isin(claim.comparison_groups).to_numpy()
    return work.loc[valid].copy(), (np.flatnonzero(valid)+1).tolist(), (np.flatnonzero(~valid)+1).tolist()


def group_arrays(used, claim):
    labels = claim.comparison_groups or list(pd.unique(used[claim.group_variable]))
    groups = [used.loc[used[claim.group_variable] == label, claim.outcome].to_numpy(dtype=float) for label in labels]
    return labels, groups


def audit_method(frame, claim, alpha=.05):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise DataValidationError('α 须在 (0,1)。')
    used, rows, excluded = prepare_analysis(frame, claim)
    alerts, errors, details = [], [], {}
    numeric = numeric_columns(frame)
    roles = {c: ('numeric' if c in numeric else 'categorical') for c in claim.columns()}
    if claim.group_variable:
        roles[claim.group_variable] = 'categorical (user specified)'
    if claim.claim_type == 'categorical_association':
        roles = {c:'categorical (user specified)' for c in claim.columns()}
    if excluded:
        alerts.append(f'{len(excluded)} 条记录不满足当前变量完整有限值/指定组条件；不插补。')
    if len(used) < 30:
        alerts.append('有效样本较少，检验近似及效应估计可能不稳定。')
    continuous = [c for c in claim.columns() if roles[c] == 'numeric']
    constant, near, extreme = [], [], []
    for c in continuous:
        v = used[c].to_numpy(dtype=float)
        if len(v) == 0 or np.ptp(v) == 0:
            constant.append(c)
        elif np.linalg.norm(v-v.mean()) < np.finfo(float).eps**.75 * abs(v.mean()):
            near.append(c)
        if len(v):
            q1,q3 = np.quantile(v,[.25,.75]); span=q3-q1
            if np.any((v<q1-1.5*span)|(v>q3+1.5*span)):
                extreme.append(c)
    if constant or near:
        errors.append('常量/近常量变量无法可靠估计：' + ', '.join(constant+near))
    if extreme:
        alerts.append('发现 IQR 潜在极端值：' + ', '.join(extreme) + '；建议查看质量敏感性路径。')
    details.update(constant_variables=constant, near_constant_variables=near, iqr_variables=extreme)
    kind = claim.claim_type
    if kind == 'correlation':
        method, alternatives = 'pearson', ['spearman']
        reason = '两个数值变量的方向相关主张；Pearson 估计线性相关，Spearman 提供秩相关备选。'
        alerts.append('请结合散点图检查非线性/极端值；未自动证实线性或观测独立性。不因单一正态性检查机械禁止 Pearson。')
        if len(used)<3:
            errors.append('相关分析至少需要3条有效观测。')
    elif kind == 'group_difference':
        labels, groups = group_arrays(used,claim)
        details['groups'] = [{'label':str(l),'n':len(g),'mean':float(g.mean()) if len(g) else None,
                              'variance':float(np.var(g,ddof=1)) if len(g)>1 else None} for l,g in zip(labels,groups)]
        k = len(groups)
        if k<2 or any(len(g)<2 for g in groups):
            errors.append('至少两组，且每组至少2条有效观测；不能悄悄省略空组。')
        if k>2 and claim.direction != 'different':
            errors.append('总体多组检验只能判断至少一组不同，不能核验指定方向。请指定两组。')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lev = stats.levene(*groups, center='median') if k>=2 and all(len(g)>1 for g in groups) else None
        lev_p = float(lev.pvalue) if lev is not None and np.isfinite(lev.pvalue) else None
        details['levene'] = {'method':'median-centered Levene', 'statistic':float(lev.statistic) if lev is not None and np.isfinite(lev.statistic) else None,
                             'p_value':lev_p, 'threshold':alpha}
        if k==2:
            method = 'welch_t' if lev_p is None or lev_p<alpha else 'student_t'
            alternatives = ['welch_t'] if method=='student_t' else ['student_t']
            reason = ('Levene 未达到显著，按规则推荐 Student；这不证明方差相等，也可主动选择 Welch。' if method=='student_t'
                      else 'Levene 提示方差不齐或无法估计，推荐不要求等方差的 Welch。')
        else:
            method,alternatives = 'anova',['kruskal']
            reason = '分类分组 × 数值结果，检验各组总体均值是否相同；总体显著不指定哪两组不同。'
        if lev_p is None or lev_p<alpha:
            alerts.append('方差齐性条件不理想或无法判断；ANOVA 谨慎解释。Kruskal 检验秩/分布，不能简单称为均值差异检验。')
        alerts.append('独立样本设计需要研究者确认；小样本、偏态和极端值可能影响参数检验。')
    elif kind == 'categorical_association':
        x,y = claim.variables
        if used[x].nunique()*used[y].nunique()>10000:
            raise DataValidationError('列联表超过10000格；请核对是否误选ID/连续变量。')
        table=pd.crosstab(used[x],used[y])
        method,alternatives='chi_square',[]
        reason='两个用户指定的分类变量，以列联表检验独立性。'
        if min(table.shape, default=0)<2:
            errors.append('两个分类变量均至少需要两个有效水平。')
        elif table.size>10000:
            errors.append('列联表超过10000格；请核对是否误选ID/连续变量。')
        else:
            expected=stats.contingency.expected_freq(table.to_numpy())
            details['expected_counts']={'minimum':float(expected.min()),'fraction_below_5':float((expected<5).mean())}
            if (expected<1).any() or (expected<5).mean()>.2:
                alerts.append('期望频数过低，χ²渐近 p 值需谨慎解释；2×2 可选择 Fisher。')
            if table.shape==(2,2):
                alternatives=['fisher']
        details['levels'] = {x:[str(v) for v in table.index], y:[str(v) for v in table.columns]}
    else:
        method,alternatives='ols',[]
        reason='多个数值预测变量及数值结果，带截距 OLS；核验指定预测变量在控制其余变量后的系数方向。'
        p=len(claim.predictors)
        if len(used)<=p+1:
            errors.append('OLS 有效样本须多于含截距的参数数量。')
        elif not constant and not near:
            design=sm.add_constant(used[claim.predictors].astype(float), has_constant='add')
            if np.linalg.matrix_rank(design)<design.shape[1]:
                errors.append('设计矩阵秩不足：完全共线，无法可靠核验单个系数。')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                vifs={c:float(variance_inflation_factor(design.to_numpy(),i+1)) for i,c in enumerate(claim.predictors)}
            details['vif']={c:v if np.isfinite(v) else None for c,v in vifs.items()}
            details['vif_infinite']=[c for c,v in vifs.items() if not np.isfinite(v)]
            if any(v>5 or not np.isfinite(v) for v in vifs.values()):
                alerts.append('VIF > 5 或无限，存在明显共线性风险；阈值是提示而非通用合格标准。')
            if len(used)<10*(p+1):
                alerts.append('相对于参数数量样本较少；10倍参数仅为经验提示。')
        alerts.append('常规 OLS 标准误假设独立、同方差残差；未自动验证模型形式或遗漏混杂。预测/关联模型结果不能自动证明因果关系。')
    return dict(variable_types=roles, recommended_method=method, alternative_methods=alternatives,
                selection_reason=reason, assumptions=['独立观测；变量语义与研究设计由用户确认。', '双侧检验（F/χ²为相应上尾总体检验）；p < α。'],
                warnings=alerts, errors=errors, details=details, sample_size=len(used),
                used_row_ids=rows, excluded_row_ids=excluded, alpha=float(alpha))


def audit_method_v4(frame, claim, alpha=.05):
    """V4 policy: Welch by default. Keep V3 API semantics for archived workflows/tests."""
    audit=audit_method(frame,claim,alpha)
    audit['selection_policy']='v4_welch_default'
    if claim.claim_type=='group_difference' and len(audit['details'].get('groups',[]))==2:
        audit['recommended_method']='welch_t'
        audit['alternative_methods']=['student_t']
        audit['selection_reason']='两组独立样本比较默认推荐 Welch；Levene 仅作为方差诊断信息，不用于决定检验方法。Student 可由用户明确选择。'
        audit['details']['levene']['role']='diagnostic_only'
    return audit
