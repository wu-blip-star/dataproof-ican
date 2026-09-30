"""Read-only, row-traceable quality evidence; flags never establish fabrication."""
import operator

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.data import DataValidationError, numeric_columns


def audit_quality(frame, analysis_variables, id_column=None, anomaly_variables=None,
                  survey=None, high_missing_rate=.2, random_state=20260929):
    if frame.empty:
        raise DataValidationError("没有可审计的记录。")
    if not 0 < high_missing_rate <= 1:
        raise DataValidationError("高缺失阈值须在 (0,1]。")
    if id_column is not None and id_column not in frame:
        raise DataValidationError("ID 字段不存在。")
    if not analysis_variables or any(c not in frame for c in analysis_variables):
        raise DataValidationError("分析变量不存在或为空。")
    f = frame.reset_index(drop=True)
    nums = numeric_columns(f)
    def identities(rows):
        return [{"row_id": int(i), "sample_id": None if id_column is None or pd.isna(f[id_column].iloc[i-1])
                 else str(f[id_column].iloc[i-1])} for i in sorted(set(rows))]
    def issue(method, rows, variables, parameters, explanation, **extra):
        rows = sorted(set(int(i) for i in rows))
        return dict(rule=method, method=method, affected_rows=identities(rows), affected_count=len(rows),
                    variables=list(variables), parameters=parameters, explanation=explanation, **extra)
    missing = f.isna()
    pair = f[analysis_variables].notna().all(axis=1)
    for col in analysis_variables:
        if col in nums:
            pair &= np.isfinite(f[col].to_numpy(dtype=float, na_value=np.nan))
    rates = missing.mean()
    missingness = issue("isna; complete finite cases for selected variables", np.flatnonzero(missing.any(axis=1))+1,
                        f.columns, {"high_missing_rate": high_missing_rate}, "只记录缺失，不进行插补。无穷值单独报告。",
                        total_missing=int(missing.sum().sum()), variable_missing_rates={str(c): float(rates[c]) for c in f},
                        row_missing_counts=[{"row_id": i+1, "count": int(n)} for i,n in enumerate(missing.sum(axis=1))],
                        high_missing_variables=[c for c in f if rates[c] >= high_missing_rate],
                        effective_sample_size=int(pair.sum()), effective_row_ids=(np.flatnonzero(pair)+1).tolist(),
                        nonfinite_rows=identities(np.flatnonzero(np.isinf(f[nums].to_numpy(dtype=float, na_value=np.nan)).any(axis=1))+1) if nums else [])
    def groups_for(columns, exclude_missing_id=False):
        mask = f.duplicated(subset=columns, keep=False)
        if exclude_missing_id:
            mask &= f[columns[0]].notna()
        groups = []
        for _, group in f[mask].groupby(columns, dropna=False, sort=False, observed=True):
            groups.append(identities((group.index+1).tolist()))
        return groups
    exact_groups = groups_for(list(f.columns))
    id_groups = groups_for([id_column], True) if id_column else []
    dup_rows = [r['row_id'] for g in exact_groups + id_groups for r in g]
    removal = set((np.flatnonzero(f.duplicated(keep='first'))+1).tolist())
    duplicates = issue("exact row equality; ID equality when selected", dup_rows, f.columns,
                       {"id_column": id_column, "scenario_rule": "exact duplicates: keep first occurrence; duplicate ID alone is not removed"},
                       "重复ID可能代表重复测量，不能直接判为错误；情景仅排除完全重复记录的后续副本。",
                       exact_groups=exact_groups, duplicate_id_groups=id_groups,
                       exact_affected_count=sum(map(len, exact_groups)), duplicate_id_affected_count=sum(map(len,id_groups)),
                       scenario_row_ids=sorted(removal))
    flagged, per_variable = set(), {}
    for col in nums:
        if col == id_column:
            continue
        a = f[col].to_numpy(dtype=float, na_value=np.nan)
        values = a[np.isfinite(a)]
        if not len(values):
            continue
        q1,q3 = np.quantile(values, [.25,.75], method='linear')
        low,high = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
        rows = (np.flatnonzero(np.isfinite(a) & ((a<low)|(a>high)))+1).tolist()
        flagged.update(rows)
        per_variable[col] = dict(q1=float(q1), q3=float(q3), lower=float(low), upper=float(high),
                                 affected_rows=identities(rows), affected_count=len(rows))
    outliers = issue("IQR: < Q1 - 1.5 IQR or > Q3 + 1.5 IQR", flagged, per_variable,
                     {"factor": 1.5, "quantile": "linear", "excluded_id": id_column},
                     "潜在极端值仅为分布提示，不等于错误数据；IQR=0 时使用相同严格边界。", per_variable=per_variable)
    anomaly_variables = list(dict.fromkeys(anomaly_variables or []))
    if any(c not in nums or c == id_column for c in anomaly_variables):
        raise DataValidationError("多变量异常检测仅接受用户选定的数值非ID字段。")
    suitable = [c for c in anomaly_variables if f[c].replace([np.inf,-np.inf],np.nan).nunique() > 1]
    scores, anomaly_rows, available = [], [], False
    skipped = [c for c in anomaly_variables if c not in suitable]
    if len(suitable) >= 2:
        values = f[suitable].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(values).all(axis=1)
        if valid.sum() >= 10:
            estimator = IsolationForest(n_estimators=100, contamination='auto', random_state=random_state, n_jobs=1)
            estimator.fit(values[valid])
            values_score = -estimator.score_samples(values[valid])
            labels = estimator.predict(values[valid])
            row_ids = np.flatnonzero(valid)+1
            anomaly_rows = row_ids[labels==-1].tolist()
            for rank, j in enumerate(np.argsort(-values_score, kind='stable'), 1):
                scores.append({**identities([row_ids[j]])[0], "score": float(values_score[j]),
                               "rank": rank, "flagged": bool(labels[j] == -1)})
            available = True
    multi = issue("Isolation Forest", anomaly_rows, suitable,
                  {"random_state": random_state, "n_estimators":100, "contamination":"auto", "score_threshold":.5,
                   "minimum_complete_rows":10, "skipped_constant_variables":skipped},
                  "仅为异常提示，不判断真假或造假；不自动删除。分数越高越异常，>0.5 标记；只使用完整有限样本。",
                  available=available, scores=scores, reason=None if available else "需主动选择至少两个非恒定数值字段及至少10条完整记录。")
    survey = survey or {}
    checks = {}
    duration = survey.get('duration_column')
    if duration:
        if duration not in nums or duration == id_column:
            raise DataValidationError("作答时长必须是数值非ID字段。")
        threshold = float(survey.get('duration_threshold', 10))
        if not np.isfinite(threshold) or threshold <= 0:
            raise DataValidationError("时长阈值必须为正数。")
        a = f[duration].to_numpy(dtype=float, na_value=np.nan)
        checks['short_duration'] = issue("duration < threshold", np.flatnonzero(np.isfinite(a)&(a<threshold))+1,
                                        [duration], {"threshold":threshold, "unit":"原字段单位"}, "短时长仅供复核。")
    likert = list(dict.fromkeys(survey.get('likert_columns', [])))
    if likert:
        if len(likert) < 3 or any(c not in nums or c==id_column for c in likert):
            raise DataValidationError("连续同选需至少3个数值非ID题目；请确认编码在同一量表范围内。")
        a = f[likert].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(a).all(axis=1)
        checks['straight_lining'] = issue("all selected items identical", np.flatnonzero(valid & (np.ptp(a,axis=1)==0))+1,
                                         likert, {"requires_complete_response":True}, "全题连续同选可能是真实态度，不是低质量定论。")
        if survey.get('similarity', False):
            if valid.sum() > 2000:
                raise DataValidationError("相似回答全配对上限为2000条完整记录，请取消该项或缩小数据。")
            threshold = float(survey.get('similarity_threshold', 1.0))
            if not 0 < threshold <= 1:
                raise DataValidationError("回答一致率阈值须在 (0,1]。")
            ids = np.flatnonzero(valid)
            pairs, similar = [], set()
            for offset, i in enumerate(ids):
                later = ids[offset+1:]
                similarities = np.mean(a[later] == a[i], axis=1)
                for j,score in zip(later[similarities>=threshold],similarities[similarities>=threshold]):
                    pairs.append({"row_a":int(i+1),"row_b":int(j+1),"similarity":float(score)})
                    similar.update([int(i+1),int(j+1)])
                    if len(pairs)>20000:
                        raise DataValidationError("高度相似配对超过20000对，记录过大；请提高一致率阈值或缩小数据。")
            checks['similar_answers'] = issue("fraction of identical selected answers", similar, likert,
                                              {"threshold":threshold}, "题目级一致率是透明相似度，不能据此判断造假。", pairs=pairs)
    ops = {'<':operator.lt, '<=':operator.le, '>':operator.gt, '>=':operator.ge, '==':operator.eq, '!=':operator.ne}
    logic_flags, rule_details = set(), []
    for rule in survey.get('logic_rules', []):
        left, right, op = rule.get('left'),rule.get('right'),rule.get('operator')
        if left not in nums or right not in nums or op not in ops:
            raise DataValidationError("逻辑规则需两个数值字段和支持的比较运算符。")
        a,b = f[left].to_numpy(dtype=float, na_value=np.nan),f[right].to_numpy(dtype=float, na_value=np.nan)
        rows = (np.flatnonzero(np.isfinite(a)&np.isfinite(b)&ops[op](a,b))+1).tolist()
        logic_flags.update(rows)
        rule_details.append({**rule, 'affected_rows':identities(rows)})
    if survey.get('logic_rules'):
        checks['logic_conflicts'] = issue("user-defined comparisons (no eval)", logic_flags,
                                         sorted({r[k] for r in survey['logic_rules'] for k in ('left','right')}),
                                         {'rules':survey['logic_rules']}, "仅依照用户明确定义的冲突条件进行提示。", rules=rule_details)
    survey_rows = {r['row_id'] for c in checks.values() for r in c['affected_rows']}
    survey_result = issue("explicit opt-in checks", survey_rows, sorted({v for c in checks.values() for v in c['variables']}),
                          survey, "未配置的调查检查不执行；各标记只作为复核线索。", checks=checks, available=bool(checks))
    return dict(missingness=missingness, duplicates=duplicates, outliers=outliers,
                multivariate_anomalies=multi, survey_quality=survey_result)
