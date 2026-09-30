"""Data Quality × Claim Impact orchestration on the existing EvidenceNode."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from uuid import uuid4

import numpy as np

from core.data import DataValidationError
from core.quality import audit_quality
from core.statistics import run_analysis
from core.method_audit import prepare_analysis, group_arrays

RULE_LABELS={'duplicates':'完全重复行的后续副本','outliers':'IQR 潜在极端值',
             'multivariate_anomalies':'Isolation Forest 异常提示','survey_quality':'用户指定的调查质量规则'}


def compare_results(original, scenario, threshold=.1):
    if not np.isfinite(threshold) or threshold<=0:
        raise DataValidationError('效应量变化阈值须为正数。')
    keys=['direction_changed','significance_changed','effect_size_changed','claim_status_changed']
    if not scenario['available']:
        return {**dict.fromkeys(keys), 'claim_flip':None, 'effect_size_delta':None,
                'effect_change_threshold':threshold, 'explanation':'敏感性场景无法估计，不能判断是否翻转。'}
    delta=scenario['effect_size']['value']-original['effect_size']['value']
    change=dict(direction_changed=original['direction']!=scenario['direction'],
                significance_changed=original['significant']!=scenario['significant'],
                effect_size_changed=abs(delta)>=threshold,
                claim_status_changed=original['claim_status']!=scenario['claim_status'])
    flip=(original['claim_status']=='Supported') != (scenario['claim_status']=='Supported')
    return {**change,'claim_flip':flip,'effect_size_delta':delta,'effect_change_threshold':threshold,
            'explanation':('研究结论在数据质量敏感性分析后发生翻转。' if flip else '当前处理路径未发生支持状态翻转。')
                          +'结果仅反映对样本处理的敏感性，不证明原研究错误；效应变化阈值是描述性规则。'}


def quality_scenarios(frame, claim, original, quality, rules, threshold=.1):
    if any(r not in RULE_LABELS for r in rules):
        raise DataValidationError('未知质量敏感性规则。')
    frozen=deepcopy(claim)
    used,_,_=prepare_analysis(frame,claim)
    if claim.claim_type=='group_difference':
        frozen.comparison_groups=group_arrays(used,claim)[0]
    original_categories = ({c:set(used[c]) for c in claim.variables}
                           if claim.claim_type=='categorical_association' else {})
    rows_by_rule={r:(quality[r]['scenario_row_ids'] if r=='duplicates'
                     else [a['row_id'] for a in quality[r]['affected_rows']]) for r in RULE_LABELS}
    def run_one(selected):
        flagged=sorted({i for r in selected for i in rows_by_rule[r]})
        removed=sorted(set(flagged)&set(original['used_row_ids']))
        keep=[i for i in range(1,len(frame)+1) if i not in set(flagged)]
        view=frame.iloc[np.array(keep,dtype=int)-1].copy().reset_index(drop=True)
        try:
            if original_categories:
                complete,_,_=prepare_analysis(view,frozen)
                if any(set(complete[c])!=levels for c,levels in original_categories.items()):
                    raise DataValidationError('处理后分类水平消失，不能作为同一假设的直接对照。')
            result=run_analysis(view,frozen,original['method'],original['alpha'])
            for key in ('used_row_ids','excluded_row_ids'):
                result[key]=[keep[i-1] for i in result[key]]
                result['method_audit'][key]=result[key].copy()
        except DataValidationError as exc:
            result={'available':False,'sample_size':len(set(original['used_row_ids'])-set(removed)),
                    'statistic':None,'effect_size':None,'p_value':None,'direction':None,'significant':None,
                    'claim_status':None,'method':original['method'],'reason':str(exc)}
        return dict(rules=list(selected), flagged_row_ids=flagged,removed_row_ids=removed,removed_n=len(removed),
                    result=result, changes=compare_results(original,result,threshold))
    chosen=list(dict.fromkeys(rules))
    scenario=run_one(chosen)
    ablation=[{'quality_issue':r, 'label':RULE_LABELS[r], **run_one([r])} for r in chosen]
    valid=[a for a in ablation if a['result']['available'] and a['removed_n']>0]
    largest=max(valid,key=lambda a:abs(a['changes']['effect_size_delta']))['quality_issue'] if valid else None
    return dict(original_result=original, quality_sensitive=scenario, ablation=ablation,
                largest_effect_change_issue=largest,
                parameters={'selected_rules':chosen,'effect_change_threshold':threshold,'same_method':original['method'],
                            'rules_fit_on_original_data':True,'original_data_modified':False},
                explanation='仅执行主动选择的敏感性规则及其逐项消融；冻结原始方法、分组顺序及阈值。不会自动挑选最小p。最大变化只描述所选路径中的绝对效应量变化，不是因果归因。')


def build_research_evidence(dataset, claim, method, alpha, quality_config, rules, threshold, include_stability):
    from core.evidence import EvidenceNode, verify_claim, add_stability_analysis, format_p, effect_size_description
    original=run_analysis(dataset.frame,claim,method,alpha)
    quality=audit_quality(dataset.frame,claim.columns(),**(quality_config or {}))
    sensitivity=quality_scenarios(dataset.frame,claim,original,quality,rules,threshold)
    node=None
    if claim.claim_type=='correlation' and original['method']=='pearson':
        node,_=verify_claim(dataset,*claim.variables,'正相关' if claim.direction=='positive' else '负相关',alpha,claim.original_text)
        if include_stability:
            node=add_stability_analysis(node,dataset)
    elif include_stability:
        raise DataValidationError('现有 V2 四类稳定性分析适用于 Pearson 方向主张；其他方法尚未推广。')
    used,rows,excluded=prepare_analysis(dataset.frame,claim)
    source={'file_name':dataset.file_name,'file_sha256':dataset.sha256,'sheet_name':dataset.sheet_name,
            'parser':dataset.parser,'total_rows':len(dataset.frame),'total_columns':len(dataset.frame.columns)}
    sample={'used_row_ids':rows,'excluded_row_ids':excluded,
            'row_id_definition':'解析后数据记录从1编号，不含表头；CSV 多行字段时不等于物理文件行号。',
            'processing':'仅按分析变量排除缺失/非有限记录，分组对比按指定组筛选；原始路径不去重、不剔除潜在极端值、不插补。',
            'used_values_sha256':sha256(used.to_csv(index=False,lineterminator='\n').encode('utf-8')).hexdigest()}
    id_col=(quality_config or {}).get('id_column')
    if id_col:
        import pandas as pd
        sample['used_samples']=[{'row_id':r,'sample_id':None if pd.isna(dataset.frame[id_col].iloc[r-1]) else str(dataset.frame[id_col].iloc[r-1])} for r in rows]
    else:
        sample['used_samples']=[{'row_id':r,'sample_id':None} for r in rows]
    stat_text='未定义（见结果说明）' if original['statistic'] is None else f"{original['statistic']:.4f}"
    relation={'correlation':'相关关系','group_difference':'组间总体差异','categorical_association':'分类变量关联','regression_association':'控制所选变量后的系数关联'}[claim.claim_type]
    target_text=(f"{claim.variables[0]} 与 {claim.outcome}" if claim.claim_type in ('correlation','categorical_association')
                 else f"{claim.variables[0]} 对 {claim.outcome}（控制其余所选预测变量）" if claim.claim_type=='regression_association'
                 else f"{claim.group_variable} 各组的 {claim.outcome}")
    observed_direction={'positive':'正方向','negative':'负方向','zero':'零','not_applicable':'无统一方向'}[original['direction']]
    wording=(f"在当前样本中，{target_text} 的{relation}检验{'达到' if original['significant'] else '未达到'}统计显著；样本估计为{observed_direction}"
             f"（{original['method_name']}，n={original['sample_size']}，statistic={stat_text}，p={format_p(original['p_value'])}）。")
    if original['method'] in ('student_t','welch_t'):
        wording+=f" 差值按 {original['details']['comparison_order'][0]} 减 {original['details']['comparison_order'][1]} 定义。"
    if original['method']=='kruskal':
        wording+=' 此处仅检验分布/秩差异，不直接核验均值差异。'
    wording+=' '+original['judgment_reason']+' 关联不能自动证明因果关系。'
    if rules:
        wording+=' '+sensitivity['quality_sensitive']['changes']['explanation']
    parameters={'implementation':original['method_name'],'alpha':alpha,'significance_rule':'p < alpha',
                'confidence_level':.95, 'library_versions':{n:version(n) for n in ('numpy','pandas','scipy','statsmodels','scikit-learn')}}
    limitations=['原始文本仅存档；核验范围由结构化字段决定，不解析文本语义。','相关不等于因果；预测/关联模型结果不能自动证明因果关系。',
                 '不显著不能证明无关系；统计显著不能代表效应强或重要。','独立性与数据真实性不能由本工具自动确认；重复探索未校正多重比较。',
                 '质量标记仅是复核线索；所有处理均是敏感性场景，不更改原始数据。']
    if node is None:
        node=EvidenceNode(schema_version='3.0',evidence_id=str(uuid4()),created_at=datetime.now(timezone.utc).isoformat(),
                          claim_text=claim.original_text or target_text,claim_type=claim.claim_type,
                          verification_scope='structured_claim_only',x_variable=claim.variables[0],y_variable=claim.outcome,
                          sample_size=original['sample_size'],method_name=original['method_name'],statistic=original['statistic'],
                          p_value=original['p_value'],direction=original['direction'],evidence_status=original['claim_status'],
                          judgment_reason=original['judgment_reason'],suggested_wording=wording,alpha=alpha,
                          significant=original['significant'],source=source,sample=sample,method_parameters=parameters,
                          limitations=limitations,chain=[],edges=[])
    node.schema_version='3.0'
    node.claim_type=claim.claim_type
    node.structured_claim=claim.to_dict()
    node.original_text=claim.original_text
    node.parsed_claim=claim.parsed_claim
    node.quality_evidence=quality
    node.method_audit=original['method_audit']
    node.quality_sensitivity=sensitivity
    node.statistical_result=original
    node.sample=sample
    node.method_parameters=parameters
    node.limitations=limitations+original['method_audit']['warnings']
    node.effect_size=original['effect_size']
    if claim.claim_type=='correlation':
        node.effect_size={**node.effect_size, **effect_size_description(original['statistic'])}
    node.suggested_wording=wording
    stability={'data_ref':'stability_analysis','available':node.stability_analysis is not None,
               'scope':'V2 Pearson: Spearman / Bootstrap / Jackknife / IQR',
               'reason':None if node.stability_analysis else '未执行；当前保留 V2 Pearson 稳定性接口，其他方法未推广。'}
    data=[('raw_data','Raw Data',source),('quality','Data Quality',quality),
          ('variables','Variable Mapping',claim.to_dict()),('sample','Used Sample',sample),
          ('method_audit','Method Audit',node.method_audit),('method','Statistical Method',parameters),
          ('result','Statistical Result',original),('stability','Evidence Stability',stability),
          ('claim','Claim',claim.to_dict()),('quality_sensitivity','Quality Sensitivity',sensitivity),
          ('judgment','Evidence Judgment',{'status':original['claim_status'],'reason':original['judgment_reason']}),
          ('wording','Suggested Wording',{'text':wording})]
    node.chain=[{'node_id':f'{node.evidence_id}:{key}','node_type':key,'label':label,'data':value} for key,label,value in data]
    node.edges=[{'from':a['node_id'],'to':b['node_id'],'relation':'next_evidence_step'} for a,b in zip(node.chain,node.chain[1:])]
    if node.stability_analysis:
        for key in ('method_sensitivity','bootstrap','jackknife','outlier_sensitivity'):
            node.edges.append({'from':f'{node.evidence_id}:stability','to':f'{node.evidence_id}:stability:{key}','relation':'has_analysis_path'})
        node.chain[7]['subnodes']=[{'node_id':f'{node.evidence_id}:stability:{key}','node_type':key,'data_ref':f'stability_analysis.{key}'}
                                 for key in ('method_sensitivity','bootstrap','jackknife','outlier_sensitivity')]
    node.to_json()  # fail early rather than expose a non-serializable audit
    return node
