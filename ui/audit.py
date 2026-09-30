"""V3 audit workspace, alongside the unchanged V1/V2 interaction path."""
import json
from html import escape

import pandas as pd
import streamlit as st

from core.claim_types import StructuredClaim
from core.data import DataValidationError, numeric_columns
from core.evidence import verify_research_claim
from core.method_audit import audit_method, METHOD_NAMES
from core.quality import audit_quality
from core.research_audit import RULE_LABELS
from core.charts import scatter_with_trend
from ui.stability import render_stability

KINDS={'correlation':'相关关系 · 数值 × 数值', 'group_difference':'组间差异 · 分类 × 数值',
       'categorical_association':'分类变量关联 · 分类 × 分类', 'regression_association':'控制其他变量后的回归关联'}
QUALITY_LABELS={'missingness':'缺失数据','duplicates':'重复记录','outliers':'潜在极端值',
                'multivariate_anomalies':'多变量异常提示','survey_quality':'可选调查质量'}


def render_quality(quality):
    cols=st.columns(5)
    for col,(key,label) in zip(cols,QUALITY_LABELS.items()):
        col.metric(label,quality[key]['affected_count'])
    st.caption('上方为受影响记录数，类别间可重叠，不可直接相加。标记不等于错误或造假。')
    missing=quality['missingness']
    st.write(f"全数据缺失单元格：{missing['total_missing']}；当前分析变量有效样本：{missing['effective_sample_size']}。")
    if missing['high_missing_variables']:
        st.warning('高缺失变量：'+', '.join(missing['high_missing_variables']))
    st.dataframe(pd.DataFrame([{'变量':k,'缺失率':v} for k,v in missing['variable_missing_rates'].items()]),hide_index=True,width='stretch')
    for key,label in QUALITY_LABELS.items():
        with st.expander(label+' · 规则、记录号 / ID 与明细'):
            st.write(quality[key]['explanation'])
            st.json(quality[key],expanded=False)
    if not quality['multivariate_anomalies']['available']:
        st.caption(quality['multivariate_anomalies']['reason'])


def render_audit(dataset):
    frame=dataset.frame
    numeric=numeric_columns(frame)
    if st.session_state.get('v3_source')!=dataset.source_id:
        for key in list(st.session_state):
            if key.startswith('v3_'):
                del st.session_state[key]
        st.session_state.v3_source=dataset.source_id
    is_d=dataset.file_name=='quality_flip_demo.csv'
    is_e=dataset.file_name=='method_audit_demo.csv'
    st.subheader('02 / 研究可信性审计引擎 · V3')
    st.caption('结构化研究主张 → 方法审计 → 真实计算 → 数据质量敏感性 → 同一 Evidence Chain')
    kind=st.selectbox('研究问题',list(KINDS),index=1 if is_e else 0,format_func=KINDS.get,key='v3_kind')
    if kind!='categorical_association' and not numeric:
        st.warning('该研究问题至少需要一个数值变量。分类关联可使用纯分类数据。')
        return
    cols=list(frame.columns)
    id_default=next((c for c in ('sample_id','student_id','id') if c in cols),None)
    candidate=[c for c in numeric if c!=id_default] or numeric
    if kind=='correlation':
        if len(candidate)<2:
            st.warning('相关分析需要两个数值变量。')
            return
        x=st.selectbox('分析变量 X',candidate,key='v3_x')
        y=st.selectbox('分析变量 Y',candidate,index=1,key='v3_y')
        direction=st.selectbox('结构化方向',['positive','negative'],format_func=lambda v:'正相关' if v=='positive' else '负相关',key='v3_corr_direction')
        claim=StructuredClaim(kind,[x,y],y,direction=direction)
    elif kind=='group_difference':
        group=st.selectbox('分组变量（包括数值编码组别）',cols,index=cols.index('teaching_mode') if is_e else 0,key='v3_group')
        y=st.selectbox('数值结果变量',candidate,index=candidate.index('exam_score') if 'exam_score' in candidate else 0,key='v3_group_y')
        groups=list(frame[group].dropna().unique())
        st.caption('用户指定分组语义；数值编码不会自动当作连续变量。方向统一为第一组减第二组。')
        if len(groups)<2:
            st.warning('分组变量至少需要两个水平。')
            return
        mode=st.radio('比较范围',['指定两组','全部组总体比较'],horizontal=True,key='v3_group_mode')
        if mode=='指定两组':
            a=st.selectbox('第一组 A',groups,key='v3_group_a')
            b=st.selectbox('第二组 B',groups,index=1,key='v3_group_b')
            direction=st.selectbox('差异主张',['positive','negative','different'],format_func=lambda v:{'positive':'A组高于B组','negative':'A组低于B组','different':'两组有差异'}[v],key='v3_group_direction')
            comparison=[a,b]
        else:
            direction,comparison='different',groups
        claim=StructuredClaim(kind,[group,y],y,group_variable=group,comparison_groups=comparison,direction=direction)
    elif kind=='categorical_association':
        if len(cols)<2:
            st.warning('需要两个分类变量。')
            return
        x=st.selectbox('分类变量 X',cols,key='v3_cat_x')
        y=st.selectbox('分类变量 Y',cols,index=1,key='v3_cat_y')
        claim=StructuredClaim(kind,[x,y],y,direction='associated')
    else:
        y=st.selectbox('回归结果 Y',candidate,index=candidate.index('exam_score') if 'exam_score' in candidate else 0,key='v3_reg_y')
        options=[c for c in candidate if c!=y]
        preferred=[c for c in ('study_hours','sleep_hours','short_video_hours') if c in options]
        predictors=st.multiselect('预测变量（含待核验变量及控制变量）',options,default=preferred or options[:2],key='v3_predictors')
        if not predictors:
            st.info('请选择预测变量。')
            return
        focus=st.selectbox('本次核验的系数变量',predictors,key='v3_focus')
        direction=st.selectbox('控制其他变量后的方向',['positive','negative'],format_func=lambda v:'正关联' if v=='positive' else '负关联',key='v3_reg_direction')
        claim=StructuredClaim(kind,[focus],y,predictors=predictors,direction=direction)
        st.info('预测/关联模型结果不能自动证明因果关系。')
    claim.original_text=st.text_area('原始主张（仅存档，V3 不解析自然语言）',key='v3_text',max_chars=2000)
    alpha=st.number_input('审计显著性水平 α',min_value=.001,max_value=.2,value=.05,step=.01,key='v3_alpha')
    st.subheader('03 / 数据质量审计')
    with st.expander('配置质量规则（只标记，不改动数据）',expanded=True):
        id_col=st.selectbox('样本 ID 字段（可选）',[None]+cols,index=cols.index(id_default)+1 if id_default else 0,format_func=lambda v:v or '不使用ID，只保留原始记录号',key='v3_id')
        options=[c for c in numeric if c!=id_col]
        anomaly=st.multiselect('Isolation Forest 数值字段（主动选择后运行）',options,key='v3_anomaly')
        high_missing=st.number_input('高缺失提醒阈值',min_value=.01,max_value=1.,value=.2,step=.05,key='v3_missing')
        survey={}
        if st.checkbox('启用用户指定的调查质量检查',key='v3_survey'):
            duration=st.selectbox('作答时长字段',[None]+options,format_func=lambda v:v or '不检测作答时长',key='v3_duration')
            if duration:
                survey['duration_column']=duration
                survey['duration_threshold']=st.number_input('极短时长阈值（原字段单位）',min_value=.01,value=10.,key='v3_duration_threshold')
            likert=st.multiselect('Likert题目字段（至少3题，同一量表编码）',options,key='v3_likert')
            survey['likert_columns']=likert
            if likert:
                survey['similarity']=st.checkbox('检测高度相似回答（最多2000条完整回答）',key='v3_similar')
                if survey['similarity']:
                    survey['similarity_threshold']=st.number_input('题目级答案一致率阈值',min_value=.01,max_value=1.,value=1.,key='v3_similarity_threshold')
            count=st.number_input('自定义潜在逻辑冲突规则数量',min_value=0,max_value=5,value=0,key='v3_logic_count')
            rules=[]
            for i in range(count):
                a,b,c=st.columns(3)
                left=a.selectbox(f'规则{i+1} 左字段',options,key=f'v3_logic_left_{i}')
                op=b.selectbox(f'规则{i+1} 冲突条件',['<','<=','>','>=','==','!='],key=f'v3_logic_op_{i}')
                right=c.selectbox(f'规则{i+1} 右字段',options,key=f'v3_logic_right_{i}')
                rules.append({'left':left,'operator':op,'right':right})
            survey['logic_rules']=rules
    config={'id_column':id_col,'anomaly_variables':anomaly,'survey':survey,'high_missing_rate':high_missing}
    quality_sig=json.dumps([dataset.source_id,claim.to_dict(),config],ensure_ascii=False,sort_keys=True,default=str)
    if st.session_state.get('v3_quality_sig')!=quality_sig:
        st.session_state.pop('v3_quality',None)
    if st.button('运行数据质量审计',key='v3_audit_quality'):
        try:
            st.session_state.v3_quality=audit_quality(frame,claim.columns(),**config)
            st.session_state.v3_quality_sig=quality_sig
        except DataValidationError as exc:
            st.error(str(exc))
    if 'v3_quality' in st.session_state:
        render_quality(st.session_state.v3_quality)
    st.subheader('04 / 方法审计')
    try:
        audit=audit_method(frame,claim,alpha)
    except DataValidationError as exc:
        st.warning(str(exc))
        return
    st.write('**推荐统计方法：** '+METHOD_NAMES[audit['recommended_method']])
    st.write(audit['selection_reason'])
    st.caption('假设条件：'+' '.join(audit['assumptions']))
    st.caption('备选方法：'+('、'.join(METHOD_NAMES[m] for m in audit['alternative_methods']) or '暂无'))
    for message in audit['warnings']:
        st.warning(message)
    for message in audit['errors']:
        st.error(message)
    with st.expander('方法诊断明细（样本量 / Levene / VIF / 期望频数）'):
        st.json(audit,expanded=False)
    choices=[audit['recommended_method']]+audit['alternative_methods']
    if st.session_state.get('v3_method') not in choices:
        st.session_state.pop('v3_method',None)
    method=st.selectbox('本次实际执行的方法',choices,format_func=METHOD_NAMES.get,key='v3_method')
    if method=='kruskal':
        st.info('此备选路径将核验秩/分布差异，不直接核验均值差异。')
    selected=st.multiselect('质量敏感性场景：主动选择暂时排除规则',list(RULE_LABELS),default=['outliers'] if is_d else [],format_func=RULE_LABELS.get,key='v3_rules')
    st.caption('原始结果始终保留；按相同统计方法重新计算。只对所选规则逐项消融。IQR扫描数值非ID列；多变量异常不自动作为删除依据。')
    threshold=st.number_input('描述性效应量变化阈值（绝对差）',min_value=.001,value=.1,step=.05,key='v3_effect_threshold')
    stability=False
    if method=='pearson':
        stability=st.checkbox('同时执行现有 V2 四类证据稳定性分析',key='v3_stability')
    sig=(quality_sig,method,alpha,tuple(selected),threshold,stability)
    if st.session_state.get('v3_signature')!=sig:
        st.session_state.pop('v3_evidence',None)
    if st.button('执行证据核验',type='primary',disabled=bool(audit['errors']),key='v3_verify'):
        try:
            with st.spinner('真实计算统计结果、质量场景与消融分析…'):
                node=verify_research_claim(dataset,claim,method,alpha,config,selected,threshold,stability)
            st.session_state.v3_evidence=node
            st.session_state.v3_signature=sig
            st.session_state.v3_quality=node.quality_evidence
            st.session_state.v3_quality_sig=quality_sig
            st.rerun()
        except DataValidationError as exc:
            st.error(str(exc))
    node=st.session_state.get('v3_evidence')
    if node is None:
        return
    result=node.statistical_result
    st.subheader('05 / 真实统计结果')
    (st.success if node.evidence_status=='Supported' else st.warning)(node.evidence_status+' · '+node.judgment_reason)
    for col,label,val in zip(st.columns(4),['有效样本 n','统计量','p-value',result['effect_size']['name']],
                             [result['sample_size'],result['statistic'],result['p_value'],result['effect_size']['value']]):
        col.metric(label, '未定义' if val is None else f'{val:.6g}')
    st.write(node.suggested_wording)
    st.info('预测/关联模型结果不能自动证明因果关系。不显著不能证明没有关系，质量标记不证明数据造假。')
    if kind=='correlation':
        sample=frame.iloc[[i-1 for i in result['used_row_ids']]][claim.variables]
        st.plotly_chart(scatter_with_trend(sample,*claim.variables),width='stretch')
        if method=='spearman':
            st.caption('散点趋势线为描述性 OLS；秩相关系数与其检验由 Spearman 独立计算。')
    if 'coefficients' in result['details']:
        st.dataframe(pd.DataFrame(result['details']['coefficients']),hide_index=True,width='stretch')
    if 'group_statistics' in result['details']:
        st.dataframe(pd.DataFrame(result['details']['group_statistics']),hide_index=True,width='stretch')
    if 'contingency_table' in result['details']:
        st.write('观测列联表')
        st.dataframe(pd.DataFrame(result['details']['contingency_table'],index=result['details']['row_labels'],columns=result['details']['column_labels']))
        st.write('期望频数')
        st.dataframe(pd.DataFrame(result['details']['expected_frequencies'],index=result['details']['row_labels'],columns=result['details']['column_labels']))
    with st.expander('完整统计结果（均值差、95%区间、df、R²等）'):
        st.json(result,expanded=False)
    st.subheader('06 / 数据质量对结论的影响')
    sensitivity=node.quality_sensitivity
    scenario=sensitivity['quality_sensitive']
    def summary(label,res):
        return {'路径':label,'n':res['sample_size'],'statistic':res['statistic'],
                'effect size':res['effect_size']['value'] if res['effect_size'] else None,
                'p-value':res['p_value'],'direction':res['direction'],'claim status':res['claim_status']}
    st.dataframe(pd.DataFrame([summary('Original',result),summary('Quality-sensitive',scenario['result'])]),hide_index=True,width='stretch')
    if scenario['changes']['claim_flip']:
        st.warning(scenario['changes']['explanation'])
    elif not scenario['result']['available']:
        st.warning(scenario['result']['reason']+'；不能判断翻转。')
    else:
        st.info(scenario['changes']['explanation'])
    st.write(f"从原始有效样本中暂时排除 {scenario['removed_n']} 条；未修改原始数据。")
    st.json(scenario['changes'],expanded=False)
    if sensitivity['ablation']:
        st.write('**Quality Issue Ablation / 数据质量问题消融分析**')
        table=[]
        for row in sensitivity['ablation']:
            table.append({**summary(row['label'],row['result']),'Removed n':row['removed_n'],'Claim changed?':row['changes']['claim_status_changed']})
        st.dataframe(pd.DataFrame(table),hide_index=True,width='stretch')
        largest=sensitivity['largest_effect_change_issue']
        if largest:
            st.caption('所选路径中绝对效应变化最大：'+RULE_LABELS[largest]+'。该类样本处理与结论变化存在敏感性，不构成因果归因。')
    else:
        st.caption('尚未选择排除规则，当前对照与原始路径相同；不会自动执行删样本场景。')
    st.subheader('07 / D2C Evidence Chain · V3')
    card_values={
        'raw_data':f"{dataset.file_name} · {len(frame)} 条原始记录\nSHA-256: {dataset.sha256}",
        'quality':' / '.join(f"{QUALITY_LABELS[k]}：{node.quality_evidence[k]['affected_count']} 条" for k in QUALITY_LABELS),
        'variables':f"研究问题：{KINDS[kind]}\n分析变量：{', '.join(claim.columns())}；结果：{claim.outcome}",
        'sample':f"有效 n = {node.sample_size}；未纳入 {len(node.sample['excluded_row_ids'])} 条\n完整记录号与 ID 均保存在本证据节点中。",
        'method_audit':f"推荐：{METHOD_NAMES[node.method_audit['recommended_method']]}\n{node.method_audit['selection_reason']}",
        'method':f"实际执行：{node.method_name}；α = {node.alpha:g}；p < α\n方向检验采用双侧 p；F/χ² 采用上尾总体检验；可用的均值差/系数区间固定95%。",
        'result':f"statistic = {result['statistic']}；p = {result['p_value']:.6g}\n{result['effect_size']['name']} = {result['effect_size']['value']:.4f}",
        'stability':'已执行 V2：Spearman / Bootstrap / Jackknife / IQR；画像见下方。' if node.stability_analysis else '未执行；V2 Pearson 稳定性分析入口保留。',
        'claim':f"结构化主张：{kind} / {claim.direction}\n原始文本仅存档：{claim.original_text or '未填写'}",
        'quality_sensitivity':f"暂时排除 {scenario['removed_n']} 条原始有效记录；{scenario['result']['claim_status'] or '无法估计'}\n{scenario['changes']['explanation']}",
        'judgment':node.evidence_status+' · '+node.judgment_reason,
        'wording':node.suggested_wording,
    }
    steps=''.join(f'<div class="chain-step"><div class="step-number">{i:02}</div><div class="step-label"><strong>{escape(s["label"])}</strong></div><div class="step-value">{escape(card_values[s["node_type"]])}</div></div>' for i,s in enumerate(node.chain,1))
    st.markdown('<div class="evidence-card-v3 evidence-card"><div class="card-head"><b>Research Credibility Audit / 研究可信性证据链</b></div>'+steps+'</div>',unsafe_allow_html=True)
    st.download_button('导出 V3 完整证据链 JSON',node.to_json(),f'evidence_v3_{node.evidence_id}.json','application/json',key='v3_download',on_click='ignore')
    with st.expander('完整 EvidenceNode / 质量规则 / 受影响记录 / 情景追溯'):
        st.json(node.to_dict(),expanded=False)
    if node.stability_analysis:
        render_stability(node)
    st.caption('V4 扩展点保留：original_text、parsed_claim、结构化 Claim、节点 ID 与 edges。本版本不调用 LLM。')
