"""V4 language -> explicit mapping confirmation -> existing executable audit."""
from datetime import datetime, timezone
import json

import streamlit as st
import streamlit.components.v1 as components

from core.claim_parser import (LLMConfig, OpenAICompatibleProvider, ParsingRecord, ClaimParserError,
    parse_claim, dataset_schema, fingerprint, causal_terms, confirm_mapping, proposal_to_claim, mapped_wording, CAUSAL_WARNING)
from core.parser_demo import OfflineDemoProvider, DEMO_TEXTS, response_for
from core.claim_types import StructuredClaim
from core.data import DataValidationError, numeric_columns
from core.method_audit import audit_method_v4, METHOD_NAMES
from core.v4_evidence import verify_parsed_claim
from core.evidence_graph import graph_html
from core.audit_report import export_html_report
from core.quality import audit_quality
from core.research_audit import RULE_LABELS
from ui.audit import KINDS, render_quality
from ui.stability import render_stability


def mapping_editor(frame,proposal,token):
    def choose(label,options,suggested,key):
        options=[None]+list(options)
        return st.selectbox(label,options,index=options.index(suggested) if suggested in options else 0,
                            format_func=lambda v:'请选择 / 待明确' if v is None else str(v),key=f'v4_map_{token}_{key}')
    kind=st.selectbox('确认研究问题类型',list(KINDS),index=list(KINDS).index(proposal['claim_type']) if proposal['claim_type'] in KINDS else 0,
                      format_func=KINDS.get,key=f'v4_map_{token}_kind')
    token=token+'_'+kind
    cols=list(frame.columns);nums=numeric_columns(frame)
    x_hint=proposal['variables'][0] if proposal['variables'] else None
    y=choose('确认结果变量 Y',cols if kind=='categorical_association' else nums,proposal['outcome'],'y')
    predictors=[];group=None;groups=[]
    if kind=='group_difference':
        group=choose('确认分组变量',cols,proposal['group_variable'],'group')
        levels=list(frame[group].dropna().unique()) if group else []
        scope=st.radio('确认比较范围',['指定两组','所有组总体比较'],index=1 if proposal['direction']=='different' and not proposal['comparison_groups'] else 0,
                       horizontal=True,key=f'v4_map_{token}_scope')
        if scope=='指定两组':
            labels=proposal['comparison_groups']
            def level(i):
                return next((v for v in levels if len(labels)>i and str(v)==labels[i]),None)
            a=choose('确认第一组 A',levels,level(0),'a');b=choose('确认第二组 B',levels,level(1),'b')
            groups=[a,b]
            direction=choose('确认差异方向',['positive','negative','different'],proposal['direction'],'direction')
            st.caption('positive = A高于B；negative = A低于B；different = 两组有差异。')
        else:
            groups=levels;direction='different'
        variables=[group,y]
    elif kind=='regression_association':
        predictors=st.multiselect('确认预测变量及控制变量',[c for c in nums if c!=y],
                                   default=[c for c in proposal['predictors'] if c in nums and c!=y],key=f'v4_map_{token}_predictors')
        x=choose('确认待核验的预测变量',predictors,x_hint,'x')
        direction=choose('确认关联方向',['positive','negative'],proposal['direction'],'direction')
        variables=[x]
    else:
        x=choose('确认变量 X',cols if kind=='categorical_association' else nums,x_hint,'x')
        direction='associated' if kind=='categorical_association' else choose('确认关联方向',['positive','negative'],proposal['direction'],'direction')
        variables=[x,y]
    if y is None or direction is None or any(v is None for v in variables+groups):
        st.info('请明确所有必要变量、比较组和方向，系统不会自动猜测。')
        return None
    claim=StructuredClaim(kind,variables,y,predictors=predictors,group_variable=group,comparison_groups=groups,direction=direction)
    try:
        claim.validate(frame)
    except DataValidationError as exc:
        st.warning(str(exc));return None
    return claim


def render_v4(dataset):
    if st.session_state.get('v4_source')!=dataset.source_id:
        for key in list(st.session_state):
            if key.startswith('v4_'): del st.session_state[key]
        st.session_state.v4_source=dataset.source_id
    st.subheader('02 / AI 自然语言研究主张解析 · V4')
    st.caption('LLM 只理解语言；变量映射由你确认，统计量、p值和证据判断由本地统计引擎计算。')
    mode=st.radio('主张输入方式',['自然语言主张','结构化输入（无需 API）'],horizontal=True,key='v4_input_mode')
    record=None
    if mode=='自然语言主张':
        try:
            LLMConfig.from_env()
            default_provider=0
        except ClaimParserError:
            default_provider=1
        provider_mode=st.radio('解析来源',['API解析','离线契约演示（固定响应，非真实LLM）'],index=default_provider,key='v4_provider_mode')
        if provider_mode.startswith('离线'):
            st.warning('当前为离线契约演示：仅支持下列5句预设文本，使用人工固定响应；不代表真实LLM解析能力。统计计算仍为真实执行。')
            demo=st.selectbox('自然语言 Demo',DEMO_TEXTS,key='v4_demo')
            if st.session_state.get('v4_previous_demo')!=demo:
                st.session_state.v4_text=demo;st.session_state.v4_previous_demo=demo
        text=st.text_area('输入一句自然语言研究结论',key='v4_text',max_chars=2000,placeholder='例如：女生满意度显著高于男生。')
        parsing_signature=(dataset.source_id,provider_mode,text)
        if st.session_state.get('v4_parsing_signature')!=parsing_signature:
            for key in ('v4_record','v4_confirmed','v4_evidence'): st.session_state.pop(key,None)
        api_ready=True
        if provider_mode=='API解析':
            try:
                config=LLMConfig.from_env()
                st.caption('已读取环境配置；模型：'+config.model+'。Key不会进入页面、证据或报告。')
            except ClaimParserError as exc:
                st.info(str(exc));api_ready=False
            st.caption('点击解析仅发送这句话、列名/类型及至多20类的标签；不发送数据行、统计结果或文件内容。')
            with st.expander('查看将发送的字段描述（无原始样本）'):
                st.json(dataset_schema(dataset),expanded=False)
        if st.button('解析研究主张',key='v4_parse',disabled=not api_ready or not text.strip(),type='primary'):
            st.session_state.pop('v4_record',None);st.session_state.pop('v4_confirmed',None);st.session_state.pop('v4_evidence',None)
            try:
                with st.spinner('仅解析语言，不执行统计判断…'):
                    provider=OfflineDemoProvider() if provider_mode.startswith('离线') else OpenAICompatibleProvider()
                    st.session_state.v4_record=parse_claim(text,dataset,provider)
                st.session_state.v4_parsing_signature=parsing_signature
            except DataValidationError as exc:
                st.error(str(exc))
        record=st.session_state.get('v4_record')
        if record is None:return
        proposal=record.proposal
        st.write('**解析识别：** '+proposal['claim_type']+' / '+proposal['direction'])
        st.write('原文声称显著：'+('是' if proposal['claims_significance'] else '否')+'；包含因果语言：'+('是' if record.causal else '否'))
        st.caption('以上是语言标记，不是显著性计算结果。未声称显著的主张也将按当前α评估其统计证据。')
        with st.expander('语言解析记录（无统计结论）'):st.json(record.to_dict(),expanded=False)
        for note in proposal['ambiguities']:st.warning('解析待确认：'+note)
        if proposal['claim_type']=='unsupported' or proposal['unsupported_reason']:
            st.warning(proposal['unsupported_reason'] or '该主张不在当前支持范围，请修改原文或采用结构化输入。');return
        editor_token=fingerprint(record.to_dict())[:12]
    else:
        st.info('手动构建 StructuredClaim；不调用 LLM，仍使用 V4 的 Welch 推荐、证据图和 HTML 导出。')
        proposal=response_for(1)
        proposal['variables']=[c for c in ['study_hours','exam_score'] if c in dataset.frame]
        proposal['outcome']='exam_score' if 'exam_score' in dataset.frame else None
        editor_token='manual'
        text=st.text_area('原始主张（手动模式可选）',key='v4_manual_text',max_chars=2000)
    st.subheader('03 / 用户确认变量映射')
    claim=mapping_editor(dataset.frame,proposal,editor_token)
    if claim is None:
        st.session_state.pop('v4_confirmed',None);st.session_state.pop('v4_evidence',None);return
    if mode.startswith('结构化'):
        original=text.strip() or mapped_wording(claim)
        manual={**proposal,'claim_type':claim.claim_type,'variables':claim.variables,'outcome':claim.outcome,
                'predictors':claim.predictors,'group_variable':claim.group_variable,
                'comparison_groups':[str(v) for v in claim.comparison_groups],'direction':claim.direction,
                'claims_significance':'显著' in original,'causal_language':bool(causal_terms(original)),
                'causal_terms':causal_terms(original)}
        record=ParsingRecord(original,dataset.source_id,fingerprint(dataset_schema(dataset)),manual,
                             {'provider':'manual','model':'none','mode':'manual'},causal_terms(original),datetime.now(timezone.utc).isoformat())
    mapping_sig=fingerprint([dataset.source_id,mode,record.original_text,record.proposal,claim.to_dict()])
    if st.session_state.get('v4_mapping_signature')!=mapping_sig:
        st.session_state.pop('v4_confirmed',None);st.session_state.pop('v4_evidence',None)
    st.write('**将核验：** '+mapped_wording(claim))
    consent=st.checkbox('我已核对变量、组别顺序及方向，确认以上映射',key='v4_accept_'+mapping_sig[:12])
    downgrade=False
    if record.causal:
        st.warning(CAUSAL_WARNING)
        downgrade=st.checkbox('我同意将原主张降级为上述关联/差异表述；原始因果主张不视为已验证',key='v4_causal_'+mapping_sig[:12])
    if not consent or (record.causal and not downgrade):
        st.session_state.pop('v4_confirmed',None);st.session_state.pop('v4_evidence',None)
    if st.button('确认映射并进入证据审计',key='v4_confirm',disabled=not consent or (record.causal and not downgrade)):
        try:
            confirmed=confirm_mapping(record,dataset,claim,mapping_confirmed=consent,causal_downgrade=downgrade)
            st.session_state.v4_confirmed=confirmed;st.session_state.v4_mapping_signature=mapping_sig
        except DataValidationError as exc:st.error(str(exc))
    confirmed=st.session_state.get('v4_confirmed')
    if confirmed is None:return
    render_verification(dataset,confirmed)


def render_verification(dataset,claim):
    st.subheader('04 / 数据质量与方法审计')
    alpha=st.number_input('V4 显著性水平 α',min_value=.001,max_value=.2,value=.05,step=.01,key='v4_alpha')
    cols=list(dataset.frame.columns);numeric=numeric_columns(dataset.frame)
    id_hint=next((c for c in ('student_id','sample_id','id') if c in cols),None)
    id_col=st.selectbox('V4 样本ID字段',[None]+cols,index=cols.index(id_hint)+1 if id_hint else 0,format_func=lambda v:v or '仅记录原始行号',key='v4_id')
    anomaly=st.multiselect('V4 多变量异常检测字段',[c for c in numeric if c!=id_col],key='v4_anomaly')
    rules=st.multiselect('V4 质量敏感性排除规则',list(RULE_LABELS),format_func=RULE_LABELS.get,key='v4_rules')
    st.caption('不改动原始数据；规则未配置或没有标记时不会移除记录。调查质量规则的完整配置继续保留在V3工作区。')
    config={'id_column':id_col,'anomaly_variables':anomaly}
    try:audit=audit_method_v4(dataset.frame,claim,alpha)
    except DataValidationError as exc:st.error(str(exc));return
    st.write('**V4推荐统计方法：** '+METHOD_NAMES[audit['recommended_method']])
    st.write(audit['selection_reason'])
    for warning in audit['warnings']:st.warning(warning)
    for error in audit['errors']:st.error(error)
    with st.expander('查看方法诊断、假设和备选'):st.json(audit,expanded=False)
    options=[audit['recommended_method']]+audit['alternative_methods']
    method=st.selectbox('V4 实际统计方法',options,format_func=METHOD_NAMES.get,key='v4_method_'+fingerprint(claim.to_dict())[:12])
    stability=st.checkbox('执行 V2 证据稳定性分析（Pearson）',value=True,key='v4_stability',disabled=method!='pearson') if method=='pearson' else False
    if method!='pearson':st.caption('此方法尚未推广V2 Bootstrap/Jackknife；图中将如实标记稳定性未执行。')
    signature=fingerprint([claim.to_dict(),alpha,config,rules,method,stability])
    if st.session_state.get('v4_result_signature')!=signature:st.session_state.pop('v4_evidence',None)
    if st.button('V4 执行真实统计核验',type='primary',key='v4_verify',disabled=bool(audit['errors'])):
        try:
            with st.spinner('本地执行数据质量、统计核验与证据审计…'):
                node=verify_parsed_claim(dataset,claim,method,alpha,config,rules,stability)
            st.session_state.v4_evidence=node;st.session_state.v4_result_signature=signature
        except DataValidationError as exc:st.error(str(exc))
    node=st.session_state.get('v4_evidence')
    if node is None:return
    render_quality(node.quality_evidence)
    st.subheader('05 / 确认后主张的真实统计结果')
    if node.parsed_claim['causal_language']:st.warning('原始因果主张：未验证。'+CAUSAL_WARNING)
    st.write('**确认后主张：** '+node.claim_text)
    (st.success if node.evidence_status=='Supported' else st.warning)('确认后主张判断：'+node.evidence_status+' · '+node.judgment_reason)
    for col,label,value in zip(st.columns(3),['有效样本 n','统计量','p-value'],[str(node.sample_size),str(node.statistic),f'{node.p_value:.6g}']):col.metric(label,value)
    st.write(node.suggested_wording)
    with st.expander('真实统计结果与质量敏感性对照'):
        st.json({'statistical_result':node.statistical_result,'quality_sensitivity':node.quality_sensitivity},expanded=False)
    st.subheader('06 / D2C Evidence Graph')
    components.html(graph_html(node),height=960 if node.stability_analysis else 760,scrolling=True)
    st.caption('图中箭头表示证据依赖，不代表因果；可点击节点核对规则、解析来源和完整统计结果。')
    st.download_button('下载 V4 完整证据 JSON',node.to_json(),f'evidence_v4_{node.evidence_id}.json','application/json',key='v4_json',on_click='ignore')
    st.download_button('一键导出 HTML 可信性审计报告',export_html_report(node),f'DataProof_Audit_{node.evidence_id}.html','text/html',key='v4_html',on_click='ignore')
    if node.stability_analysis:render_stability(node)
    with st.expander('完整证据节点及解释边界'):st.json(node.to_dict(),expanded=False)
