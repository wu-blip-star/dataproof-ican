"""V4 calls existing executable verification, then appends parsing provenance to the same node."""
from core.claim_parser import validate_confirmation, CAUSAL_WARNING
from core.evidence import verify_research_claim
from core.method_audit import audit_method_v4


def verify_parsed_claim(dataset,claim,method=None,alpha=.05,quality_config=None,
                        sensitivity_rules=(),include_stability=False):
    validate_confirmation(claim,dataset)
    audit=audit_method_v4(dataset.frame,claim,alpha)
    selected=method or audit['recommended_method']
    node=verify_research_claim(dataset,claim,selected,alpha,quality_config,sensitivity_rules,.1,include_stability)
    node.schema_version='4.0'
    node.method_audit=audit
    node.statistical_result['method_audit']=audit
    node.method_parameters['selection_policy']='v4_welch_default'
    node.method_parameters['selected_method']=selected
    parsed=claim.parsed_claim
    causal=parsed['causal_language']
    node.claim_text=parsed['verified_wording']
    node.verification_scope='user_confirmed_association_or_difference_only'
    node.final_judgment={'verified_claim':node.claim_text,'association_status':node.evidence_status,
                         'original_claim_status':'Not Verified (Causal Claim)' if causal else 'See confirmed interpretation',
                         'causal_claim_verified':False,'causal_downgrade_confirmed':parsed['causal_downgrade_confirmed'],
                         'warning':CAUSAL_WARNING if causal else '仅核验用户确认的结构化主张；未自动核验原文全部限定。',
                         'stability_evaluated':node.stability_analysis is not None,
                         'quality_claim_flip':node.quality_sensitivity['quality_sensitive']['changes']['claim_flip']}
    if causal:
        node.suggested_wording=CAUSAL_WARNING+' 原始因果主张未得到验证。以下结果仅针对用户确认的关联降级表述：'+node.suggested_wording
    mode=parsed['provider']['mode']
    node.limitations=[item for item in node.limitations if not item.startswith('原始文本仅存档')]
    node.limitations.insert(0,'语言解析只生成候选结构；用户确认后的变量/方向决定计算范围，模型不生成统计结果。')
    if mode=='offline_contract_demo':
        node.limitations.insert(0,'离线契约演示：解析为人工固定响应，非真实LLM调用；统计结果由SciPy/statsmodels实际计算。')
    original={'node_id':f'{node.evidence_id}:original_claim','node_type':'original_claim','label':'Original Claim',
              'data':{'text':claim.original_text,'causal_language':causal,'claims_significance':parsed['proposal']['claims_significance']}}
    parsing={'node_id':f'{node.evidence_id}:claim_parsing','node_type':'claim_parsing','label':'AI Claim Parsing',
             'data':parsed}
    position=next(i for i,s in enumerate(node.chain) if s['node_type']=='variables')
    node.chain[position:position]=[original,parsing]
    for step in node.chain:
        kind=step['node_type']
        if kind=='variables':
            step['data']={'confirmed_claim':parsed['confirmed_claim'],'confirmed_at':parsed['confirmed_at'],
                          'mapping_confirmed':True,'source_sha256':dataset.sha256}
        elif kind=='method_audit':
            step['data']=audit
        elif kind=='claim':
            step['label']='Confirmed Claim'
            step['data']={'text':node.claim_text,'causal_downgrade_confirmed':parsed['causal_downgrade_confirmed']}
        elif kind=='judgment':
            step['label']='Final Judgment'
            step['data']=node.final_judgment
        elif kind=='wording':
            step['data']={'text':node.suggested_wording}
    # Reuse all V2 subnodes and only rebuild main-chain edges around the inserted steps.
    branches=[edge for edge in node.edges if edge['relation']=='has_analysis_path']
    node.edges=[{'from':a['node_id'],'to':b['node_id'],'relation':'next_evidence_step'} for a,b in zip(node.chain,node.chain[1:])]+branches
    node.to_json()
    return node
