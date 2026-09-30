"""Fixed offline contract fixtures, NOT real LLM outputs or a general-language parser."""
from copy import deepcopy
from core.claim_parser import ClaimParserError, fingerprint

DEMO_TEXTS=['女生满意度显著高于男生。','学习时间越长，考试成绩越高。','短视频使用导致考试成绩下降。',
            '控制睡眠时间和短视频使用时间后，学习时间与考试成绩仍呈显著正相关。','不同年级的满意度存在显著差异。']


def response_for(index):
    base={'claim_type':'correlation','variables':['study_hours','exam_score'],'outcome':'exam_score','predictors':[],
          'group_variable':None,'comparison_groups':[],'direction':'positive','claims_significance':False,
          'causal_language':False,'causal_terms':[],'ambiguities':[],'unsupported_reason':None}
    if index==0:
        base.update(claim_type='group_difference',variables=['gender','satisfaction'],outcome='satisfaction',
                    group_variable='gender',comparison_groups=['女','男'],claims_significance=True)
    elif index==2:
        base.update(variables=['short_video_hours','exam_score'],direction='negative',causal_language=True,causal_terms=['导致'])
    elif index==3:
        base.update(claim_type='regression_association',variables=['study_hours'],
                    predictors=['study_hours','sleep_hours','short_video_hours'],claims_significance=True)
    elif index==4:
        base.update(claim_type='group_difference',variables=['grade','satisfaction'],outcome='satisfaction',
                    group_variable='grade',direction='different',claims_significance=True)
    return deepcopy(base)


class OfflineDemoProvider:
    def complete(self,text,schema):
        if text not in DEMO_TEXTS:
            raise ClaimParserError('离线契约演示仅支持预设的5句话，不具备任意自然语言理解能力；请使用API或结构化输入。')
        return response_for(DEMO_TEXTS.index(text)),{'provider':'offline-fixed-fixture','model':'none',
                    'mode':'offline_contract_demo','request_sha256':fingerprint({'text':text,'schema':schema})}
