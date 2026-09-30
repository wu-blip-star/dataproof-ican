"""Deterministic single-file HTML audit export. No LLM authorship or remote assets."""
from html import escape
from core.evidence_graph import graph_fragment, GRAPH_STYLE, GRAPH_SCRIPT


def export_html_report(node):
    payload=node.to_json()
    result=node.statistical_result or {}
    effect=result.get('effect_size') or node.effect_size or {}
    parsing=node.parsed_claim or {}
    mode=parsing.get('provider',{}).get('mode','manual')
    source_label={'offline_contract_demo':'离线固定响应演示（非真实LLM调用）','live_api':'OpenAI兼容API语言解析','manual':'手动结构化输入（未调用LLM）'}.get(mode,'解析来源未指定')
    summary=[('来源文件',node.source['file_name']),('SHA-256',node.source['file_sha256']),('解析来源',source_label),
             ('原始主张',node.original_text or node.claim_text),('本次确认的主张',node.claim_text),
             ('实际方法',node.method_name),('有效样本n',str(node.sample_size)),('统计量',str(node.statistic)),
             ('p-value',format(node.p_value,'.12g')),('效应量',str(effect.get('value',effect))),
             ('确认后主张判断',node.evidence_status),('原始主张核验范围',str((node.final_judgment or {}).get('original_claim_status','仅结构化主张')))]
    rows=''.join('<tr><th>'+escape(k)+'</th><td>'+escape(str(v))+'</td></tr>' for k,v in summary)
    sections=''.join('<details><summary>'+escape(step['label'])+'</summary><pre>'+escape(__import__('json').dumps(step['data'],ensure_ascii=False,indent=2,allow_nan=False))+'</pre></details>' for step in node.chain)
    limitations=''.join('<li>'+escape(v)+'</li>' for v in node.limitations)
    caution=(node.final_judgment or {}).get('warning','相关不等于因果。')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>DataProof 可信性审计报告</title><style>
body{{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#192b40;background:#f5f8fb;margin:0}}main{{max-width:1160px;margin:auto;padding:28px}}
header{{background:#192b40;color:white;padding:26px;border-radius:14px}}h1{{margin:0}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #d9e3ec;padding:10px;text-align:left;overflow-wrap:anywhere}}th{{width:170px}}.warning{{background:#fff8df;border-left:4px solid #b48825;padding:14px;margin:18px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:480px;overflow:auto;font-size:12px}}details{{background:white;padding:12px;border:1px solid #d9e3ec;margin:8px 0;border-radius:8px}}summary{{cursor:pointer;font-weight:600}}.muted{{color:#607587}}
@media(max-width:600px){{main{{padding:12px}}th{{width:95px}}}}@media print{{body{{background:white}}.d2c-scroll{{overflow:visible}}.d2c svg{{min-width:0}}pre{{max-height:none}}}}
{GRAPH_STYLE}</style></head><body><main><header><h1>DataProof 数证链 · 可信性审计报告</h1><p>由确定性程序从真实统计证据生成 · 非AI生成统计结论</p></header>
<p class="muted">记录 {escape(node.evidence_id)} · {escape(node.created_at)} · Schema {escape(node.schema_version)}</p>
<div class="warning">{escape(caution)}<br>{escape(source_label)}</div><h2>核验摘要</h2><table>{rows}</table>
<h2>建议表述</h2><p>{escape(node.suggested_wording)}</p>{graph_fragment(node)}
<h2>节点证据明细</h2>{sections}<h2>方法适用范围与限制</h2><ul>{limitations}</ul>
<details id="complete-evidence"><summary>完整 EvidenceNode JSON（含质量、样本、稳定性和消融）</summary><pre id="evidence-json">{escape(payload)}</pre></details>
<p class="muted">本文件可离线打开，无CDN、无外部脚本。复核计算请同时保留原始数据和运行环境；文件哈希不证明数据真实性。</p>
</main><script>{GRAPH_SCRIPT}</script></body></html>'''
