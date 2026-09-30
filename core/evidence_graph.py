"""Self-contained SVG graph, backed by EvidenceNode IDs/edges, no CDN or rendering service."""
from html import escape
import json
import math


def graph_data(node):
    nodes=[]
    for step in node.chain:
        nodes.append({'id':step['node_id'],'type':step['node_type'],'label':step['label'],'data':step['data']})
        for child in step.get('subnodes',[]):
            data=node.to_dict()
            for part in child['data_ref'].split('.'):
                data=data.get(part,{}) if isinstance(data,dict) else {}
            nodes.append({'id':child['node_id'],'type':child['node_type'],'label':child['node_type'],
                          'data':data,'parent':step['node_id']})
    ids={n['id'] for n in nodes}
    if len(ids)!=len(nodes) or any(e['from'] not in ids or e['to'] not in ids for e in node.edges):
        raise ValueError('Evidence graph has duplicate IDs or dangling edges')
    return {'evidence_id':node.evidence_id,'nodes':nodes,'edges':node.edges}


CN={'raw_data':'原始数据','quality':'数据质量','original_claim':'原始主张','claim_parsing':'语言解析',
    'variables':'变量映射','sample':'有效样本','method_audit':'方法审计','method':'执行方法',
    'result':'统计结果','stability':'证据稳定性','claim':'确认后的主张','quality_sensitivity':'质量敏感性',
    'judgment':'最终证据判断','wording':'建议表述'}


def node_summary(item):
    d=item['data'];t=item['type']
    if t=='raw_data': return d.get('file_name','')
    if t=='quality': return '规则、标记、原始记录号/ID'
    if t=='claim_parsing': return {'live_api':'API语言解析 · 非统计结果','offline_contract_demo':'离线固定响应 · 非真实LLM','manual':'未调用LLM · 手动输入'}.get(d.get('provider',{}).get('mode'),'解析记录')
    if t=='variables': return '用户已确认' if d.get('mapping_confirmed') else '结构化变量'
    if t=='method_audit': return str(d.get('recommended_method',''))
    if t=='method': return str(d.get('selected_method',d.get('implementation',d.get('name',''))))[:32]
    if t=='result': return f"n={d.get('sample_size','')} · p={d.get('p_value',0):.3g}"
    if t=='stability': return '已执行' if d.get('available') else '未执行 / 不适用'
    if t=='judgment': return str(d.get('association_status',d.get('status','')))
    if t=='quality_sensitivity': return '翻转' if d.get('quality_sensitive',{}).get('changes',{}).get('claim_flip') else '场景对照与消融'
    if t=='sample': return f"{len(d.get('used_row_ids',[]))} 条有效记录"
    return str(d.get('text','可点击查看完整证据'))[:28]


def compact_label(text, budget=29):
    """Budget CJK glyphs as two Latin characters; full content stays in title/inspector."""
    units=0;result=''
    for char in text:
        units+=1 if ord(char)<128 else 2
        if units>budget:return result+'…'
        result+=char
    return result


GRAPH_STYLE='''
.d2c{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#192b40}
.d2c-scroll{overflow:auto;border:1px solid #d9e3ec;border-radius:12px;background:#f8fafc}
.d2c svg{display:block;min-width:920px;width:100%;height:auto}.d2c .node{cursor:pointer;outline:none}
.d2c .node rect{fill:white;stroke:#b9cbd9;stroke-width:1.5}.d2c .node:hover rect,.d2c .node:focus rect,.d2c .node.selected rect{stroke:#157a78;stroke-width:3;fill:#edf7f5}
.d2c .node text{fill:#192b40;pointer-events:none}.d2c .node .sub{fill:#536a7d;font-size:11px}
.d2c-inspector{background:#edf5f4;border-radius:10px;padding:14px;margin-top:12px}
.d2c-inspector pre{max-height:290px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
'''


def graph_fragment(node):
    graph=graph_data(node)
    mains=[n for n in graph['nodes'] if 'parent' not in n]
    children=[n for n in graph['nodes'] if 'parent' in n]
    positions={}
    for i,item in enumerate(mains):
        row,col=divmod(i,4)
        if row%2: col=3-col
        positions[item['id']]=(25+col*240,25+row*130)
    start=math.ceil(len(mains)/4)
    for i,item in enumerate(children):
        row,col=divmod(i,4);positions[item['id']]=(25+col*240,25+(start+row)*130)
    height=(start+math.ceil(len(children)/4))*130+20
    edges=[]
    for edge in graph['edges']:
        x1,y1=positions[edge['from']];x2,y2=positions[edge['to']]
        if y1==y2:
            sx,ex=(x1+210,x2) if x2>x1 else (x1,x2+210)
            path=f'M {sx} {y1+43} L {ex} {y2+43}'
        else:
            path=f'M {x1+105} {y1+86} C {x1+105} {y1+110} {x2+105} {y2-20} {x2+105} {y2}'
        edges.append(f'<path data-from="{escape(edge["from"],quote=True)}" data-to="{escape(edge["to"],quote=True)}" d="{path}" fill="none" stroke="#7194a3" stroke-width="1.6" marker-end="url(#d2c-arrow)"/>')
    boxes=[]
    for i,item in enumerate(mains+children):
        x,y=positions[item['id']]
        boxes.append(f'<g class="node" role="button" tabindex="0" aria-label="{escape(item["label"],quote=True)}" data-node-index="{i}" transform="translate({x} {y})">'
                     f'<rect width="210" height="86" rx="10"/><text x="12" y="23" font-size="14" font-weight="600">{escape(CN.get(item["type"],item["type"]))}</text>'
                     f'<title>{escape(node_summary(item))}</title><text x="12" y="42" class="sub">{escape(compact_label(item["label"]))}</text><text x="12" y="66" class="sub">{escape(compact_label(node_summary(item)))}</text></g>')
    graph['nodes']=mains+children
    payload=json.dumps(graph,ensure_ascii=False,allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    return ('<section class="d2c"><h2>D2C Evidence Graph / 证据图</h2><p>箭头表示证据追溯关系，不表示因果。点击节点查看数据、规则与来源。</p>'
            f'<div class="d2c-scroll"><svg id="d2c-graph" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 {height}" aria-label="D2C Evidence Graph">'
            '<defs><marker id="d2c-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7194a3"/></marker></defs>'
            +''.join(edges)+''.join(boxes)+'</svg></div><div class="d2c-inspector"><strong id="d2c-title">选择任一节点</strong><pre id="d2c-detail">完整证据也保存在下方JSON中。</pre></div>'
            f'<script type="application/json" id="d2c-data">{payload}</script></section>')


GRAPH_SCRIPT='''
const graph=JSON.parse(document.getElementById('d2c-data').textContent);
document.querySelectorAll('#d2c-graph .node').forEach(el=>{
 const select=()=>{document.querySelectorAll('.node.selected').forEach(n=>n.classList.remove('selected'));el.classList.add('selected');
 const n=graph.nodes[Number(el.dataset.nodeIndex)];document.getElementById('d2c-title').textContent=n.label+' · '+n.id;
 document.getElementById('d2c-detail').textContent=JSON.stringify(n.data,null,2);};
 el.addEventListener('click',select);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
});
'''


def graph_html(node):
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>'+GRAPH_STYLE+'</style></head><body>'+graph_fragment(node)+'<script>'+GRAPH_SCRIPT+'</script></body></html>'
