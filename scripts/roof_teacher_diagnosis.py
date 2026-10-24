"""Isolated Gemini teacher sessions using episode_query tools and repository diagnosis skills.

No DSH deployment changes; same documented gateway/model, credential only from environment.
Raw API messages and tool traces stay under ignored artifacts/task46/phase2.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL='gemini-3.8-flash-high'
OUT=ROOT/'artifacts/task46/phase2'
RESOURCE='/home/newbiexvwu/.cache/pvz-research-resources'


def query(archive, command, arguments):
    allowed={'index','frame','lane','actions','whatif','constants','vocabulary','policy','events','between','trace'}
    if command not in allowed:return '不支持此命令；只允许读取、定位与whatif。'
    if any(a.split('=',1)[0] in {'--archive','--save-best','--resource-dir'} for a in arguments):return '存档自动限定；此会话不另存whatif。'
    cmd=[sys.executable,str(ROOT/'scripts/episode_query.py'),command,*arguments]
    if command not in {'constants','vocabulary'}:cmd+=['--archive',str(archive)]
    if command=='whatif':cmd+=['--resource-dir',RESOURCE]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=600,
                     env={**os.environ,'PVZ_NO_POPUP':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    return (p.stdout+ ('\nSTDERR:\n'+p.stderr[-4000:] if p.returncode else ''))[:60000]


def run(level, resume=False):
    mode = str(level)
    archive=OUT/f'l{level}-45000000-teacher'
    folder=OUT/f'gemini-{mode}'
    folder.mkdir(parents=True,exist_ok=True)
    document=ROOT/f'MEMORY/docs/teacher-diag-roof-l{level}.md'
    if document.exists() and not resume:
        raise RuntimeError('verbatim response already saved; use --resume to append evidence supplement')
    skills=(ROOT/'dsh/pvz-teacher/skills/pvz-episode-diagnosis/SKILL.md').read_text()
    meta=json.loads((archive/'meta.json').read_text())
    rows=[json.loads(x) for x in (archive/'timeline.jsonl').read_text().splitlines()]
    game=dict(seed=meta['task']['seed'],archive_sha256=hashlib.sha256((archive/'frames.npz').read_bytes()).hexdigest())
    system='你是植物大战僵尸项目的独立LLM教师。用简体中文回答。必须先index，再frame/lane/actions定位，再whatif验证；常量从工具读，不能用通用游戏记忆。不得编写规则策略、打法清单或奖励公式。\n'+skills
    prompt=f'''诊断实际搜索老师L{level}原版难度的输局。r7初始deck=[0,1,2,3,32,35,33]，dw4i_s1学生、v4叶值、branch1静态候选网；verify=9，student-candidates=2，其他command_for默认。L45是原生传送带，不能把初始deck当全局可用卡，逐帧核对手牌。
需回答：阳光收支与投入时序、花盆上落点、各行火力及投掷数量、破路类型/波次；卡组缺投掷或缺伞的证据有多强，学生/老师的屋顶落点与经济问题，叶值压平和工程问题各有什么实证。没有证据就标待检验，不因输局自动归因缺伞，也不要把射手株数当有效DPS。
先index，再frame/lane/actions定位，至少两个不同决策点（早期与破路前）的合法whatif候选。原存档由搜索老师执行；whatif是原老师前缀+候选+同一学生续走；工具同时重跑原动作的同学生续走基线。不能直接拿变体与老师后续原局比较，也不能把波次提高叫存活时间延长。分别报win/wave/tick/丢车/被吃/拒绝动作。行列0起，决策1起，tick为引擎时钟。区分事实、假设、反事实证据、后续具体数据需求；最多15轮工具交互，结束输出完整诊断原文，不写文件、不生成技能。
工具query支持index/frame/lane/actions/whatif/constants/vocabulary/policy/events/between/trace；arguments是CLI数组，如frame ["--tick","2040"]，actions ["--from","1","--to","100"]，whatif ["--decision","37","--try","wait:60","--try","plant:cabbage:0:1"]。plant数字是植物ID不是卡槽，先vocabulary核实；不要传--archive，系统自动限定。constants --list-sections 或 --section收窄。每个whatif最多3个明确候选，不枚举整盘。
终局与计数（只作目录，请自己用工具查）：{json.dumps(meta['outcome'],ensure_ascii=False)}
阳光初值{rows[0]['sun']}，最高{max(max(x['sun'],x['after_sun']) for x in rows)}，投掷峰值{max(sum(x['row_pults']) for x in rows)}，Python/native动作错位{sum(not x['native_matches'] for x in rows)}。'''
    if resume:
        prompt=(folder/'prompt.txt').read_text()
    else:
        (folder/'prompt.txt').write_text(prompt)
    declaration={'name':'query','description':'读取此局的episode_query或精确反事实；参数语义与CLI一致。',
        'parameters':{'type':'OBJECT','properties':{'command':{'type':'STRING'},'arguments':{'type':'ARRAY','items':{'type':'STRING'}}},'required':['command','arguments']}}
    contents=[{'role':'user','parts':[{'text':prompt}]}]
    endpoint='http://localhost:8317/v1beta/models/'+MODEL+':generateContent'
    resume_hint_sent = False
    tested_decisions = set()
    for turn in range(40):
        cached=folder/f'response-{turn:02}.json'
        if resume and cached.exists():
            result=json.loads(cached.read_text())
        else:
            if resume and not resume_hint_sent:
                contents.append({'role':'user','parts':[{'text':'延续已保存工具上下文；若不足两个决策点，补一个合法whatif，然后根据已有证据收敛输出最终诊断。'}]})
                resume_hint_sent=True
            payload={'systemInstruction':{'parts':[{'text':system}]},'contents':contents,
                     'tools':[{'functionDeclarations':[declaration]}],
                     'generationConfig':{'maxOutputTokens':14000,'thinkingConfig':{'thinkingBudget':8192}}}
            if turn>=32:
                payload.pop('tools')
                contents.append({'role':'user','parts':[{'text':'请结束工具探索，基于已有证据直接输出最终诊断；不确定处注明限制。'}]})
            request=urllib.request.Request(endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','x-goog-api-key':os.environ['GEMINI_API_KEY']})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request,timeout=240) as response:result=json.load(response)
                    break
                except urllib.error.HTTPError as e:
                    if e.code not in (429,500,502,503,504) or attempt==3:
                        detail=e.read().decode().replace(os.environ['GEMINI_API_KEY'],'[REDACTED]')
                        raise RuntimeError(f'Gemini gateway HTTP {e.code}: {detail[:2000]}') from None
                    print(mode,'retry',attempt,'HTTP',e.code,flush=True)
                    time.sleep(10*(attempt+1))
                except (TimeoutError,urllib.error.URLError):
                    print(mode,'retry',attempt,'transport timeout/error',flush=True)
                    if attempt==3:raise RuntimeError('transport failed; rerun with --resume') from None
            cached.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        candidate=result['candidates'][0]
        content=candidate['content'];contents.append(content)
        calls=[p['functionCall'] for p in content.get('parts',[]) if 'functionCall' in p]
        print(mode,'turn',turn,'calls',[(c['name'],c.get('args')) for c in calls],flush=True)
        if not calls:
            if len(tested_decisions) < 2 and turn < 32:
                contents.append({'role':'user','parts':[{'text':
                    '尚未完成两个不同决策点的合法反事实。现在停止查常量/目录，直接根据已读帧选早期与破路前两个决策号，'
                    '分别调用whatif，每处至少测试wait:60（如有另一合法落点可再加一个）。不要因当前动作不合法而放弃等待候选。'
                    '完成后再输出最终完整诊断，区分原老师终局与匹配学生续走基线。'}]})
                continue
            answer='\n'.join(p['text'] for p in content.get('parts',[]) if 'text' in p and not p.get('thought'))
            if not answer:raise RuntimeError('empty final teacher answer')
            header=f'# L{level} 屋顶实际搜索老师：Gemini诊断原文\n\n模型：`{MODEL}`；通过本机已授权CLIProxyAPI，凭据仅取`$GEMINI_API_KEY`。工具为本树episode_query，技能为dsh/pvz-teacher/skills。\n存档seed {game["seed"]}，frames SHA256 `{game["archive_sha256"]}`。以下为最终回答原文，未改写。\n\n---\n\n'
            if document.exists():
                with document.open('a') as stream:
                    stream.write('\n\n---\n\n## 反事实补充：同会话续接的最终回答原文\n\n'+answer+'\n')
            else:
                document.write_text(header+answer+'\n')
            (folder/'messages.json').write_text(json.dumps(contents,ensure_ascii=False,indent=2))
            print('SAVED',document,flush=True);return
        responses=[]
        for call in calls:
            args=call.get('args',{})
            tool_cache=folder/f'tool-{turn:02}-{len(responses)}.txt'
            if resume and tool_cache.exists():text=tool_cache.read_text()
            else:
                text=query(archive,args.get('command',''),args.get('arguments',[]))
                tool_cache.write_text(text)
            if args.get('command') == 'whatif' and '重放失败' not in text and '试 ' in text:
                arguments = args.get('arguments', [])
                if '--decision' in arguments:
                    tested_decisions.add(arguments[arguments.index('--decision') + 1])
            responses.append({'functionResponse':{'name':call['name'], **({'id':call['id']} if 'id' in call else {}), 'response':{'result':text}}})
        contents.append({'role':'user','parts':responses})
    raise RuntimeError('teacher session turn budget exhausted')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('level',type=int,choices=range(42,46));p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.level,a.resume)
