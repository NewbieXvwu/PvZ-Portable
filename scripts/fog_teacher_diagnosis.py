"""Isolated Gemini teacher sessions using episode_query tools and repository diagnosis skills.

No DSH deployment changes; same documented gateway/model, credential only from environment.
Raw API messages and tool traces stay under ignored artifacts/task45/phase-d.
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
OUT=ROOT/'artifacts/task45/phase-d'
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


def run(mode, resume=False):
    archive=OUT/f'l31-{mode}-teacher'
    folder=OUT/f'gemini-{mode}'
    folder.mkdir(parents=True,exist_ok=True)
    document=ROOT/f'MEMORY/docs/teacher-diag-fog-l31-{mode}.md'
    if document.exists():raise RuntimeError('verbatim response already saved; do not overwrite')
    skills='\n\n'.join(p.read_text() for p in sorted((ROOT/'dsh/pvz-teacher/skills').glob('*/SKILL.md')))
    timeline=json.loads((ROOT/'MEMORY/docs/task45-fog-timelines.json').read_text())
    game=next(g for g in timeline['games'] if g['level']==31 and g['observation']==mode and g['policy']=='teacher')
    system='你是植物大战僵尸项目的LLM教师。用简体中文回答。下列技能属于待验证先验，不代替本局证据。必须先index，再frame/lane/actions定位，再whatif验证；常量从工具读，不能用通用游戏记忆。\n'+skills
    prompt=f'''诊断实际搜索老师的L31原版难度对局，观测臂={mode}。八卡f8，学生dw4i_s1，叶值v4、静态branch1，搜索配方见存档。任务：分析夜间泳池经济、具体落点、水路、非法动作和老师为何近似跟随学生。先用工具核实，不把以下统计当成无条件结论。不要写手工策略或奖励公式；建议应指向工程检查或待收集的状态/数据。
该存档确由搜索老师执行，不是纯学生。whatif明确采用“原老师动作前缀 + 此步候选 + 同一学生续走”；原动作基线也以同一学生续走重跑，因此比较只支持该续走条件下的动作效果，不能声称搜索老师续走一定同效。至少核验两处决策（例如早期经济/睡莲与晚期补防），候选必须合法；把没有改善和无结论也如实写明。两臂独立诊断，不猜另一臂结果。报告注明行列0起、tick单位与数值口径；区分事实、假设、反事实证据、下一步数据需求。最多25轮工具交互，完成后直接输出完整诊断，不生成技能、不写文件。
量化证据（原始timeline逐决策保存；最大火力是引擎SHOOTER子类计数，不是DPS）：
{json.dumps(game,ensure_ascii=False)}
工具query支持命令index/frame/lane/actions/whatif/constants/vocabulary/policy/events/between/trace，arguments是CLI参数数组。示例frame ["--tick","2040"]，actions ["--from","1","--to","100"]，whatif ["--decision","37","--try","wait:60","--try","plant:sunshroom:0:1"]；CLI的plant参数是植物ID或植物名，会映射到当时的卡槽；不是卡槽号。先读vocabulary/frame核实。存档自动限定，勿传--archive。constants用--list-sections和--section收窄。'''
    if resume:
        prompt=(folder/'prompt.txt').read_text()
    else:
        (folder/'prompt.txt').write_text(prompt)
    declaration={'name':'query','description':'读取此局的episode_query或精确反事实；参数语义与CLI一致。',
        'parameters':{'type':'OBJECT','properties':{'command':{'type':'STRING'},'arguments':{'type':'ARRAY','items':{'type':'STRING'}}},'required':['command','arguments']}}
    contents=[{'role':'user','parts':[{'text':prompt}]}]
    endpoint='http://localhost:8317/v1beta/models/'+MODEL+':generateContent'
    resume_hint_sent = False
    for turn in range(40):
        cached=folder/f'response-{turn:02}.json'
        if resume and cached.exists():
            result=json.loads(cached.read_text())
        else:
            if resume and not resume_hint_sent:
                contents.append({'role':'user','parts':[{'text':'上次API请求超时，延续已保存的完整工具上下文。请收敛：若尚未测两个不同决策点，再测一个不同决策点的合法候选，然后输出最终诊断。静态branch1是价值网名字，不是搜索只有一个分支；实际候选每卡1格+等待+学生原动作+至多2个学生候选。请控制报告篇幅，数值直接引用已得证据，不必重复查看相同工具。'}]})
                resume_hint_sent=True
            payload={'systemInstruction':{'parts':[{'text':system}]},'contents':contents,
                     'tools':[{'functionDeclarations':[declaration]}],
                     'generationConfig':{'maxOutputTokens':14000,'thinkingConfig':{'thinkingBudget':8192}}}
            if turn>=25:
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
            answer='\n'.join(p['text'] for p in content.get('parts',[]) if 'text' in p and not p.get('thought'))
            if not answer:raise RuntimeError('empty final teacher answer')
            header=f'# L31 {mode} 实际搜索老师：Gemini诊断原文\n\n模型：`{MODEL}`；通过本机已授权CLIProxyAPI，凭据仅取`$GEMINI_API_KEY`。工具为本树episode_query，技能为dsh/pvz-teacher/skills。\n存档seed {game["seed"]}，frames SHA256 `{game["archive_sha256"]}`。以下为最终回答原文，未改写。\n\n---\n\n'
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
            responses.append({'functionResponse':{'name':call['name'], **({'id':call['id']} if 'id' in call else {}), 'response':{'result':text}}})
        contents.append({'role':'user','parts':responses})
    raise RuntimeError('teacher session turn budget exhausted')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['human','xray']);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.mode,a.resume)
