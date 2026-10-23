"""task52 experimental proposer. Only llm_candidate_experiment installs it in search_play."""
from __future__ import annotations

from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

PROMPT_VERSION = 1
MODEL = 'gemini-3.8-flash-high'
SYSTEM = ('你是植物大战僵尸候选提议者。仅根据当前公开局面，利用游戏知识提出最多3个现在可执行的'
          '种植动作，供模拟搜索检验。只能从给出的引擎合法动作表选择；packet是卡槽索引，'
          'row/col均从0开始。不要建议换卡、未来动作或未持有植物。不确定也可以返回空列表。'
          '回答JSON对象：{"actions":[{"type":"plant","packet":0,"row":0,"col":0}],'
          '"reason":"简短原因"}。不需要复述局面。')
GENERATION = {'temperature': 0, 'maxOutputTokens': 8192, 'responseMimeType': 'application/json'}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(canonical(value) + '\n')


def prompt_for(obs):
    from episode_query import _fmt_frame
    from render_episode import _frame
    # _frame explicitly selects public fields; never serialize a whole observation or
    # privileged_state. Human fog's zombies list is filtered by the engine itself.
    frame = _frame(obs)
    legal = obs.get('legal_actions', {}).get('plants', [])
    # _fmt_frame includes exactly episode_query's scene/bank/legend/board rendering.
    # Explicit lists disambiguate stacked plants and glyph collisions for zombies too.
    detail = {k: obs[k] for k in ('plants', 'zombies', 'fog_amount', 'fog_observation') if k in obs}
    return (_fmt_frame(None, frame, 0, 1) + '\n公开实体：' + canonical(detail)
            + '\n引擎合法种植动作：' + canonical(legal))


def filter_actions(payload, legal):
    """Strict engine-table membership; duplicates and over-limit suggestions stay measurable."""
    counts = Counter(returned=0, malformed=0, illegal=0, excess=0, duplicates=0, valid=0)
    if not isinstance(payload, dict) or not isinstance(payload.get('actions'), list):
        counts['malformed'] += 1
        return [], dict(counts)
    entries = payload['actions']
    counts['returned'] = len(entries)
    allowed = {(a['packet'], a['row'], a['col']) for a in legal}
    seen, accepted = set(), []
    for index, a in enumerate(entries):
        if index >= 3:
            counts['excess'] += 1
            continue
        if (not isinstance(a, dict) or a.get('type') != 'plant'
                or any(type(a.get(k)) is not int for k in ('packet', 'row', 'col'))):
            counts['malformed'] += 1
            continue
        key = (a['packet'], a['row'], a['col'])
        if key not in allowed:
            counts['illegal'] += 1
        elif key in seen:
            counts['duplicates'] += 1
        else:
            accepted.append(dict(type='plant', packet=key[0], row=key[1], col=key[2]))
            seen.add(key)
    counts['valid'] = len(accepted)
    return accepted, dict(counts)


def parse_payload(raw):
    # This gateway can wrap JSON in a Markdown fence despite responseMimeType. Strip
    # only a complete outer fence; prose or a damaged/truncated document stays invalid.
    fenced = re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*', raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    try:
        return json.loads(raw)
    except ValueError:
        return None


class GeminiCache:
    def __init__(self, directory, *, base_url='http://localhost:8317', model=MODEL,
                 timeout=180, retries=2, transport=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.base_url, self.model = base_url.rstrip('/'), model
        self.timeout, self.retries, self.transport = timeout, retries, transport

    def request(self, body):
        key = os.environ.get('GEMINI_API_KEY')
        if not key:
            raise RuntimeError('GEMINI_API_KEY is not loaded')
        req = urllib.request.Request(f'{self.base_url}/v1beta/models/{self.model}:generateContent',
                                     data=canonical(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    def propose(self, obs):
        prompt = prompt_for(obs)
        body = {'systemInstruction': {'parts': [{'text': SYSTEM}]},
                'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                'generationConfig': GENERATION}
        identity = {'version': PROMPT_VERSION, 'model': self.model, 'endpoint': self.base_url,
                    'body': body}
        position = digest(identity)
        path = self.directory / f'{position}.json'
        # Serialise identical requests across the two workers, including on cold cache.
        with (self.directory / f'{position}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            hit = path.exists()
            if hit:
                saved = json.loads(path.read_text())
                if saved['identity'] != identity:
                    raise ValueError('LLM cache identity mismatch')
            else:
                attempts = []
                for attempt in range(self.retries + 1):
                    started = time.monotonic()
                    try:
                        response = (self.transport or self.request)(body)
                        if not response.get('candidates'):
                            raise ValueError('Gemini response has no candidates')
                        entry = dict(position=position, context=getattr(self, 'context', {}), attempt=attempt, ok=True,
                                     seconds=time.monotonic()-started,
                                     usage=response.get('usageMetadata', {}))
                        attempts.append(entry)
                        append_json(self.directory / 'requests.jsonl', entry)
                        break
                    except Exception as error:
                        # Never record exception text/headers/body: a provider can echo credentials.
                        entry = dict(position=position, context=getattr(self, 'context', {}), attempt=attempt, ok=False,
                                     status=getattr(error, 'code', type(error).__name__),
                                     seconds=time.monotonic()-started)
                        attempts.append(entry)
                        append_json(self.directory / 'requests.jsonl', entry)
                        if attempt == self.retries:
                            raise RuntimeError(f'Gemini request failed: {entry["status"]}') from None
                        time.sleep(1 + attempt)
                saved = dict(identity=identity, response=response, attempts=attempts)
                atomic_json(path, saved)
        parts = saved['response']['candidates'][0].get('content', {}).get('parts', [])
        raw = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
        payload = parse_payload(raw)
        actions, counts = filter_actions(payload, (obs.get('legal_actions') or {}).get('plants', []))
        return dict(position=position, cache_hit=hit, actions=actions, counts=counts,
                    response_sha256=digest(saved['response']),
                    usage=saved['response'].get('usageMetadata', {}),
                    model_version=saved['response'].get('modelVersion'),
                    finish_reason=saved['response']['candidates'][0].get('finishReason'),
                    payload=payload, attempts=[] if hit else saved['attempts'])


class Triggers:
    def __init__(self, *, interval=1200, max_calls=4, adjacent_rows=1, threshold=None):
        self.interval, self.max_calls, self.adjacent_rows = interval, max_calls, adjacent_rows
        self.threshold = threshold
        self.seen, self.pending = set(), set()
        self.calls, self.last_tick = 0, -10**12

    def observe(self, obs):
        # obs['zombies'] is already the public visibility-filtered engine list.
        types = {z['type'] for z in obs.get('zombies', [])}
        self.pending.update(types - self.seen)
        self.seen.update(types)

    def budget_allows(self, tick):
        return self.calls < self.max_calls and tick - self.last_tick >= self.interval

    def low(self, values):
        return bool(values) and self.threshold is not None and all(v < self.threshold for v in values)

    def novel(self, obs, reference_row):
        if reference_row is None:
            return []
        return sorted({z['type'] for z in obs.get('zombies', []) if z['type'] in self.pending
                       and abs(z['row']-reference_row) <= self.adjacent_rows})

    def called(self, tick, types=()):
        self.calls += 1
        self.last_tick = tick
        self.pending.difference_update(types)
