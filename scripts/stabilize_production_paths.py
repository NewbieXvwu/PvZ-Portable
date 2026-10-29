"""Path-only migration for existing code snapshots; no training or archive replay."""
from __future__ import annotations

import json
from pathlib import Path
import re

PANELS = {
    'dw4/artifacts/task23/unified_manifest.json': 'unified_v1.json',
    'dw5/artifacts/task27/unified_v2.json': 'unified_v2.json',
    'identity/artifacts/stage4/day_panel_identity.json': 'day_identity.json',
}


def conveyor_text(text, name):
    """Small in-place edits for production snapshots, preserving their other changes."""
    if name.startswith('panel_') and name.endswith('.json'):
        return re.sub(r'(?m)^\s*"day_orig_manifest"\s*:\s*"[^"]*",?\n', '', text)
    if name not in ('make_multiplier_tasks.py', 'render_episode.py', 'b3_noninferiority.py',
                    'build_adventure_gap.py', 'unified_panel.py', 'pvz_recipe.py'):
        return text
    before = text
    panel = (r'(?:\.claude/' r'worktrees/conveyor/artifacts/stage4/day_panel\.json|'
             r'configs/panels/day_orig_panel\.json)')
    text = re.sub(r"(?:MAIN|ROOT|Path\(__file__\)\.resolve\(\)\.parents\[1\])\s*/\s*(['\"])" + panel + r'\1',
                  'DAY_ORIG_MANIFEST', text)
    text = re.sub(r"Path\((['\"])[^'\"]*/" + panel + r'\1\)', 'DAY_ORIG_MANIFEST', text)
    if name == 'make_multiplier_tasks.py':
        text = text.replace("overrides.get('day_orig') or DAY_ORIG_MANIFEST", 'DAY_ORIG_MANIFEST')
        if 'if is_day_original_source(path):' not in text:
            text = text.replace('    document = json.loads(path.read_text())',
                                '    if is_day_original_source(path):\n'
                                '        return day_original_tasks()\n'
                                '    document = json.loads(path.read_text())', 1)
        text = text.replace("    parser.add_argument('--day-orig-manifest', type=Path)\n", '')
        text = text.replace("'day_orig': args.day_orig_manifest,", '')
    elif name == 'render_episode.py' and 'is_day_original_source(manifest_path)' not in text:
        text = text.replace('    doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))',
                            "    doc = ({'tasks': list(day_original_tasks().values())} if is_day_original_source(manifest_path)\n"
                            '           else json.loads(Path(manifest_path).read_text(encoding="utf-8")))')
    elif name == 'build_adventure_gap.py' and 'day_original_tasks().items()' not in text:
        text = re.sub(r'(?m)^    paths \+= \[DAY_ORIG_MANIFEST\]\n', '', text)
        text = text.replace("        for t in doc['tasks']:registry[t['task_id']]=(t,key)",
                            "        for t in doc['tasks']:registry[t['task_id']]=(t,key)\n"
                            "    sources['R5'] = dict(path='scripts/day_orig_tasks.py:day_original_tasks',\n"
                            "                         date=meta['captured_at'], note='generated original day tasks')\n"
                            '    for task_id, task in day_original_tasks().items():\n'
                            "        registry[task_id] = (task, 'R5')")
    elif name == 'unified_panel.py':
        text = text.replace('read(args.day_orig_manifest)', "{'tasks': list(day_original_tasks().values())}")
    elif name == 'pvz_recipe.py':
        text = text.replace("'perturb_manifest', 'night_manifest', 'day_orig_manifest', 'pool_manifest', 'pressure_manifest'}",
                            "'perturb_manifest', 'night_manifest', 'pool_manifest', 'pressure_manifest'}")
        if "allowed.add('day_orig_manifest')" not in text:
            text = text.replace("        allowed.add('rollout_policy')", "        allowed.add('rollout_policy')\n"
                                "        allowed.add('day_orig_manifest')  # Legacy frozen recipes only.")
    if text != before and name != 'pvz_recipe.py' and 'from day_orig_tasks import' not in text:
        text = text.replace('from pathlib import Path', 'from pathlib import Path\n'
                            'from day_orig_tasks import DAY_ORIG_MANIFEST, day_original_tasks, is_day_original_source', 1)
    return text
LEGACY = re.compile(r'\.claude/worktrees/(?!cq(?:/|\b)|obsup(?:/|\b))'
                    r'([A-Za-z0-9_-]+)/([A-Za-z0-9_./+*-]+)')
BARE = re.compile(r'\.claude/worktrees/(?!cq\b|obsup\b)([A-Za-z0-9_-]+)(?=[\s\'"\)]|$)')


def rewrite(text, suffix):
    prefix = '.claude/' + 'worktrees/'
    for old, name in PANELS.items():
        text = text.replace(prefix + old, 'configs/panels/' + name)
    text = text.replace('PACKAGE=$R/' + prefix + 'prodint16', 'PACKAGE=$R')
    text = text.replace('PACKAGE=$R/' + prefix + 'deploy17', 'PACKAGE=$R')
    text = text.replace('SRC=$R/' + prefix + 'deploy15', 'SRC=${PVZ_DEPLOY14F_SOURCE:-$R}')
    text = text.replace('EXE16=$PACKAGE/build16/pvz-portable', 'EXE16=$R/artifacts/production/deploy16/pvz-portable')
    text = text.replace(prefix + 'prodint16/build16/pvz-portable',
                        'artifacts/production/deploy16/pvz-portable')
    text = LEGACY.sub(lambda m: 'artifacts/archive/' + m[1] + '/' + m[2].removeprefix('artifacts/'), text)
    text = BARE.sub(lambda m: 'artifacts/archive/' + m[1], text)
    text = text.replace('.claude/worktrees/*/artifacts/', 'artifacts/archive/*/')
    if suffix == '.py':
        text = re.sub(r'MAIN\s*/\s*([\'"])(configs/panels/[^\'"]+)\1',
                      lambda m: 'Path(__file__).resolve().parents[1] / ' + repr(m[2]), text)
        text = re.sub(r'Path\(([\'"])/home/newbiexvwu/PvZ-search/(configs/panels/[^\'"]+)\1\)',
                      lambda m: '(Path(__file__).resolve().parents[1] / ' + repr(m[2]) + ')', text)
    return text


def recipe_text(text):
    from pvz_corpus_policy import retained_labels
    text = rewrite(text, '.json')
    value = json.loads(text)
    if value.get('kind') == 'training':
        labels = retained_labels(value['labels'])
        if labels != value['labels']:
            # Keep existing recipe formatting; only rewrite the label-list fragment.
            match = re.search(r'"labels"\s*:\s*(\[[\s\S]*?\])', text)
            fragment = match[1]
            reduced = re.sub(r'(?m)^[ \t]*"d10orig[12]_fast_search",?\n', '', fragment)
            try:
                if json.loads(reduced) != labels:
                    reduced = json.dumps(labels)
            except ValueError:
                reduced = json.dumps(labels)
            text = text[:match.start(1)] + reduced + text[match.end(1):]
        if str(value.get('data', '')).startswith('artifacts/archive/'):
            text = re.sub(r'("data"\s*:\s*)"[^"]+"',
                          r'\1"artifacts/student/frames_v2"', text, count=1)
    return text


def planned_changes(root):
    """Report (path, replacement) only; the boundary installer owns backups/writes."""
    for folder in ('scripts', 'python', 'recipes', 'configs'):
        for path in sorted((root / folder).rglob('*')):
            if not path.is_file() or path.suffix not in ('.py', '.sh', '.json') or path.name.startswith('test_'):
                continue
            before = path.read_text()
            # Existing code under these trees may differ; rewrite paths without replacing it.
            updated = conveyor_text(before, path.name)
            after = recipe_text(updated) if path.parent.name == 'recipes' and path.suffix == '.json' else rewrite(updated, path.suffix)
            if path.name == 'run_dw4_study.py':
                after = after.replace(
                    ",MAIN/'artifacts/archive/conveyor/task20/frames'", '')
                after = after.replace(
                    "    assert all(counts[label]==n for label,n in old['archive_counts'].items())\n"
                    "    assert counts['d10orig1_fast_search']==counts['d10orig2_fast_search']==128",
                    "    from pvz_corpus_policy import ABANDONED_LABELS\n"
                    "    assert all(counts[label]==n for label,n in old['archive_counts'].items()\n"
                    "               if label not in ABANDONED_LABELS)")
            if path.name == 'analyze_l20_conveyor.py':
                after = after.replace("str(ROOT / 'python')", "str(Path(__file__).resolve().parents[1] / 'python')")
                after = after.replace("ROOT / 'artifacts/task26/frames'", "ROOT / 'task26/frames'")
            if after != before:
                yield path, after
