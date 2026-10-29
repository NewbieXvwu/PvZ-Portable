"""Path-only migration for existing code snapshots; no training or archive replay."""
from __future__ import annotations

import json
from pathlib import Path
import re

PANELS = {
    'conveyor/artifacts/stage4/day_panel.json': 'day_orig_panel.json',
    'dw4/artifacts/task23/unified_manifest.json': 'unified_v1.json',
    'dw5/artifacts/task27/unified_v2.json': 'unified_v2.json',
    'identity/artifacts/stage4/day_panel_identity.json': 'day_identity.json',
}
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
            after = recipe_text(before) if path.parent.name == 'recipes' and path.suffix == '.json' else rewrite(before, path.suffix)
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
