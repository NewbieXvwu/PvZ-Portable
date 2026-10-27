"""task133 的纯逻辑小测试；真实引擎只做任务书指定的一关开局冒烟。"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_TEMP = tempfile.TemporaryDirectory(prefix="llmplay2-test-")
os.environ["PVZ_LLMPLAY_SESSION"] = str(Path(_TEMP.name) / "budget.json")
os.environ["LLMPLAY_OUT"] = str(Path(_TEMP.name) / "out")

from llmplay.game import Game, validate_deck
from llmplay.delivery import _deck_progress, _robust, _derive


def test_boss_tick_checkpoint_uses_recorded_deck_and_counts_eight_futures():
    class Env:
        def snapshot_fast(self): return "snapshot"
        def restore_fast(self, snapshot): pass
        def observe(self): return dict(tick=6000, wave=0, result=1)
        def reseed(self, seed): pass
        def release_snapshot(self, snapshot): pass

    class Trial:
        tick, terminal, ops = 6000, True, []
        def __init__(self): self.env = Env()
        def _start(self): pass
        def _reset_bookkeeping(self, obs): self.obs = obs
        def close(self): pass

    def policy(game): game.wait(30)

    parent = Game("demo_boss_50_orig_m100", seed=81032000, tick=6100)
    with patch.object(parent, "_start"), patch.object(parent, "_persist_best"), \
            patch("llmplay.game.Game", return_value=Trial()) as construct:
        report = parent.robustness(policy)
    assert report["passed"]
    assert report["checkpoints"]["tick:6000"] == dict(wins=8, k=8, reached=True, tick=6000)
    assert construct.call_args.kwargs["deck"] == parent.deck
    assert construct.call_args.kwargs["tick"] == 6000
    assert _robust(dict(task=dict(level=50), robustness=report))
    assert not _robust(dict(task=dict(level=49), robustness=report))
    parent.close()


def test_custom_deck_roundtrip_keeps_abandonment_and_rejects_more_actions():
    game = Game("demo_roof_49_doom_m200", seed=81000000,
                deck=[1, 33, 32, 34, 2, 3], reason="试双投手", strategy="投手", deck_review="经济较省")
    game.abandon_reason = "换成蘑菇试试"
    path = Path(_TEMP.name) / "run.json"
    game.save(path)
    restored = Game.load(path)
    assert restored.deck == game.deck
    assert restored.deck_reason == game.deck_reason
    assert restored.deck_review == game.deck_review
    assert restored.abandon_reason == game.abandon_reason
    try:
        restored.wait(1)
        raise AssertionError("放弃局仍可推进")
    except RuntimeError:
        pass
    game.close()
    restored.close()


def test_only_terrain_length_and_unique_cards_constrain_selection():
    validate_deck(dict(level=42), [1, 33, 32, 34, 2, 38])  # 金盏花不在黑名单
    for level, deck in [(22, [0, 1, 2, 3, 4, 5]), (42, [0, 1, 2, 3, 4, 5]),
                        (2, [0, 1]), (2, [0, 1, 2, 3, 4, 4])]:
        try:
            validate_deck(dict(level=level), deck)
            raise AssertionError("硬约束未生效")
        except ValueError:
            pass


def test_three_distinct_decks_need_two_unique_seeds_each():
    entries = []
    for i, deck in enumerate(([0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8], [0, 1, 2, 9, 10, 11])):
        for seed in (2*i, 2*i+1):
            entries.append(dict(deck=deck, strategy=str(i), seed=seed, won=True, robust=True))
    groups, selected = _deck_progress(entries + entries[:1])
    assert len(selected) == 3
    assert all(g["wins"] == 2 for g in groups.values())
    assert _deck_progress(entries[:-1])[1] is None
    reordered = dict(entries[0], deck=list(reversed(entries[0]["deck"])), won=False, robust=False)
    groups, _ = _deck_progress(entries + [reordered])
    assert len(groups) == 3 and sum(g["losses"] for g in groups.values()) == 1


def test_total_wins_count_deck_seed_pairs_and_robust_seeds_remain_unique():
    entries = []
    for i, (deck, seeds) in enumerate((([0,1,2,3,4,33], [10,11]),
                                      ([0,1,2,5,6,7,33], [11,12]),
                                      ([0,1,2,8,9,10,33], [10,12]))):
        for seed in seeds:
            entries.append(dict(task_id="llmplay2_f_l42_m100", deck=deck, strategy=str(i),
                                seed=seed, valid=True, won=True, robust=True, robustness={},
                                outcome=dict(final_tick=1)))
    task = _derive(entries)["tasks"]["llmplay2_f_l42_m100"]
    assert task["wins"] == 6 and task["robust"] == 3 and task["complete"]
