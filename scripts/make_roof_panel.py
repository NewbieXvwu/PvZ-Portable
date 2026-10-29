"""task46 roof registry: independent 32-seed evaluation and r7 training (no L50)."""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pvz_tasks import deck_for_level, profile_for_deck
from pvz_env import PvZEnv, TaskSpec
from zone_panel import Family, Zone, main_for

EVAL_START = 45000000
EVAL_COUNT = 32
TRAIN_BASE = 46000000
TRAIN_LEVELS = (41, 42, 43, 44, 46, 47, 48, 49)

ROOF = Zone(key='roof', task_prefix='roof_l', first_level=41, last_level=49, terrain='roof',
    default_out=ROOT / 'artifacts/task46/roof_panel.json',
    purpose='task46 roof original-difficulty evaluation and teacher-selected r7 training registry',
    families=(Family(''), Family('_r7', deck_for_level(41, family='roof_r7')),
              Family('_r7', deck_for_level(41, family='roof_r7'), 'train', exclude_levels=(45,))),
    train_seed_base=TRAIN_BASE, train_seeds=8192, train_seed_stride=10000,
    reserved_seed_ranges=((0, 8997663), (44000000, 44099999), (EVAL_START, EVAL_START + EVAL_COUNT - 1)),
    special_notes={45: 'Bungee Blitz conveyor: deck specifies only the initial hand; excluded from ordinary training'})


def main():
    unlocked = {}
    with PvZEnv(Path.home() / '.cache/pvz-research-resources',
                executable=ROOT / 'build/pvz-portable') as env:
        for level in range(41, 50):
            deck = deck_for_level(level, family='roof_r7')
            env.reset(deck=deck, task=TaskSpec(level=level, seed=EVAL_START,
                                              profile=profile_for_deck(deck)))
            unlocked[level] = env.deck_options()['available_plants']
            if not set(deck) <= set(unlocked[level]) or env.deck_warnings(deck)['warnings']:
                raise RuntimeError(f'L{level}: selected deck fails actual engine unlock/warning check')
    main_for(replace(ROOF, unlocked_by_level=unlocked),
             ['--combined', '--seed-start', str(EVAL_START), '--seeds', str(EVAL_COUNT),
                    *sys.argv[1:]])


if __name__ == '__main__':
    main()
