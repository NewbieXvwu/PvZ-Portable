"""The source gate must keep future identities and archive input indices intact."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from task78_data import midpoints, select, world_identity


def test_future_identity_is_not_collapsed_to_demonstration_seed():
    assert world_identity(Path('/data/seed_20261033_f44100001/demo')) == {
        'seed': 20261033, 'future_seed': 44100001}


def test_search_indices_are_input_frames_and_sparse_quota_is_deterministic():
    assert midpoints([0, 3, 8, 11, 15, 20, 40, 80]) == [8, 40]
    assert midpoints([3]) == [3]
    assert midpoints([]) == []


def test_selection_counts_worlds_not_duplicate_archive_labels():
    names = ['foguserl31snowm100', 'demofog31snow', 'demofog32snow',
             'demofog33cactussnow', 'demofog33cattailsnow', 'roofl41r7m100']
    paths = [Path(f'/data/seed_{1000+j*100+i}/c{round_}{name}_fast_search')
             for j, name in enumerate(names) for i in range(32) for round_ in range(2)]
    chosen = select(paths)
    assert len(chosen) == 64
    identities = [(group, tuple(world_identity(path).items())) for group, path in chosen]
    assert len(set(identities)) == 64
    assert select(list(reversed(paths))) == chosen
