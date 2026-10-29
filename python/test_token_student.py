"""学生 v3 的公开输入、指针头与原生根节点契约。"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck
from render_episode import _frame

RESOURCE = Path(os.environ.get("PVZ_RESOURCE_DIR", "/home/newbiexvwu/.cache/pvz-research-resources"))
BINARY = Path(os.environ.get("PVZ_EXECUTABLE", ROOT / "build/pvz-portable"))
engine = pytest.mark.skipif(not BINARY.is_file() or not (RESOURCE / "main.pak").is_file(), reason="需引擎资源")


@engine
def test_public_tokens_opt_in_and_fog():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        deck = deck_for_level(31)
        original, _ = env.reset(deck=deck, task=TaskSpec(level=31, seed=157,
                               profile=profile_for_deck(deck), fog_observation="human_v2",
                               preplanted=((0, 0, 7),)))
        original, *_ = env.step(dict(type="wait", ticks=500))
        assert "v3" not in original
        env.set_token_observation(True)
        obs = env.observe()
        compact = _frame(obs)
        tokens = compact.pop("v3")
        assert compact == _frame(original)
        assert all(len(t) == 38 for t in tokens["tokens"])
        plant, = [t for t in tokens["tokens"] if t[0] == 3]
        assert plant[3:5] == [0, 7]
        assert plant[6 + 16] == 0  # fog-hidden dynamics unknown
        assert plant[6 + 2:6 + 10] == [-1] * 8
        env.set_lean_observation(True)
        assert env.observe()["v3"] == tokens
        env.set_token_observation(False)
        assert "v3" not in env.observe()


@engine
def test_public_tokens_moving_nut_and_backfill(tmp_path):
    from backfill_observation_v3 import replay_archive
    from pvz_event_env import EventWaitEnv
    from pvz_frame_store import write_frames, read_frames
    with EventWaitEnv(RESOURCE, executable=BINARY) as env:
        env.set_token_observation(True)
        deck = (3, 1, 0, 2, 4, 5)
        obs, _ = env.reset(deck=deck, task=TaskSpec(level=5, seed=157,
                          profile=profile_for_deck(deck), preplanted=((3, 2, 0),)))
        before = [t for t in obs["v3"]["tokens"] if t[0] == 3]
        obs, *_ = env.step(dict(type="wait", ticks=10))
        after = [t for t in obs["v3"]["tokens"] if t[0] == 3]
        assert before and after and before[0][6 + 4] != after[0][6 + 4]
        assert after[0][6 + 17] == 1 and after[0][6 + 18] > 0
        # A complete two-frame replay exercises the real action/label alignment and source immutability.
        from dataclasses import asdict
        task = TaskSpec(level=7, seed=157, profile=profile_for_deck(deck))
        obs, _ = env.reset(deck=deck, task=task)
        first = _frame(obs)
        obs, *_ = env.step(dict(type="wait", ticks=60))
        second = {**_frame(obs), "action": dict(type="wait", ticks=60),
                  "label": dict(type="plant", packet=0, row=1, col=1)}
        frames = [{k: v for k, v in f.items() if k not in ("v3", "sleep")} for f in (first, second)]
        source = tmp_path / "source/seed_157/teacher/frames.npz"
        source.parent.mkdir(parents=True)
        write_frames(source, frames)
        (source.parent / "meta.json").write_text(json.dumps({"task": {"seed": 157, "deck": deck},
                                                           "environment_task": asdict(task)}))
        original = source.read_bytes()
        result = replay_archive(env, source, tmp_path / "filled")
        assert not result["mismatch"] and result["frames_written"] == 2, result
        saved = read_frames(tmp_path / "filled/frames/seed_157/teacher/frames.npz")
        assert [{k: v for k, v in f.items() if k != "v3"} for f in saved] == frames
        assert source.read_bytes() == original
        frames[1]["sun"] += 1
        write_frames(source, frames)
        assert replay_archive(env, source, tmp_path / "bad")["mismatch"]
        assert not (tmp_path / "bad/frames").exists()


def test_token_pointer_permutation_padding_and_checkpoint(tmp_path):
    from pvz_token_student import TokenStudent, token_batch
    from pvz_rollout_policy import load_student
    torch.set_num_threads(1)
    torch.manual_seed(157)
    array = np.zeros((69, 38), np.float32)
    array[:, 3:6] = -1
    array[0, 0], array[1, 0] = 0, 1
    for i in range(54):
        array[2 + i, :5] = [2, 2, 2, i // 9, i % 9]
    array[56:66, 0] = 8
    array[56:63, 1] = np.arange(1, 8)
    array[66:, 0] = [3, 4, 5]
    array[66:, 1] = [2, 3, 4]
    net = TokenStudent(24, 2, 4, 48).eval()
    original = net(token_batch([array]), packet=torch.tensor([2]))
    permutation = np.array([4, 0, 2, 6, 1, 5, 3, 9, 8, 7])
    changed = array.copy()
    changed[56:66] = array[56:66][permutation]
    actual = net(token_batch([changed]), packet=torch.tensor([int(np.flatnonzero(permutation == 2)[0])]))
    torch.testing.assert_close(actual["plant"], original["plant"], rtol=0, atol=0)
    torch.testing.assert_close(actual["packet"], original["packet"][:, permutation], rtol=0, atol=0)
    torch.testing.assert_close(actual["cell"], original["cell"], rtol=0, atol=0)
    padded = net(token_batch([array, np.concatenate((array, array[66:]))]), packet=torch.tensor([2, 2]))
    torch.testing.assert_close(padded["cell"][:1], original["cell"], rtol=1e-5, atol=1e-5)
    checkpoint = tmp_path / "model.pt"
    torch.save(net.checkpoint(), checkpoint)
    restored = load_student(checkpoint)
    torch.testing.assert_close(restored(token_batch([array]))["plant"], original["plant"], rtol=0, atol=0)
    net.train()
    output = net(token_batch([array]), packet=torch.tensor([2]))
    loss = output["plant"].sum() + output["packet"][0, 2] + output["cell"][0, 5]
    loss.backward()
    assert torch.isfinite(net.encoder[0].qkv.weight.grad).all()
