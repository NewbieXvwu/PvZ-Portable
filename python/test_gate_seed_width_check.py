from unittest.mock import patch

import gate_seed_width_check as probe


def sample(_env, seed, *_args):
    return dict(seed=seed, observed_rand_seed=seed, chain=[str(seed)])


def test_repeat_must_be_an_independent_sample_and_can_fail():
    calls = 0

    def nondeterministic(env, seed, *args):
        nonlocal calls
        calls += 1
        result = sample(env, seed, *args)
        if calls == 2:
            result['chain'] = ['different second reset']
        return result

    with patch.object(probe, 'chain', side_effect=nondeterministic):
        result = probe.measure(None, 100_000_000, 7, 100, 3)
    assert result['deterministic'] is False
    assert result['runs'][0]['chain'] != result['runs'][1]['chain']


def test_engine_seed_echo_detects_truncation_even_if_observations_differ():
    def truncated(env, seed, *args):
        result = sample(env, seed, *args)
        result['observed_rand_seed'] %= 65536
        return result

    with patch.object(probe, 'chain', side_effect=truncated):
        result = probe.measure(None, 100_000_000, 7, 100, 3)
    assert result['differs_from_uint16_residue']
    assert not result['exact_uint32_echo']
    assert (1 << 32) - 1 in result['seeds']
    assert 100_999_999 in result['seeds']
