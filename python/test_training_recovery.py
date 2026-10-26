"""A small interrupted optimizer run must retain its order, RNG, and momentum."""
import random

import numpy as np
import pytest
import torch

from pvz_training_recovery import TrainingRecovery, epoch_permutation


def setup():
    torch.manual_seed(19)
    random.seed(23)
    np.random.seed(29)
    net = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=.02)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1-step/20)
    return net, opt, schedule


def step(net, opt, schedule, index):
    x = torch.tensor([[float(index), .3]])
    loss = net(x).square().mean()
    opt.zero_grad(); loss.backward(); opt.step(); schedule.step()


def test_mid_epoch_resume_preserves_optimizer_and_rng(tmp_path):
    reference, opt, schedule = setup()
    order, _ = epoch_permutation(torch.arange(8))
    for index in order:
        step(reference, opt, schedule, index)
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    expected = {k: v.clone() for k, v in reference.state_dict().items()}
    net, opt, schedule = setup()
    order, epoch_rng = epoch_permutation(torch.arange(8))
    for index in order[:3]:
        step(net, opt, schedule, index)
    path = tmp_path/'recovery.pt'
    recovery = TrainingRecovery(path, {'recipe': 'frozen', 'rows': 8})
    recovery.save(net, opt, schedule, dict(epoch=1, next_start=3, epoch_rng=epoch_rng, steps=3))
    recovery.close()
    net, opt, schedule = setup()  # deliberately consume fresh initialization RNG again
    recovery = TrainingRecovery(path, {'recipe': 'frozen', 'rows': 8})
    try:
        state = recovery.load(net, opt, schedule)
        resumed, _ = epoch_permutation(torch.arange(8), state['epoch_rng'])
        assert torch.equal(order, resumed)
        for index in resumed[state['next_start']:]:
            step(net, opt, schedule, index)
        assert all(torch.equal(expected[k], v) for k, v in net.state_dict().items())
        assert opt.param_groups[0]['lr'] == .012
        assert recovery.resumes == 1
        assert random.random() == expected_rng[0]
        assert np.random.random() == expected_rng[1]
        assert torch.equal(torch.rand(3), expected_rng[2])
    finally:
        recovery.close()


def test_wrong_recipe_is_rejected_without_loading_weights(tmp_path):
    net, opt, schedule = setup()
    recovery = TrainingRecovery(tmp_path/'recovery.pt', {'recipe': 'original'})
    recovery.save(net, opt, schedule, {'steps': 0})
    recovery.close()
    recovery = TrainingRecovery(tmp_path/'recovery.pt', {'recipe': 'changed'})
    try:
        with pytest.raises(ValueError, match='contract differs'):
            recovery.load(net, opt, schedule)
    finally:
        recovery.close()
