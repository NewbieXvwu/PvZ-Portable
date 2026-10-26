"""Opt-in local optimizer recovery; final student artifacts remain model.pt."""
import os
from pathlib import Path
import random
import signal

import numpy as np
import torch


class TrainingRecovery:
    def __init__(self, path, contract, *, cuda=False):
        self.path = Path(path) if path else None
        self.contract, self.cuda = contract, cuda
        self.requested = False
        self.previous_handler = None
        self.resumes = 0
        if self.path:
            self.previous_handler = signal.signal(signal.SIGTERM, self._request)

    def _request(self, _number, _frame):
        self.requested = True  # serialize only at a completed optimizer/validation batch

    def rng_state(self):
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if self.cuda else None)

    @staticmethod
    def restore_rng(state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if state['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['cuda'])

    def load(self, net, optimizer, schedule):
        if not self.path or not self.path.exists():
            return None
        state = torch.load(self.path, map_location='cpu', weights_only=False)
        if state['version'] != 1 or state['contract'] != self.contract:
            raise ValueError('recovery checkpoint recipe/row contract differs')
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        if (schedule is None) != (state['schedule'] is None):
            raise ValueError('recovery learning-rate schedule differs')
        if schedule is not None:
            schedule.load_state_dict(state['schedule'])
        self.resumes = state['resumes'] + 1
        self.restore_rng(state['rng'])
        return state['progress']

    def save(self, net, optimizer, schedule, progress, *, rng=None):
        if not self.path:
            return
        state = dict(version=1, contract=self.contract, net=net.state_dict(),
                     optimizer=optimizer.state_dict(),
                     schedule=None if schedule is None else schedule.state_dict(),
                     progress=progress, rng=self.rng_state() if rng is None else rng,
                     resumes=self.resumes)
        temporary = self.path.with_name(self.path.name+f'.tmp-{os.getpid()}')
        try:
            torch.save(state, temporary)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self):
        if self.path:
            signal.signal(signal.SIGTERM, self.previous_handler)


def epoch_permutation(train_idx, epoch_rng=None):
    """Regenerate a partial epoch's order without consuming its current RNG state."""
    if epoch_rng is None:
        epoch_rng = torch.get_rng_state()
        return train_idx[torch.randperm(len(train_idx))], epoch_rng
    current = torch.get_rng_state()
    try:
        torch.set_rng_state(epoch_rng)
        return train_idx[torch.randperm(len(train_idx))], epoch_rng
    finally:
        torch.set_rng_state(current)
