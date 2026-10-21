"""Identity packet heads are equivariant to slots, including duplicates with distinct cooldowns."""
import copy
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from pvz_rollout_policy import RolloutNet, act, export_student, features, load_student, packet_features
from pvz_value_net import BOARD_CHANNELS, SCALARS


class IdentityPacketsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.net = RolloutNet(8, 'meanmax', packet_head='identity').eval()
        with torch.no_grad():
            self.net.plant.weight.zero_()
            self.net.plant.bias.fill_(1)
        self.frame = dict(plants=[], zombies=[], mowers=[], lanes=list(range(5)),
                          tick=0, wave=0, sun=1000, scene={'wave_count': 10},
                          packets=[[i, kind, True, 0, 100, 25] for i, kind in enumerate([0, 1, 2, 3, 4, 5, 7])])

    @staticmethod
    def permuted(frame, order):
        result = copy.deepcopy(frame)
        result['packets'] = [[i, *frame['packets'][j][1:]] for i, j in enumerate(order)]
        return result

    def prediction(self, frame, net=None):
        board, scalars = features(frame, None)
        return (net or self.net)(board[None], scalars[None], packet_info=packet_features(frame)[None])

    def test_slot_permutation_changes_slots_but_not_plant_or_cell(self):
        legal = [{'packet': p, 'row': r, 'col': c} for p in range(7) for r in range(5) for c in range(9)]
        first = act(self.net, self.frame, None, legal)
        for order in ([6, 5, 4, 3, 2, 1, 0], [3, 5, 1, 6, 2, 0, 4]):
            frame = self.permuted(self.frame, order)
            out = self.prediction(frame)
            torch.testing.assert_close(out['packet'][0, :7], self.prediction(self.frame)['packet'][0, order], rtol=0, atol=1e-7)
            chosen = act(self.net, frame, None, legal)
            self.assertEqual(frame['packets'][chosen['packet']][1], self.frame['packets'][first['packet']][1])
            self.assertEqual((chosen['row'], chosen['col']), (first['row'], first['col']))

    def test_duplicate_cards_have_separate_state_and_symmetric_global_context(self):
        self.frame['packets'][1][1] = 0
        self.frame['packets'][1][2:5] = [False, 25, 100]
        order = [1, 0, 2, 3, 4, 5, 6]
        before = self.prediction(self.frame)
        after = self.prediction(self.permuted(self.frame, order))
        torch.testing.assert_close(before['plant'], after['plant'], rtol=0, atol=0)
        torch.testing.assert_close(before['packet'][0, order], after['packet'][0, :7], rtol=0, atol=0)
        self.assertEqual(before['packet'][0, 1].item(), -1e9)

    def test_empty_cooldown_and_unaffordable_slots_are_masked(self):
        self.frame['packets'][1][2] = False
        self.frame['packets'][2][-1] = 2000
        logits = self.prediction(self.frame)['packet'][0]
        self.assertEqual(logits[1].item(), -1e9)
        self.assertEqual(logits[2].item(), -1e9)
        self.assertTrue((logits[7:] == -1e9).all())
        self.assertGreater(logits[0].item(), -1e9)

    def test_selected_card_uses_the_shared_plant_embedding(self):
        self.frame['packets'][1][1] = 0
        b, s = features(self.frame, None)
        info = packet_features(self.frame)[None]
        h, g = self.net.encode(b[None], s[None], packet_info=info)
        torch.testing.assert_close(self.net.cell_logits(h, g, torch.tensor([0]), info),
                                   self.net.cell_logits(h, g, torch.tensor([1]), info), rtol=0, atol=0)
        loss = self.net.cell_logits(h, g, torch.tensor([0]), info).sum()
        loss.backward()
        self.assertGreater(self.net.packet_embedding.weight.grad[0].abs().sum().item(), 0)

    def test_checkpoint_roundtrip_and_v6_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'
            torch.save(dict(state_dict=self.net.state_dict(), pool='meanmax', packet_head='identity',
                            plant_embedding_dim=8), path)
            restored = load_student(path)
            torch.testing.assert_close(self.prediction(self.frame)['packet'],
                                       self.prediction(self.frame, restored)['packet'], rtol=0, atol=0)
            export_student(restored, Path(folder) / 'model.bin')
            raw = (Path(folder) / 'model.bin').read_bytes()
            self.assertEqual(raw[:8], b'PVZRN006')
            self.assertEqual(struct.unpack('<7I', raw[8:36]), (8, 1, 0, 1, 0, 1, 8))

    def test_missing_identity_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'packet_info'):
            self.net(torch.zeros(1, BOARD_CHANNELS, 6, 10), torch.zeros(1, SCALARS + 2))


if __name__ == '__main__':
    unittest.main()
