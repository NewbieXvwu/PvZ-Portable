"""Value checkpoint dispatch (no simulator required)."""
from pathlib import Path
import tempfile
import unittest
import torch
from pvz_value_net import ENCODER_V1, ENCODER_V2, ValueNet, load_value_net


class ValueCheckpointDispatchTest(unittest.TestCase):
    def test_checkpoint_dispatch_and_unknown_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            for version, encoder in ((1, ENCODER_V1), (2, ENCODER_V2)):
                net = ValueNet(8, encoder_version=version).eval()
                path = Path(tmp) / 'model.pt'
                torch.save(dict(state_dict=net.state_dict(), width=8, encoder=encoder), path)
                self.assertEqual(load_value_net(path).encoder_version, version)
            torch.save(dict(state_dict=net.state_dict(), encoder='unknown'), path)
            with self.assertRaises(ValueError):
                load_value_net(path)


if __name__ == '__main__':
    unittest.main()
