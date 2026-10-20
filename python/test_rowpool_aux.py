import struct
import tempfile
from pathlib import Path

import torch

from pvz_rollout_policy import RolloutNet, export_student, load_student


def test_rowpool_checkpoint_and_export_without_aux():
    torch.manual_seed(17)
    net = RolloutNet(13, "meanmax", True, "v2", row_pool=True, aux=True).eval()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "model.pt"
        binary = Path(directory) / "model.bin"
        torch.save({"state_dict": net.state_dict(), "pool": "meanmax", "context": True,
                    "obs": "v2", "row_pool": True}, checkpoint)
        loaded = load_student(checkpoint)
        assert loaded.row_pool and loaded.aux is None
        export_student(net, binary)
        data = binary.read_bytes()
        assert data[:28] == b"PVZRN005" + struct.pack("<IIIII", 13, 1, 1, 3, 1)
        expected_size = 28 + 4 * sum(value.numel() for name, value in net.state_dict().items()
                                    if not name.startswith("aux."))
        assert len(data) == expected_size
        export_student(loaded, binary)
        assert binary.read_bytes() == data
