"""Export a value checkpoint to PVZVN001 (no torch needed by the engine)."""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def export_value_net(net, path):
    from pvz_value_net import CONTEXT
    width = net.conv[0].out_channels
    with Path(path).open("wb") as output:
        output.write(struct.pack("<8sIII", b"PVZVN001", width, net.encoder_version,
                                 CONTEXT if net.context else 0))
        for parameter in net.state_dict().values():
            output.write(parameter.detach().cpu().numpy().astype("<f4").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from pvz_value_net import load_value_net
    export_value_net(load_value_net(args.checkpoint), args.output)


if __name__ == "__main__":
    main()
