"""Compatibility gates for the value-only v2 encoder (no simulator required)."""
import math
from pathlib import Path
import tempfile
import unittest
import torch
from pvz_frame_store import encoder_signature
from pvz_value_net import (ENCODER_V1, ENCODER_V2, SCALARS, SCALAR_BASE, PLANT_TYPES,
                           ValueNet, encode_frame, featurize, _featurize_reference, load_value_net)


def frame():
    return dict(tick=100,wave=1,sun=0,scene=dict(wave_count=20,night=True),plants=[],zombies=[],
                mowers=[],lanes=list(range(5)),packets=[[0,-1,False,0,0,0],[1,2,True,0,3500,150]],
                conveyor_start='original',graves=[[0,0]])


class ValueEncoderTest(unittest.TestCase):
    def test_legacy_bits_and_signature(self):
        for a,b in zip(featurize(frame()),_featurize_reference(frame())):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(encoder_signature(),'5c34482620514a7434e638058f9b75d05e51a8aa9c18b634a8f80e4dd424f008')
        self.assertEqual(featurize(frame())[1][4],0)  # intentional legacy bug stays frozen

    def test_empty_slots_and_free_duplicate_packets(self):
        f=frame();f['packets'] += [[2,2,False,0,3500,150]]
        b,s=encode_frame(f,2)
        self.assertAlmostEqual(float(s[4]),math.log1p(20)/math.log1p(50),places=6)
        self.assertEqual(s[SCALAR_BASE+2],1)
        self.assertEqual(s[SCALAR_BASE+PLANT_TYPES+2],1)
        self.assertEqual(s[SCALAR_BASE+PLANT_TYPES-1],0)
        self.assertEqual(len(s),SCALARS+6)
        self.assertEqual(s[SCALARS],1); self.assertEqual(s[SCALARS+4],1)
        f['packets'].reverse()
        torch.testing.assert_close(s,encode_frame(f,2)[1],rtol=0,atol=0)
        f.pop('conveyor_start')
        self.assertEqual(encode_frame(f,2)[1][SCALAR_BASE+2],0)

    def test_checkpoint_dispatch(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import search_play
        with tempfile.TemporaryDirectory() as tmp:
            for version,encoder in [(1,ENCODER_V1),(2,ENCODER_V2)]:
                net=ValueNet(8,encoder_version=version).eval();p=Path(tmp)/'model.pt'
                torch.save(dict(state_dict=net.state_dict(),width=8,encoder=encoder),p)
                loaded=load_value_net(p);self.assertEqual(loaded.encoder_version,version)
                b,s=encode_frame(frame(),version)
                with torch.inference_mode():expected=torch.sigmoid(net(b[None],s[None])).item()
            torch.save(dict(state_dict=net.state_dict(),encoder='unknown'),p)
            with self.assertRaises(ValueError):load_value_net(p)
        search_play._W.clear()

if __name__=='__main__':unittest.main()
