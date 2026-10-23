#!/usr/bin/env python3
"""Emulate per-tensor int8 activations + per-output-channel int8 weights through the trunk.

The kernel experiment (scripts/bench_int8_kernel.cpp) proves the VNNI kernel is bit-exact with
the dequantised float reference, so this emulation measures exactly what the int8 path would
cost in accuracy: quantise, dequantise, run the same float convolution.  It runs no simulator.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
import torch
import torch.nn.functional as F

from bench_engine import STUDENT, _archive_rows
from pvz_rollout_policy import load_student
from pvz_student_observation import StudentHistory


def quantised_conv(x, conv):
    """Per-tensor u8 activations, per-output-channel s8 weights, dequantised back to float."""
    peak = x.abs().amax()
    scale = peak / 255.0 if peak > 0 else torch.tensor(1.0)
    qx = (x / scale).round().clamp(0, 255)
    weight_peak = conv.weight.abs().amax(dim=(1, 2, 3), keepdim=True)
    weight_scale = torch.where(weight_peak > 0, weight_peak / 127.0, torch.ones_like(weight_peak))
    qw = (conv.weight / weight_scale).round().clamp(-127, 127)
    return F.conv2d(qx * scale, qw * weight_scale, conv.bias, padding=conv.padding)


def trunk(net, board, quantised):
    x = board
    for index in range(0, len(net.trunk), 2):
        conv = net.trunk[index]
        x = quantised_conv(x, conv) if quantised else conv(x)
        x = torch.relu(x)
    return x


def main():
    torch.set_num_threads(1)
    net = load_student(str(STUDENT))
    if net.obs != "v2" or net.context:
        raise ValueError("this frozen experiment expects dw4i_s1 v2 without task context")
    manifest = json.loads((ROOT / "artifacts/perf/engine_roots.json").read_text())
    rows = []
    with torch.inference_mode():
        for family, entries in manifest["families"].items():
            for entry in entries:
                archive = Path(entry["archive"])
                frames = _archive_rows(archive / ("frames.npz" if (archive / "frames.npz").exists() else "frames.jsonl"))
                history = StudentHistory(net.obs)
                for index in range(len(frames)):
                    board, scalars = history.features(frames[index], frames[index - 1] if index else None)
                    if index not in entry["decisions"]:
                        continue
                    exact = trunk(net, board[None], False)
                    quant = trunk(net, board[None], True)
                    # The global context the plant and packet heads read.
                    shared = net.scalar(scalars[None])
                    g_exact = torch.cat([exact.mean(dim=(2, 3)), exact.amax(dim=(2, 3)), shared], dim=1)
                    g_quant = torch.cat([quant.mean(dim=(2, 3)), quant.amax(dim=(2, 3)), shared], dim=1)
                    rows.append({
                        "family": family, "frame": index,
                        "h_rms": float((quant - exact).pow(2).mean().sqrt()),
                        "h_scale": float(exact.pow(2).mean().sqrt()),
                        "g_rms": float((g_quant - g_exact).pow(2).mean().sqrt()),
                        "g_scale": float(g_exact.pow(2).mean().sqrt()),
                        "g_cosine": float(F.cosine_similarity(g_quant, g_exact, dim=1).item()),
                        "plant_exact": float(net.plant(g_exact).item()),
                        "plant_quant": float(net.plant(g_quant).item()),
                    })
    summary = {
        "roots": len(rows),
        "h_relative_rms": sum(r["h_rms"] for r in rows) / sum(r["h_scale"] for r in rows),
        "g_relative_rms": sum(r["g_rms"] for r in rows) / sum(r["g_scale"] for r in rows),
        "g_cosine_min": min(r["g_cosine"] for r in rows),
        "g_cosine_mean": sum(r["g_cosine"] for r in rows) / len(rows),
        "plant_abs_error_max": max(abs(r["plant_quant"] - r["plant_exact"]) for r in rows),
        "plant_sign_agreement": sum((r["plant_quant"] > 0) == (r["plant_exact"] > 0) for r in rows) / len(rows),
    }
    out = ROOT / "artifacts/perf/int8_emulation.json"
    out.write_text(json.dumps({"summary": summary, "roots": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
