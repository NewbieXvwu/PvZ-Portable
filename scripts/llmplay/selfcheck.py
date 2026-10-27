"""task133 的一次开局冒烟：自选卡、放弃、同种子换卡重开并存档。"""
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    if sys.version_info < (3, 10) or platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        print("SELFCHECK FAILED: 需要 Python>=3.10 与 x86_64 Linux")
        return 1
    with tempfile.TemporaryDirectory(prefix="llmplay2-smoke-") as temporary:
        root = Path(temporary)
        os.environ["LLMPLAY_OUT"] = str(root / "out")
        os.environ["PVZ_LLMPLAY_SESSION"] = str(root / "budget.json")
        from llmplay import Game, progress
        task = "llmplay2_f_l42_m100"
        seed = 133000000
        with Game(task, seed=seed, deck=[1,33,32,34,2,3,17],
                  reason="开局试双投手", strategy="双投手") as first:
            first.wait(1)
            first.deck_review = "双投手前期经济省，先换蘑菇观察"
            first.abandon(reason="测试同种子换卡重开")
            first.save(root / "abandoned.json")
        with Game(task, seed=seed, deck=[1,33,8,10,15,35,17],
                  reason="换蘑菇与爆破", strategy="蘑菇爆破") as second:
            second.wait(1)
            second.deck_review = "具备清场，咖啡豆经济仍需实战判断"
            second.save(root / "restarted.json")
            archive = second.export_archive(root / "archive")
        meta = json.loads((archive / "meta.json").read_text())
        document = progress()
        assert len(document["tasks"]) == 19
        assert len(document["archives"]) == 2
        assert document["archives"][0]["abandoned"]
        assert meta["task"]["deck"] == [1,33,8,10,15,35,17]
        assert not meta["abandoned"] and meta["deck_reason"] == "换蘑菇与爆破"
        assert (root / "out" / "out.zip").is_file()
        print(json.dumps(dict(task=task, seed=seed, attempts=len(document["archives"]),
                              abandoned=document["archives"][0]["abandon_reason"],
                              restarted_deck=meta["task"]["deck"], saved=True), ensure_ascii=False))
    print("SELFCHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
