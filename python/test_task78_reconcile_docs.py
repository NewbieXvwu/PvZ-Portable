"""The doc reconciliation tool must refuse to run where the receipts are absent."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/task78_reconcile_docs.py"


def test_exits_three_without_the_ignored_receipts(tmp_path):
    # A copy of the script under a root with no artifacts/task78 must exit 3, not pass.
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "task78_reconcile_docs.py").write_text(SCRIPT.read_text())
    result = subprocess.run([sys.executable, str(tmp_path / "scripts/task78_reconcile_docs.py")],
                            capture_output=True, text=True)
    assert result.returncode == 3, (result.returncode, result.stdout, result.stderr)
