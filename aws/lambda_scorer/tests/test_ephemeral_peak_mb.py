import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "infra" / "ephemeral_peak_mb.py"


def test_ephemeral_peak_mb_script_stdout():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--fetch-batch-size", "100", "--avg-kb-per-image", "300"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "fetch_batch_size=100" in r.stdout
