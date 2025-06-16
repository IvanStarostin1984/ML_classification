import os
import sys
import subprocess
import sysconfig
from pathlib import Path


def test_cli_manifest(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )

    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("x")
    f2.write_text("y")

    env = os.environ.copy()
    scripts_dir = Path(sysconfig.get_path("scripts"))
    env["PATH"] = str(scripts_dir) + os.pathsep + env.get("PATH", "")

    subprocess.run(
        ["mlcls-manifest", str(f1), str(f2), "--out", "manifest.txt"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    out = tmp_path / "manifest.txt"
    assert out.exists()
    text = out.read_text()
    assert str(f1) in text
    assert "python" in text

