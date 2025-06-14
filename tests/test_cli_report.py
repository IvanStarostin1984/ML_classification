from __future__ import annotations

import os
import sys
import subprocess
import sysconfig
from pathlib import Path


def test_cli_report(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable,
        '-m',
        'pip',
        'install',
        '-e',
        str(root),
    ], check=True, capture_output=True, text=True)

    env = os.environ.copy()
    scripts_dir = Path(sysconfig.get_path('scripts'))
    env['PATH'] = str(scripts_dir) + os.pathsep + env.get('PATH', '')

    subprocess.run(
        ['mlcls-report'],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    out = tmp_path / 'report_artifacts' / 'pipeline_tables.txt'
    assert out.exists()
