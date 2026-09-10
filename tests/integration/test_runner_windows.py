from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

from bg6022.orca.runner import RunnerResources, run_orca


@pytest.mark.windows_process
def test_cancel_stops_parent_and_child_process_tree(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows Job Object integration test")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "geometry.xyz").write_text("1\nhydrogen\nH 0 0 0\n", encoding="utf-8")
    (attempt / "input.inp").write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    facts = run_orca(
        executable=sys.executable,
        attempt_dir=attempt,
        resources=RunnerResources(
            cores=1,
            memory_mb=512,
            output_limit_bytes=1024 * 1024,
            workdir_limit_bytes=16 * 1024 * 1024,
        ),
        cancel=cancel,
        deadline=__import__("time").monotonic() + 30,
        data_root=tmp_path / "data",
    )
    assert facts.status in {"cancelled", "interrupted"}
    assert facts.process_tree_empty is True
