import subprocess
import sys
from pathlib import Path


def test_installed_package_imports_without_gpu_or_network(tmp_path: Path) -> None:
    code = """
from unittest.mock import patch

import torch

with (
    patch("torch.cuda.is_available", return_value=False),
    patch("torch.backends.mps.is_available", return_value=False),
    patch("torch.cuda.init", side_effect=AssertionError("GPU initialization on import")),
    patch("socket.socket.connect", side_effect=AssertionError("Network access on import")),
):
    import tlab
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
