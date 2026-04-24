"""Grader: compile candidate's kernel for each test case, run, diff vs oracle.

Assumes:
  - Candidate's submission dir contains layout.csl, one or more pe_*.csl, run.py
  - Candidate's run.py accepts --name and --case args (like reference run.py)
  - Candidate's run.py prints 'PASS: <case>' on success, and exits 0
  - The cslc / cs_python binaries are on $PATH

Run:
  python3 -m pytest test_correctness.py -v --submission=../src-starter
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference import ALL_CASES  # type: ignore


# (case_name, compile params dict)
CASE_PARAMS = {
    "baseline":   {"P": 4, "d_dim": 32, "rows_per_pe": 128, "K": 16},
    "k_eq_1":     {"P": 2, "d_dim": 32, "rows_per_pe": 256, "K": 1},
    "k_large":    {"P": 2, "d_dim": 16, "rows_per_pe": 256, "K": 256},
    "uneven":     {"P": 4, "d_dim": 32, "rows_per_pe": 64,  "K": 16},
    "all_equal":  {"P": 2, "d_dim": 16, "rows_per_pe": 256, "K": 16},
    "duplicates": {"P": 2, "d_dim": 16, "rows_per_pe": 256, "K": 8},
}


def pytest_addoption(parser):
    parser.addoption("--submission", required=True, help="path to submission dir")


@pytest.fixture(scope="session")
def submission_dir(request):
    p = Path(request.config.getoption("--submission")).resolve()
    assert (p / "layout.csl").exists(), f"layout.csl not found in {p}"
    assert (p / "run.py").exists(), f"run.py not found in {p}"
    return p


def _params_flag(params: dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in params.items())


@pytest.mark.parametrize("case_name", list(CASE_PARAMS))
def test_case(case_name: str, submission_dir: Path, tmp_path: Path):
    params = CASE_PARAMS[case_name]
    build_dir = tmp_path / f"out_{case_name}"

    # Compile
    compile_cmd = [
        "cslc", "--arch=wse2", "layout.csl",
        "--fabric-dims=11,6", "--fabric-offsets=4,1",
        f"--params={_params_flag(params)}",
        "--memcpy", "--channels=1",
        "-o", str(build_dir),
    ]
    r = subprocess.run(compile_cmd, cwd=submission_dir, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"cslc failed:\n{r.stderr}\n{r.stdout}"

    # Run
    run_cmd = ["cs_python", "run.py", "--name", str(build_dir), "--case", case_name]
    r = subprocess.run(run_cmd, cwd=submission_dir, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"run.py failed:\n{r.stderr}\n{r.stdout}"
    assert f"PASS: {case_name}" in r.stdout, f"PASS marker missing in:\n{r.stdout}"
