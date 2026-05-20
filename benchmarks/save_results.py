"""Run benchmarks/run_all.py per tier and persist the table to BENCHMARKS.md."""

from __future__ import annotations

import datetime as _dt
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "BENCHMARKS.md"


def _run(script: str, *extra: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / script), *extra],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return proc.stdout.strip()


def run_tier(tier: str) -> str:
    return _run("run_all.py", "--tier", tier)


def sweep_tier(tier: str) -> str:
    return _run("sweep.py", "--tier", tier, "--markdown")


def _gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    except Exception:
        pass
    return "CPU fallback"


def main() -> None:
    headline = {tier: run_tier(tier) for tier in ("easy", "medium", "hard")}
    sweeps = {tier: sweep_tier(tier) for tier in ("easy", "medium", "hard")}
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = [
        "# Benchmarks",
        "",
        "Measured speedups of the Triton kernels in `src/flagos_track1/ops/` "
        "vs the PyTorch reference paths in `src/flagos_track1/reference/`.",
        "",
        f"- Device: **{_gpu_info()}**",
        f"- Last run: **{now}**",
        "- Methodology: median of 20 reps after 5 warmups via `triton.testing.do_bench`",
        "",
        "Re-run with:",
        "",
        "```bash",
        "python benchmarks/run_all.py --tier {easy,medium,hard,all}     # single-shape headline",
        "python benchmarks/sweep.py    --tier {easy,medium,hard,all} --markdown  # multi-shape",
        "python benchmarks/save_results.py                              # regenerates this file",
        "```",
        "",
        "## Headline (representative shape per op)",
    ]
    for tier in ("easy", "medium", "hard"):
        parts += ["", f"### {tier.capitalize()} tier", "", headline[tier]]

    parts += ["", "## Multi-shape sweep (small / medium / large per op)", ""]
    for tier in ("easy", "medium", "hard"):
        parts += [f"### {tier.capitalize()} tier sweep", "", sweeps[tier], ""]

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
