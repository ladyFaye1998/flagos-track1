"""``flagos`` command-line interface.

Examples
--------
  flagos list
  flagos test --tier easy
  flagos test --op softmax
  flagos bench --tier hard
  flagos package --out submission.zip
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import zipfile
from typing import Iterable

import click

from . import OP_REGISTRY, list_ops
from .bench import bench_op, format_results
from .testing import gen_input


@click.group()
@click.version_option()
def main() -> None:
    """FlagOS Track 1 — operator dev / test / bench / package CLI."""


@main.command("list")
@click.option("--tier", type=click.Choice(["easy", "medium", "hard", "all"]), default="all")
def list_cmd(tier: str) -> None:
    """List all registered operators grouped by tier."""
    tier_filter = None if tier == "all" else tier
    entries = list_ops(tier_filter)
    click.echo(f"{'Op':<22}  {'Tier':<8}")
    click.echo("-" * 34)
    for e in entries:
        click.echo(f"{e.name:<22}  {e.tier:<8}")
    click.echo("-" * 34)
    click.echo(f"{len(entries)} operators")


# ------------------------- TEST --------------------------------------------
def _resolve(ops: Iterable[str] | None, tier: str | None) -> list[str]:
    if ops:
        return list(ops)
    if tier:
        return [e.name for e in list_ops(tier)]
    return list(OP_REGISTRY)


@main.command("test")
@click.option("--op", "ops", multiple=True, help="Operator(s) to test (repeatable).")
@click.option("--tier", type=click.Choice(["easy", "medium", "hard"]), default=None)
@click.option("--verbose", is_flag=True)
def test_cmd(ops: tuple[str, ...], tier: str | None, verbose: bool) -> None:
    """Run pytest correctness suite (filter by op / tier)."""
    names = _resolve(ops, tier)
    args = [sys.executable, "-m", "pytest"]
    if tier:
        args.append(f"tests/{tier}")
    else:
        args.append("tests")
    args += ["-k", " or ".join(names)] if ops else []
    if verbose:
        args.append("-vv")
    click.echo("$ " + " ".join(args))
    sys.exit(subprocess.call(args))


# ------------------------- BENCH -------------------------------------------
@main.command("bench")
@click.option("--op", "ops", multiple=True)
@click.option("--tier", type=click.Choice(["easy", "medium", "hard"]), default=None)
@click.option("--shape", default="1024,4096", help="Comma-separated shape, e.g. 1024,4096")
def bench_cmd(ops: tuple[str, ...], tier: str | None, shape: str) -> None:
    """Run a quick wall-clock benchmark of one or more operators."""
    import torch

    from .ops.medium.layer_norm import layer_norm_op
    from .ops.medium.rms_norm import rms_norm_op
    from .ops.medium.cross_entropy import cross_entropy_op
    from .ops.medium.embedding import embedding_op
    from .ops.medium.dropout import dropout_op
    from .ops.medium.matmul import matmul_op
    from .ops.hard.flash_attention import flash_attention_op
    from .ops.hard.rope import rope_op
    from .ops.hard.fused_moe_topk import fused_moe_topk_op
    from .ops.hard.rms_norm_backward import rms_norm_backward_op
    from .reference import torch_ref as ref

    sh = tuple(int(s) for s in shape.split(","))
    names = _resolve(ops, tier)
    results = []
    for name in names:
        entry = OP_REGISTRY[name]
        x = gen_input(sh, dtype=torch.float16)
        try:
            if name in {
                "abs", "exp", "log", "sigmoid", "relu", "tanh", "gelu", "silu",
                "softmax", "argmax",
            }:
                ours = lambda x=x, fn=entry.op: fn(x)
                ref_fn = lambda x=x, fn=entry.reference: fn(x)
            elif name == "layer_norm":
                w = torch.ones(sh[-1], device=x.device, dtype=x.dtype)
                b = torch.zeros(sh[-1], device=x.device, dtype=x.dtype)
                ours = lambda: layer_norm_op(x, (sh[-1],), w, b)
                ref_fn = lambda: ref.ref_layer_norm(x, (sh[-1],), w, b)
            elif name == "rms_norm":
                w = torch.ones(sh[-1], device=x.device, dtype=x.dtype)
                ours = lambda: rms_norm_op(x, w)
                ref_fn = lambda: ref.ref_rms_norm(x, w)
            elif name == "cross_entropy":
                logits = x
                targets = torch.randint(0, sh[-1], (sh[0],), device=x.device)
                ours = lambda: cross_entropy_op(logits, targets)
                ref_fn = lambda: ref.ref_cross_entropy(logits, targets)
            elif name == "embedding":
                vocab, dim = 8192, sh[-1]
                w = gen_input((vocab, dim), dtype=torch.float16)
                idx = torch.randint(0, vocab, (sh[0],), device=x.device)
                ours = lambda: embedding_op(idx, w)
                ref_fn = lambda: ref.ref_embedding(idx, w)
            elif name == "dropout":
                ours = lambda: dropout_op(x, 0.1, seed=42)
                ref_fn = lambda: ref.ref_dropout(x, 0.1, seed=42)
            elif name == "matmul":
                a = gen_input((sh[0], sh[-1]), dtype=torch.float16)
                b_ = gen_input((sh[-1], sh[0]), dtype=torch.float16)
                ours = lambda: matmul_op(a, b_)
                ref_fn = lambda: ref.ref_matmul(a, b_)
            elif name == "flash_attention":
                B, H, N, D = 1, 8, 1024, 64
                q = gen_input((B, H, N, D), dtype=torch.float16)
                k = gen_input((B, H, N, D), dtype=torch.float16, seed=1)
                v = gen_input((B, H, N, D), dtype=torch.float16, seed=2)
                ours = lambda: flash_attention_op(q, k, v, causal=True)
                ref_fn = lambda: ref.ref_flash_attention(q, k, v, causal=True)
            elif name == "rope":
                B, N, D = sh[0], 1, sh[-1]
                xx = gen_input((B, N, D), dtype=torch.float16)
                cos = gen_input((B * N, D // 2), dtype=torch.float16, seed=1)
                sin = gen_input((B * N, D // 2), dtype=torch.float16, seed=2)
                ours = lambda: rope_op(xx, cos.view(B, N, D // 2), sin.view(B, N, D // 2))
                ref_fn = lambda: ref.ref_rope(xx, cos.view(B, N, D // 2), sin.view(B, N, D // 2))
            elif name == "fused_moe_topk":
                B, H, E, K = sh[0], sh[-1], 64, 4
                hidden = gen_input((B, H), dtype=torch.float16)
                router = gen_input((E, H), dtype=torch.float16, seed=1)
                ours = lambda: fused_moe_topk_op(hidden, router, K)
                ref_fn = lambda: ref.ref_fused_moe_topk(hidden, router, K)
            elif name == "rms_norm_backward":
                w = torch.ones(sh[-1], device=x.device, dtype=x.dtype)
                g = gen_input(sh, dtype=torch.float16, seed=1)
                ours = lambda: rms_norm_backward_op(g, x, w)
                ref_fn = lambda: ref.ref_rms_norm_backward(g, x, w)
            else:
                click.secho(f"  [skip] no bench harness for {name}", fg="yellow")
                continue
            results.append(bench_op(name, ours, ref_fn))
        except Exception as exc:  # pragma: no cover
            click.secho(f"  [error] {name}: {exc}", fg="red")
    if results:
        click.echo(format_results(results))


# ------------------------- PACKAGE -----------------------------------------
@main.command("package")
@click.option("--out", "out_path", default="submission.zip", show_default=True)
def package_cmd(out_path: str) -> None:
    """Zip src + tests + benchmarks + docs into a Track-1 submission archive."""
    root = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(out_path).resolve()
    include_dirs = ["src", "tests", "benchmarks", "scripts", "docs"]
    include_files = ["README.md", "pyproject.toml", "requirements.txt", "LICENSE"]

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d in include_dirs:
            base = root / d
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    zf.write(p, p.relative_to(root))
        for f in include_files:
            fp = root / f
            if fp.exists():
                zf.write(fp, f)

    size_kb = out.stat().st_size / 1024
    click.secho(f"Packaged {out} ({size_kb:.1f} KB)", fg="green")


@main.command("info")
def info_cmd() -> None:
    """Print environment info (Torch / Triton / CUDA / detected device)."""
    import torch

    from .device_caps import detect

    click.echo(f"python   : {sys.version.split()[0]}")
    click.echo(f"torch    : {torch.__version__}")
    click.echo(f"cuda     : {torch.version.cuda} (available={torch.cuda.is_available()})")
    try:
        import triton  # type: ignore

        click.echo(f"triton   : {triton.__version__}")
    except Exception:
        click.echo("triton   : <not installed>")
    caps = detect()
    click.echo(f"device   : {caps.name}")
    click.echo(f"vendor   : {caps.vendor}/{caps.arch}" + (f" (sm{caps.sm})" if caps.sm else ""))
    click.echo(f"cwd      : {os.getcwd()}")


if __name__ == "__main__":  # pragma: no cover
    main()
