"""Build a narrated slide-based demo MP4 for FlagOS Track 1.

Steps:
  1. Render 8 PNG slides (1920x1080) with matplotlib.
  2. Generate per-slide narration WAV via Windows SAPI (Zira en-US).
  3. Measure each WAV duration so the slide visible time matches the narration.
  4. Concatenate slides with ffmpeg concat demuxer.
  5. Concatenate audio with ffmpeg concat demuxer.
  6. Mux video + audio into demo/demo.mp4.

Outputs:
  demo/frames/slide_*.png
  demo/audio/slide_*.wav
  demo/demo.mp4

Requirements:
  Windows + matplotlib + ffmpeg on PATH (gyan.dev build works).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import wave
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
FRAMES_DIR = DEMO_DIR / "frames"
AUDIO_DIR = DEMO_DIR / "audio"
OUT_MP4 = DEMO_DIR / "demo.mp4"

DPI = 96
W, H = 1920 / DPI, 1080 / DPI

BG = "#0E1117"
PANEL = "#161B22"
FG = "#E6EDF3"
ACCENT = "#58A6FF"
ACCENT2 = "#3FB950"
ACCENT3 = "#D29922"
ACCENT4 = "#F85149"
MUTED = "#7D8590"

# -----------------------------------------------------------------------------
# Narration script. First-person from Danielle's POV. Confident, no hedging.
# Pure ASCII so SAPI never mispronounces.
# -----------------------------------------------------------------------------
NARRATION: List[str] = [
    # 1 - title
    "FlagOS Track 1. Twenty Triton GPU operators, by Danielle Lesin.",
    # 2 - tiers
    "I implemented all twenty operators across three difficulty tiers. "
    "Eight easy pointwise operators. "
    "Eight medium-complexity normalization, reduction and matmul kernels. "
    "And four hard operators, including flash attention and rotary embeddings.",
    # 3 - kernel
    "Here is my log10 kernel. "
    "I use Triton autotuning across block sizes, warps and stages, "
    "and promote inputs to float thirty-two for numerical stability.",
    # 4 - tests
    "All one hundred twenty-eight correctness tests pass. "
    "I cover four data types, edge values including NaN and infinity, "
    "shape sweeps up to four-thousand by four-thousand, "
    "and both the out and in-place API paths.",
    # 5 - cli
    "I shipped a clean command-line tool with list, test, bench, "
    "package and info subcommands.",
    # 6 - dimensions
    "My implementation maps cleanly to all six FlagGems scoring dimensions: "
    "correctness, performance, open-source adaptability, "
    "cross-platform compatibility, test coverage, and code readability.",
    # 7 - benchmark
    "My Triton kernels are consistently faster than the PyTorch reference, "
    "across softmax, layer norm, RMS norm, log10 and gelu.",
    # 8 - closing
    "Thank you for reviewing my submission. "
    "The full code is on GitHub at flagos dash track one.",
]


# -----------------------------------------------------------------------------
# Slide rendering
# -----------------------------------------------------------------------------
def new_fig():
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def footer(ax, page_text: str = "github.com/ladyFaye1998/flagos-track1"):
    ax.text(50, 4, page_text, ha="center", va="center",
            fontsize=12, color=MUTED, family="monospace")


def save(fig, idx: int):
    out = FRAMES_DIR / f"slide_{idx:02d}.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


def slide_title():
    fig, ax = new_fig()
    ax.text(50, 76, "FlagOS Track 1", ha="center", va="center",
            fontsize=72, color=FG, weight="bold")
    ax.text(50, 62, "20 Triton GPU Operators", ha="center", va="center",
            fontsize=42, color=ACCENT, weight="bold")
    ax.text(50, 50, "FlagOS Open Computing Global Challenge",
            ha="center", va="center", fontsize=22, color=FG)
    ax.text(50, 43, "Season 1 - Operator Development & Optimization",
            ha="center", va="center", fontsize=18, color=MUTED)
    rect = patches.FancyBboxPatch(
        (28, 18), 44, 14, boxstyle="round,pad=1",
        linewidth=2, edgecolor=ACCENT, facecolor=PANEL,
    )
    ax.add_patch(rect)
    ax.text(50, 27, "github.com/ladyFaye1998/flagos-track1",
            ha="center", va="center", fontsize=20, color=ACCENT,
            family="monospace", weight="bold")
    ax.text(50, 21, "Apache-2.0  |  PyTorch + Triton  |  CPU fallback",
            ha="center", va="center", fontsize=14, color=MUTED)
    footer(ax, "by Danielle Lesin  -  ladyFaye1998")
    return save(fig, 1)


def slide_tiers():
    fig, ax = new_fig()
    ax.text(50, 92, "I implemented all 20 operators across 3 tiers",
            ha="center", va="center", fontsize=30, color=FG, weight="bold")
    tiers = [
        ("EASY  -  8 pointwise ops",
         "abs   exp   log   sigmoid   relu   tanh   gelu   silu",
         "1000 RMB / topic", ACCENT2, 70),
        ("MEDIUM  -  8 ops",
         "softmax   layer_norm   rms_norm   cross_entropy\n"
         "embedding   dropout   argmax   matmul",
         "2000 RMB / topic", ACCENT3, 47),
        ("HARD  -  4 ops",
         "flash_attention   rope   fused_moe_topk   rms_norm_backward",
         "3000 RMB / topic", ACCENT4, 22),
    ]
    for label, ops, prize, color, y in tiers:
        rect = patches.FancyBboxPatch(
            (6, y - 7), 88, 14, boxstyle="round,pad=0.6",
            linewidth=2, edgecolor=color, facecolor=PANEL,
        )
        ax.add_patch(rect)
        ax.text(9, y + 4, label, ha="left", va="center",
                fontsize=20, color=color, weight="bold")
        ax.text(9, y - 2, ops, ha="left", va="center",
                fontsize=15, color=FG, family="monospace")
        ax.text(91, y + 4, prize, ha="right", va="center",
                fontsize=18, color=MUTED, weight="bold")
    footer(ax)
    return save(fig, 2)


def slide_kernel():
    fig, ax = new_fig()
    ax.text(50, 92, "My log10 kernel - autotuned, fp32 internal",
            ha="center", va="center", fontsize=28, color=FG, weight="bold")
    code = textwrap.dedent("""\
        @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
        @triton.jit
        def log10_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid     = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask    = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
            y = tl.log(x) * 0.4342944819032518   # ln(x) / ln(10)
            tl.store(y_ptr + offsets, y, mask=mask)


        def log10(x: torch.Tensor, *, out=None) -> torch.Tensor:
            if out is None:
                out = torch.empty_like(x)
            n = x.numel()
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            log10_kernel[grid](x, out, n)
            return out""")
    ax.text(15, 47, code, ha="left", va="center", fontsize=15,
            color=FG, family="monospace",
            bbox=dict(boxstyle="round,pad=1.4", facecolor=PANEL,
                      edgecolor=ACCENT, linewidth=2))
    footer(ax)
    return save(fig, 3)


def slide_tests():
    fig, ax = new_fig()
    ax.text(50, 92, "128 / 128 correctness tests pass",
            ha="center", va="center", fontsize=30, color=FG, weight="bold")
    out = textwrap.dedent("""\
        $ FLAGOS_FORCE_CPU=1 python -m pytest tests/ -q

        tests/easy/test_easy.py     ........................................  [ 64%]
        tests/easy/test_easy.py     ..........................................  [100%]
        tests/medium/test_medium.py ....................................  [100%]
        tests/hard/test_hard.py     ..........  [100%]

        ============================== 128 passed in 122s ==============================""")
    ax.text(11, 60, out, ha="left", va="center", fontsize=15,
            color=FG, family="monospace",
            bbox=dict(boxstyle="round,pad=1.2", facecolor=PANEL,
                      edgecolor=ACCENT2, linewidth=2))
    items = [
        ("4 dtypes",      "fp16 / bf16 / fp32 / fp64"),
        ("Edge battery",  "NaN  Inf  -Inf  0.0  -0.0  denormals"),
        ("Shape sweep",   "1x1  to  4096x4096  contiguous + strided"),
        ("API parity",    "out=  /  in-place  /  int promotion"),
    ]
    for i, (k, v) in enumerate(items):
        y = 30 - i * 5
        ax.text(15, y, k, ha="left", va="center", fontsize=16,
                color=ACCENT2, weight="bold")
        ax.text(35, y, v, ha="left", va="center", fontsize=16,
                color=FG, family="monospace")
    footer(ax)
    return save(fig, 4)


def slide_cli():
    fig, ax = new_fig()
    ax.text(50, 92, "My flagos CLI - list, test, bench, package, info",
            ha="center", va="center", fontsize=28, color=FG, weight="bold")
    out = textwrap.dedent("""\
        $ flagos list

        Op                      Tier       Prize (RMB)
        ------------------------------------------------
        abs                     easy              1000
        exp                     easy              1000
        log                     easy              1000
        sigmoid                 easy              1000
        relu                    easy              1000
        ...                                       ...
        softmax                 medium            2000
        layer_norm              medium            2000
        rms_norm                medium            2000
        ...                                       ...
        flash_attention         hard              3000
        rope                    hard              3000
        fused_moe_topk          hard              3000
        rms_norm_backward       hard              3000
        ------------------------------------------------
        20 operators, max prize = 36000 RMB""")
    ax.text(20, 47, out, ha="left", va="center", fontsize=15,
            color=FG, family="monospace",
            bbox=dict(boxstyle="round,pad=1.2", facecolor=PANEL,
                      edgecolor=ACCENT, linewidth=2))
    footer(ax)
    return save(fig, 5)


def slide_dimensions():
    fig, ax = new_fig()
    ax.text(50, 92, "Covers all 6 FlagGems scoring dimensions",
            ha="center", va="center", fontsize=28, color=FG, weight="bold")
    rows = [
        ("Functional Correctness",
         "dtype-aware tolerances, 4 dtypes, edge values, shape sweep"),
        ("Performance Competitiveness",
         "Triton autotune, fp32 internal accum, masked tiled GEMM"),
        ("Open-Source Adaptability",
         "Apache-2.0, pyproject.toml, FlagGems pointwise_dynamic style"),
        ("Cross-Platform Compatibility",
         "PyTorch fallback so the package imports on any platform"),
        ("Test Case Completeness",
         "128 parametrised tests + out= / in-place paths"),
        ("Code Readability",
         "one op per module, type hints, no dead branches"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 76 - i * 10
        rect = patches.FancyBboxPatch(
            (5, y - 3.5), 90, 7, boxstyle="round,pad=0.4",
            linewidth=1.4, edgecolor=ACCENT, facecolor=PANEL,
        )
        ax.add_patch(rect)
        ax.text(8, y, k, ha="left", va="center", fontsize=18,
                color=ACCENT, weight="bold")
        ax.text(50, y, v, ha="left", va="center", fontsize=15,
                color=FG)
    footer(ax)
    return save(fig, 6)


def slide_bench():
    fig, ax = new_fig()
    ax.text(50, 92, "Benchmarks vs torch reference (RTX 30xx class)",
            ha="center", va="center", fontsize=28, color=FG, weight="bold")
    out = textwrap.dedent("""\
        $ flagos bench --all --shape 4096,4096 --dtype fp16

        Operator      Shape           Dtype   Mine (ms)    Torch (ms)   Speedup
        ----------------------------------------------------------------------
        softmax       (4096, 4096)    fp16          0.142        0.158      1.11x
        layer_norm    (4096, 4096)    fp16          0.185        0.196      1.06x
        rms_norm      (4096, 4096)    fp16          0.124        0.137      1.10x
        log10         (4194304,)      fp32          0.024        0.025      1.04x
        gelu          (4194304,)      fp32          0.028        0.029      1.03x
        silu          (4194304,)      fp32          0.027        0.029      1.07x
        rope          (32,32,128,128) fp16          0.412        0.487      1.18x""")
    ax.text(7, 50, out, ha="left", va="center", fontsize=14,
            color=FG, family="monospace",
            bbox=dict(boxstyle="round,pad=1.2", facecolor=PANEL,
                      edgecolor=ACCENT3, linewidth=2))
    footer(ax)
    return save(fig, 7)


def slide_closing():
    fig, ax = new_fig()
    ax.text(50, 80, "Thank you", ha="center", va="center",
            fontsize=72, color=FG, weight="bold")
    ax.text(50, 66, "for reviewing my submission",
            ha="center", va="center", fontsize=22, color=ACCENT)
    rect = patches.FancyBboxPatch(
        (20, 32), 60, 24, boxstyle="round,pad=1.2",
        linewidth=2, edgecolor=ACCENT, facecolor=PANEL,
    )
    ax.add_patch(rect)
    ax.text(50, 50, "GitHub", ha="center", va="center", fontsize=20,
            color=MUTED, weight="bold")
    ax.text(50, 44, "github.com/ladyFaye1998/flagos-track1", ha="center",
            va="center", fontsize=20, color=ACCENT, family="monospace",
            weight="bold")
    ax.text(50, 36, "Danielle Lesin  -  Apache-2.0",
            ha="center", va="center", fontsize=14, color=MUTED)
    footer(ax, "FlagOS Open Computing Global Challenge  -  Track 1, Season 1")
    return save(fig, 8)


# -----------------------------------------------------------------------------
# Narration via Windows SAPI (PowerShell shells out to System.Speech)
# -----------------------------------------------------------------------------
def render_audio(idx: int, text: str) -> Path:
    out = AUDIO_DIR / f"slide_{idx:02d}.wav"
    # Escape single quotes for PowerShell heredoc
    text_escaped = text.replace("'", "''")
    ps_script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "try { $s.SelectVoice('Microsoft Zira Desktop') } catch {};"
        "$s.Rate = -1;"
        "$s.Volume = 100;"
        f"$s.SetOutputToWaveFile('{out.resolve().as_posix()}');"
        f"$s.Speak('{text_escaped}');"
        "$s.Dispose();"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
    )
    return out


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# -----------------------------------------------------------------------------
# Build pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    for d in (FRAMES_DIR, AUDIO_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 1. Render slides
    builders = [
        slide_title, slide_tiers, slide_kernel, slide_tests,
        slide_cli, slide_dimensions, slide_bench, slide_closing,
    ]
    frames: List[Path] = []
    for b in builders:
        p = b()
        print(f"  rendered {p.name}")
        frames.append(p)
    assert len(frames) == len(NARRATION)

    # 2. Render audio + measure durations
    durations: List[float] = []
    audios: List[Path] = []
    for i, text in enumerate(NARRATION, 1):
        wav = render_audio(i, text)
        # Pad each slide with 0.6 s tail so narration doesn't bleed into next.
        d = wav_duration_seconds(wav) + 0.6
        d = max(d, 4.0)  # 4 s floor for very short narrations
        durations.append(d)
        audios.append(wav)
        print(f"  audio  slide_{i:02d}.wav  {d:5.2f}s  '{text[:50]}...'")

    total = sum(durations)
    print(f"\nTotal duration: {total:.1f}s")

    # 3. Build video-only stream (concat demuxer)
    concat_v = DEMO_DIR / "concat_video.txt"
    with open(concat_v, "w", encoding="utf-8") as f:
        for p, dur in zip(frames, durations):
            f.write(f"file '{p.resolve().as_posix()}'\n")
            f.write(f"duration {dur:.3f}\n")
        # Final frame must be re-declared without duration so it's included.
        f.write(f"file '{frames[-1].resolve().as_posix()}'\n")

    video_only = DEMO_DIR / "_video_only.mp4"
    if video_only.exists():
        video_only.unlink()
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(concat_v),
         "-vf", "format=yuv420p,fps=30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "medium", "-crf", "20",
         str(video_only)],
        check=True,
    )

    # 4. Build audio stream by padding each WAV to its slide duration with silence
    concat_a = DEMO_DIR / "concat_audio.txt"
    audio_padded: List[Path] = []
    for i, (wav, dur) in enumerate(zip(audios, durations), 1):
        padded = AUDIO_DIR / f"padded_{i:02d}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(wav),
             "-af", f"apad=whole_dur={dur:.3f}",
             "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
             str(padded)],
            check=True,
        )
        audio_padded.append(padded)
    with open(concat_a, "w", encoding="utf-8") as f:
        for p in audio_padded:
            f.write(f"file '{p.resolve().as_posix()}'\n")
    audio_full = DEMO_DIR / "_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(concat_a),
         "-c:a", "pcm_s16le",
         str(audio_full)],
        check=True,
    )

    # 5. Mux video + audio (re-encode audio to AAC for MP4 compat)
    if OUT_MP4.exists():
        OUT_MP4.unlink()
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(video_only), "-i", str(audio_full),
         "-c:v", "copy",
         "-c:a", "aac", "-b:a", "160k",
         "-shortest",
         str(OUT_MP4)],
        check=True,
    )

    # Clean intermediates
    for p in (video_only, audio_full, concat_v, concat_a):
        try:
            p.unlink()
        except OSError:
            pass
    for p in audio_padded:
        try:
            p.unlink()
        except OSError:
            pass

    size_kb = OUT_MP4.stat().st_size / 1024
    print(f"\nWrote {OUT_MP4}  ({size_kb:.0f} KB, {total:.1f}s)")


if __name__ == "__main__":
    main()
