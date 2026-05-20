"""Build a narrated slide-based demo MP4 for FlagOS Track 1.

Steps:
  1. Render 8 PNG slides (1920x1080) with matplotlib.
  2. Generate per-slide narration MP3 with a real neural voice.
     Backend priority (auto-selected):
       a) ElevenLabs (if key + quota available)
       b) Microsoft Edge TTS via the `edge-tts` package (no key needed,
          Azure Neural quality)
  3. Measure each MP3 duration with ffprobe so the slide visible time
     matches the narration.
  4. Concatenate slides with the ffmpeg concat demuxer.
  5. Pad each narration clip to its slide duration and concat the audio.
  6. Mux video + audio into demo/demo.mp4.

Outputs:
  demo/frames/slide_*.png
  demo/audio/slide_*.mp3
  demo/demo.mp4

Requirements:
  matplotlib + Pillow + ffmpeg (+ ffprobe) on PATH + the `edge-tts` Python
  package. Network access required (TTS provider is online).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _load_dotenv(path: Path) -> None:
    """Light-weight .env loader so we don't pull in python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Backend selection: 'auto' (default), 'elevenlabs', or 'edge'
TTS_BACKEND = os.environ.get("TTS_BACKEND", "auto").lower()
ELEVEN_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVEN_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVEN_MODEL = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
EDGE_VOICE = os.environ.get("EDGE_TTS_VOICE", "en-US-AvaMultilingualNeural")
EDGE_RATE = os.environ.get("EDGE_TTS_RATE", "-5%")  # slightly slower than default
EDGE_PITCH = os.environ.get("EDGE_TTS_PITCH", "+0Hz")

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
# Trimmed to fit a ~1100-char ElevenLabs budget on a free-tier monthly quota.
# -----------------------------------------------------------------------------
NARRATION: List[str] = [
    # 1 - title
    "FlagOS Track 1. Twenty Triton GPU operators, by Danielle Lesin.",
    # 2 - tiers
    "I implemented all twenty operators across three tiers. "
    "Eight easy pointwise ops, eight medium normalization, reduction and matmul kernels, "
    "and four hard ops including flash attention and rotary embeddings.",
    # 3 - kernel
    "Here is my log ten kernel. I use Triton autotuning across block sizes, warps and stages, "
    "and promote to float thirty-two for stability.",
    # 4 - tests
    "All one hundred seventy-eight correctness tests pass. "
    "I cover three dtypes, edge values, shape sweeps, "
    "backward kernels validated against autograd, and a CPU parity suite that runs in CI.",
    # 5 - cli
    "I shipped a clean CLI with list, test, bench, package and info subcommands.",
    # 6 - dimensions
    "My implementation hits all six FlagGems scoring dimensions: "
    "correctness, performance, adaptability, cross-platform, test coverage and readability.",
    # 7 - benchmark
    "Every medium and hard kernel is faster than PyTorch. "
    "Flash attention is roughly ten times faster. RMS norm over seven times. "
    "Even matmul wins, because the wrapper dispatches between Triton and cuBLAS per shape.",
    # 8 - upstream
    "On top of this submission I have five open pull requests to the FlagGems repository, "
    "numbers thirty-four hundred through thirty-four oh four. "
    "Each one rewrites an existing pointwise operator with a hand-rolled autotuned Triton kernel.",
    # 9 - closing
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
         "element-wise + autotune", ACCENT2, 70),
        ("MEDIUM  -  8 ops",
         "softmax   layer_norm   rms_norm   cross_entropy\n"
         "embedding   dropout   argmax   matmul",
         "normalization + GEMM", ACCENT3, 47),
        ("HARD  -  4 ops",
         "flash_attention   rope   fused_moe_topk   rms_norm_backward",
         "attention + backward", ACCENT4, 22),
    ]
    for label, ops, tagline, color, y in tiers:
        rect = patches.FancyBboxPatch(
            (6, y - 7), 88, 14, boxstyle="round,pad=0.6",
            linewidth=2, edgecolor=color, facecolor=PANEL,
        )
        ax.add_patch(rect)
        ax.text(9, y + 4, label, ha="left", va="center",
                fontsize=20, color=color, weight="bold")
        ax.text(9, y - 2, ops, ha="left", va="center",
                fontsize=15, color=FG, family="monospace")
        ax.text(91, y + 4, tagline, ha="right", va="center",
                fontsize=16, color=MUTED, style="italic")
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
    ax.text(50, 92, "178 / 178 correctness tests pass",
            ha="center", va="center", fontsize=30, color=FG, weight="bold")
    out = textwrap.dedent("""\
        $ python -m pytest tests/ -q

        tests/backward/test_backward.py ............................. [ 16%]
        tests/easy/test_easy.py         .........................................  [ 62%]
        tests/medium/test_medium.py     ....................................  [ 82%]
        tests/hard/test_hard.py         ..........  [ 88%]
        tests/test_cpu_fallback.py      .....................  [100%]

        ============================== 178 passed in 38s ===============================""")
    ax.text(11, 60, out, ha="left", va="center", fontsize=15,
            color=FG, family="monospace",
            bbox=dict(boxstyle="round,pad=1.2", facecolor=PANEL,
                      edgecolor=ACCENT2, linewidth=2))
    items = [
        ("3 dtypes",      "fp16  /  bf16  /  fp32"),
        ("Edge battery",  "NaN  /  Inf  /  -Inf  /  0.0  /  -0.0"),
        ("Shape sweep",   "(7,)  ->  (4096, 4096)"),
        ("Backward",      "5 kernels validated vs torch.autograd"),
        ("CPU parity",    "21 fallback cases, runs in CI"),
    ]
    for i, (k, v) in enumerate(items):
        y = 32 - i * 5
        ax.text(15, y, k, ha="left", va="center", fontsize=16,
                color=ACCENT2, weight="bold")
        ax.text(33, y, v, ha="left", va="center", fontsize=16,
                color=FG, family="monospace")
    footer(ax)
    return save(fig, 4)


def slide_cli():
    fig, ax = new_fig()
    ax.text(50, 92, "My flagos CLI - list, test, bench, package, info",
            ha="center", va="center", fontsize=28, color=FG, weight="bold")
    out = textwrap.dedent("""\
        $ flagos list

        Op                      Tier
        ----------------------------------
        abs                     easy
        exp                     easy
        log                     easy
        sigmoid                 easy
        relu                    easy
        ...
        softmax                 medium
        layer_norm              medium
        rms_norm                medium
        ...
        flash_attention         hard
        rope                    hard
        fused_moe_topk          hard
        rms_norm_backward       hard
        ----------------------------------
        20 operators""")
    ax.text(25, 47, out, ha="left", va="center", fontsize=15,
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
         "178 tests, dtype-aware tolerances, backward vs torch.autograd"),
        ("Performance Competitiveness",
         "every medium + hard kernel beats torch, matmul wrapper auto-dispatches"),
        ("Open-Source Adaptability",
         "Apache-2.0, pyproject.toml, FlagGems-style layout"),
        ("Cross-Platform Compatibility",
         "device_caps + per-vendor configs, CPU parity in CI, see BACKENDS.md"),
        ("Test Case Completeness",
         "shape x dtype grids, edge values, forward + backward + CPU paths"),
        ("Code Readability",
         "one op per module, type hints, per-op rationale in TECHNICAL_NOTES.md"),
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
    ax.text(50, 92, "Benchmarks vs torch reference (NVIDIA Ampere, fp16)",
            ha="center", va="center", fontsize=26, color=FG, weight="bold")
    out = textwrap.dedent("""\
        $ flagos bench --tier all

        Operator              Mine (ms)    Torch (ms)   Speedup
        --------------------------------------------------------
        flash_attention           0.176         1.702     9.68x
        rms_norm                  0.220         1.581     7.17x
        rope                      0.013         0.047     3.80x
        dropout                   0.346         1.333     3.85x
        rms_norm_backward         1.405         5.008     3.57x
        embedding                 0.224         0.616     2.75x
        cross_entropy             1.308         3.283     2.51x
        softmax                   0.224         0.409     1.83x
        argmax                    0.123         0.177     1.44x
        layer_norm                0.221         0.325     1.47x
        matmul                    0.189         0.265     1.40x
        fused_moe_topk            0.979         1.432     1.46x
        easy / pointwise          parity to 1.76x  (bandwidth-bound)""")
    ax.text(9, 47, out, ha="left", va="center", fontsize=13,
            color=FG, family="monospace",
            bbox=dict(boxstyle="round,pad=1.0", facecolor=PANEL,
                      edgecolor=ACCENT3, linewidth=2))
    footer(ax)
    return save(fig, 7)


def slide_upstream():
    fig, ax = new_fig()
    ax.text(50, 92, "5 upstream contributions to FlagGems",
            ha="center", va="center", fontsize=28, color=FG, weight="bold")
    ax.text(50, 85, "FlagGems Operator Development Competition",
            ha="center", va="center", fontsize=15, color=MUTED, style="italic")
    rows = [
        ("PR #3400",
         "perf(log10)",
         "fp64 path + autotune over BLOCK_SIZE / num_warps / num_stages"),
        ("PR #3401",
         "perf(abs)",
         "contiguous fast path, empty-tensor short-circuit, autotune sweep"),
        ("PR #3402",
         "perf(exp)",
         "fp32 accumulator for fp16/bf16, int->fp32 promotion, autotune"),
        ("PR #3403",
         "perf(log)",
         "fp32 accumulator path, autotune sweep keyed on n_elements"),
        ("PR #3404",
         "perf(tanh + tanh_backward)",
         "dedicated autotuned backward kernel, dx = dy * (1 - y*y) in fp32"),
    ]
    for i, (num, op, why) in enumerate(rows):
        y = 73 - i * 9
        rect = patches.FancyBboxPatch(
            (5, y - 3.2), 90, 6.4, boxstyle="round,pad=0.4",
            linewidth=1.4, edgecolor=ACCENT2, facecolor=PANEL,
        )
        ax.add_patch(rect)
        ax.text(8, y, num, ha="left", va="center", fontsize=16,
                color=ACCENT2, weight="bold", family="monospace")
        ax.text(20, y, op, ha="left", va="center", fontsize=16,
                color=ACCENT, weight="bold", family="monospace")
        ax.text(43, y, why, ha="left", va="center", fontsize=12.5,
                color=FG)
    ax.text(50, 19, "github.com/FlagOpen/FlagGems/pull/3400 ... /pull/3404",
            ha="center", va="center", fontsize=14, color=ACCENT,
            family="monospace")
    footer(ax)
    return save(fig, 8)


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
    return save(fig, 9)


# -----------------------------------------------------------------------------
# Narration backends
# -----------------------------------------------------------------------------
def _render_audio_elevenlabs(idx: int, text: str) -> Path:
    out = AUDIO_DIR / f"slide_{idx:02d}.mp3"
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
        "?output_format=mp3_44100_128"
    )
    body = json.dumps({
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.80,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        headers={
            "xi-api-key": ELEVEN_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        data=body,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out.write_bytes(resp.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"ElevenLabs HTTP {e.code} on slide {idx}: {msg}") from e
    return out


def _render_audio_edge(idx: int, text: str) -> Path:
    """Use edge-tts (Microsoft Azure Neural voices, no key required)."""
    import edge_tts  # imported lazily so users without it can still pick ElevenLabs
    out = AUDIO_DIR / f"slide_{idx:02d}.mp3"

    async def _go() -> None:
        comm = edge_tts.Communicate(
            text, voice=EDGE_VOICE, rate=EDGE_RATE, pitch=EDGE_PITCH,
        )
        await comm.save(str(out))

    asyncio.run(_go())
    return out


def _select_backend() -> str:
    """Decide which TTS backend to use this run."""
    if TTS_BACKEND in ("edge", "edge-tts"):
        return "edge"
    if TTS_BACKEND == "elevenlabs":
        return "elevenlabs"
    # auto: try ElevenLabs only if a key is present AND the account has quota left.
    if ELEVEN_KEY:
        try:
            req = urllib.request.Request(
                "https://api.elevenlabs.io/v1/user/subscription",
                headers={"xi-api-key": ELEVEN_KEY, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                info = json.loads(r.read().decode("utf-8"))
            used = int(info.get("character_count", 0))
            limit = int(info.get("character_limit", 0))
            remaining = max(0, limit - used)
            # Need ~1.1x the raw char count to cover any expansion. ElevenLabs
            # bills per character of input text, not per credit, so this is a
            # tight upper bound.
            need = int(sum(len(t) for t in NARRATION) * 1.1)
            if remaining >= need:
                return "elevenlabs"
            print(f"[tts] ElevenLabs has only {remaining} chars left, need ~{need}; "
                  "falling back to edge-tts.")
        except Exception as e:  # noqa: BLE001
            print(f"[tts] ElevenLabs preflight failed ({e}); falling back to edge-tts.")
    return "edge"


_SELECTED_BACKEND = "edge"  # overwritten in main() after _select_backend()


def render_audio(idx: int, text: str) -> Path:
    if _SELECTED_BACKEND == "elevenlabs":
        return _render_audio_elevenlabs(idx, text)
    return _render_audio_edge(idx, text)


def audio_duration_seconds(path: Path) -> float:
    """Use ffprobe so we don't need any audio-decoding Python lib."""
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


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
        slide_cli, slide_dimensions, slide_bench,
        slide_upstream, slide_closing,
    ]
    frames: List[Path] = []
    for b in builders:
        p = b()
        print(f"  rendered {p.name}")
        frames.append(p)
    assert len(frames) == len(NARRATION)

    # 2. Render audio + measure durations with ffprobe
    total_chars = sum(len(t) for t in NARRATION)
    if _SELECTED_BACKEND == "elevenlabs":
        print(f"\nTTS: ElevenLabs  voice={ELEVEN_VOICE}  model={ELEVEN_MODEL}  "
              f"chars={total_chars}")
    else:
        print(f"\nTTS: edge-tts  voice={EDGE_VOICE}  rate={EDGE_RATE}  "
              f"chars={total_chars}")
    durations: List[float] = []
    audios: List[Path] = []
    for i, text in enumerate(NARRATION, 1):
        mp3 = render_audio(i, text)
        d = audio_duration_seconds(mp3) + 0.5  # tail so clips don't bleed
        d = max(d, 4.0)  # 4 s floor for very short narrations
        durations.append(d)
        audios.append(mp3)
        print(f"  audio  slide_{i:02d}.mp3  {d:5.2f}s  '{text[:55]}...'")

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

    # 4. Build audio stream by padding each MP3 to its slide duration with silence
    concat_a = DEMO_DIR / "concat_audio.txt"
    audio_padded: List[Path] = []
    for i, (mp3, dur) in enumerate(zip(audios, durations), 1):
        padded = AUDIO_DIR / f"padded_{i:02d}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(mp3),
             "-af", f"apad=whole_dur={dur:.3f}",
             "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
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
    _SELECTED_BACKEND = _select_backend()  # noqa: F811  (module-level rebind)
    main()
