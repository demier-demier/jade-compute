#!/usr/bin/env python3
"""Rebuild Jade's original vector branding and evidence-based benchmark figures.

Dependencies: Python 3, matplotlib, plus rsvg-convert or cairosvg for SVG previews.
No benchmark is executed and no source report is changed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAVY = "#07141d"
PANEL = "#0c2029"
INK = "#edf8f4"
MUTED = "#94aaa9"
JADE = "#68eab0"
AMBER = "#e9bc70"
STROKE = "#244048"
FONT = "Arial, Helvetica, sans-serif"


def normalized_text(text: str) -> str:
    """Keep generated text reviewable: LF endings, no trailing whitespace."""
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def svg_start(width: int, height: int, title: str, description: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{title}</title><desc id="desc">{description}</desc>
<defs>
 <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#18323b" stroke-width="0.6" opacity="0.48"/></pattern>
 <linearGradient id="jade" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#a1f5ce"/><stop offset="1" stop-color="#46cb98"/></linearGradient>
 <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1 1L9 5L1 9" fill="none" stroke="#8baba9" stroke-width="1.4"/></marker>
</defs>
<rect width="{width}" height="{height}" fill="{NAVY}"/>
<g font-family="{FONT}">'''


def hero(height: int = 480) -> str:
    offset = (height - 480) // 2
    body = svg_start(1280, height, "Jade Compute — GPU compute, made inspectable.",
        "Public research prototype. Rust runtime, native Metal, and CoreX research. Original geometric Jade monogram.")
    body += f'''
<rect x="862" width="418" height="{height}" fill="url(#grid)"/>
<path d="M862 0V{height}" stroke="#16313a"/>
<g transform="translate(0 {offset})">
 <path d="M72 66H98" stroke="{JADE}" stroke-width="3"/>
 <text x="111" y="71" fill="{MUTED}" font-size="12" letter-spacing="2.6">INDEPENDENT COMPUTE RESEARCH</text>
 <text x="70" y="187" fill="{INK}" font-size="66" font-weight="700" letter-spacing="-1.7">JADE COMPUTE</text>
 <text x="73" y="246" fill="{INK}" font-size="29" letter-spacing="-0.5">GPU compute, made inspectable.</text>
 <text x="74" y="292" fill="{MUTED}" font-size="18">Rust runtime · Native Metal · CoreX research</text>
 <rect x="74" y="337" width="249" height="30" rx="5" fill="#112b2c" stroke="#275248"/>
 <circle cx="90" cy="352" r="3" fill="{JADE}"/>
 <text x="103" y="356" fill="#b9d8ce" font-size="10.5" font-weight="700" letter-spacing="1.35">PUBLIC RESEARCH PROTOTYPE</text>
 <path d="M74 422H808" stroke="{STROKE}"/>
 <text x="74" y="445" fill="#6e9295" font-size="11" letter-spacing="1.6">SOURCE · EXECUTION · EVIDENCE</text>
 <g transform="translate(0 0)">
  <path d="M967 137H1132V278L1060 350H978L926 298V243H974V277L998 301H1039L1084 256V185H967Z" fill="#143a39" transform="translate(12 12)"/>
  <path d="M967 137H1132V278L1060 350H978L926 298V243H974V277L998 301H1039L1084 256V185H967Z" fill="url(#jade)"/>
  <path d="M967 137H1132V278L1060 350" fill="none" stroke="#d4ffe9" stroke-width="1.2" opacity="0.7"/>
  <path d="M919 111H940M929.5 100.5V121.5M1153 377H1174M1163.5 366.5V387.5" stroke="#4a786f" stroke-width="1"/>
  <circle cx="1164" cy="112" r="3" fill="#31584f"/>
  <text x="1025" y="412" fill="#628e86" font-size="10" letter-spacing="3.8">JADE / 01</text>
 </g>
</g></g></svg>'''
    return body


def architecture() -> str:
    body = svg_start(1000, 480, "Jade runtime architecture and backend status",
        "Rust host code owns buffers, compiles kernels and submits ordered batches. Checked KernelSpec IR lowers to native Apple Metal, executed locally, and Iluvatar CoreX source, prepared but not executed on target hardware.")
    body += f'''
<text x="48" y="51" fill="{INK}" font-size="25" font-weight="700" letter-spacing="-0.5">One checked IR. Explicit backends.</text>
<text x="49" y="79" fill="{MUTED}" font-size="14">Native execution and validation status remain visible.</text>
<rect x="48" y="146" width="250" height="251" rx="12" fill="{PANEL}" stroke="{STROKE}"/>
<text x="72" y="178" fill="{JADE}" font-size="10" font-weight="700" letter-spacing="1.8">RUST HOST</text>
<text x="71" y="212" fill="{INK}" font-size="25" font-weight="700">Jade runtime</text>
<path d="M72 234H274" stroke="{STROKE}"/>
<circle cx="78" cy="265" r="4" fill="{JADE}"/>
<text x="94" y="270" fill="#d7e8e3" font-size="16">Own device buffers</text>
<circle cx="78" cy="310" r="4" fill="{JADE}"/>
<text x="94" y="315" fill="#d7e8e3" font-size="16">Compile kernels</text>
<circle cx="78" cy="355" r="4" fill="{JADE}"/>
<text x="94" y="360" fill="#d7e8e3" font-size="16">Submit ordered batches</text>
<path d="M298 271H347" fill="none" stroke="#8baba9" stroke-width="1.6" marker-end="url(#arrow)"/>
<rect x="356" y="202" width="203" height="136" rx="12" fill="#10272c" stroke="#3e6863"/>
<text x="378" y="231" fill="{JADE}" font-size="10" font-weight="700" letter-spacing="1.8">CHECKED IR</text>
<text x="377" y="268" fill="{INK}" font-size="25" font-weight="700">KernelSpec</text>
<text x="378" y="300" fill="{MUTED}" font-size="12.5">Bounds · bindings · aliases</text>
<path d="M559 271H594M594 186V355M594 186H631M594 355H631" fill="none" stroke="#8baba9" stroke-width="1.6"/>
<path d="M622 186H631" fill="none" stroke="#8baba9" stroke-width="1.6" marker-end="url(#arrow)"/>
<path d="M622 355H631" fill="none" stroke="#8baba9" stroke-width="1.6" marker-end="url(#arrow)"/>
<circle cx="594" cy="271" r="3" fill="#8baba9"/>
<rect x="640" y="119" width="312" height="136" rx="12" fill="#0e2927" stroke="#376955"/>
<text x="663" y="145" fill="#82a99b" font-size="9.5" font-weight="700" letter-spacing="1.5">NATIVE BACKEND</text>
<rect x="839" y="132" width="92" height="22" rx="4" fill="#214938"/>
<text x="850" y="147" fill="{JADE}" font-size="9.5" font-weight="700" letter-spacing="0.7">EXECUTED</text>
<text x="662" y="183" fill="{INK}" font-size="25" font-weight="700">Apple Metal</text>
<text x="663" y="212" fill="#bdd8cb" font-size="14">MSL · native command buffers</text>
<text x="663" y="236" fill="#82a99b" font-size="12.5">Validated locally on Apple M4 Pro</text>
<rect x="640" y="285" width="312" height="136" rx="12" fill="#27251f" stroke="#6c5836"/>
<text x="663" y="311" fill="#b1a188" font-size="9.5" font-weight="700" letter-spacing="1.5">RESEARCH BACKEND</text>
<rect x="798" y="298" width="133" height="22" rx="4" fill="#413721"/>
<text x="809" y="313" fill="{AMBER}" font-size="9" font-weight="700" letter-spacing="0.5">SOURCE PREPARED</text>
<text x="662" y="349" fill="{INK}" font-size="25" font-weight="700">Iluvatar CoreX</text>
<text x="663" y="378" fill="#d7c7ac" font-size="14">CoreX C++ · vendor compiler</text>
<text x="663" y="402" fill="#b1a188" font-size="12.5">SDK build &amp; GPU execution pending</text>
<text x="49" y="456" fill="#799494" font-size="12.5">Status as delivered · Metal results do not establish CoreX support.</text>
</g></svg>'''
    return body


def render_svg(svg: Path, png: Path, width: int, height: int) -> str:
    converter = shutil.which("rsvg-convert")
    if converter:
        subprocess.run([converter, "--width", str(width), "--height", str(height),
                        "--output", str(png), str(svg)], check=True)
        return "rsvg-convert"
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError("PNG previews require rsvg-convert or cairosvg") from exc
    cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=width, output_height=height)
    return "cairosvg"


def chart(source: Path, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    raw = source.read_bytes()
    report = json.loads(raw)
    series = {}
    repeat_counts = set()
    for key in ("batched_samples", "separate_submission_samples"):
        samples = report[key]
        if len(samples) != 7:
            raise ValueError(f"Expected seven measured samples in {key}; inspect changed report before republishing")
        values = []
        for item in samples:
            if item["repeats"] <= 0 or item["dispatches"] != item["repeats"] * len(report["specs"]):
                raise ValueError("Inconsistent repeat/dispatch count in source report")
            value = item["host_ms"] / item["repeats"]
            repeat_counts.add(item["repeats"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid measured host time")
            values.append(value)
        series[key] = values
    batched = series["batched_samples"]
    separate = series["separate_submission_samples"]
    batch_median, separate_median = statistics.median(batched), statistics.median(separate)
    if not math.isclose(batch_median, report["median_batched_host_ms_per_block"], rel_tol=1e-12):
        raise ValueError("Batched median does not agree with report")
    if not math.isclose(separate_median, report["median_separate_host_ms_per_block"], rel_tol=1e-12):
        raise ValueError("Separate-submission median does not agree with report")
    if report["runtime"]["backend"] != "native-metal" or len(report["specs"]) != 6:
        raise ValueError("This figure's labels require the native Metal, six-operation workload")
    if report.get("trained_model") is not False:
        raise ValueError("This figure is scoped to a synthetic workload")
    if len(repeat_counts) != 1:
        raise ValueError("Figure footnote requires an equal repeat count across samples")
    repeats = next(iter(repeat_counts))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "path", "svg.hashsalt": "jade-compute-evidence-v1", "axes.facecolor": NAVY,
        "figure.facecolor": NAVY, "savefig.facecolor": NAVY,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": STROKE,
    })
    fig = plt.figure(figsize=(12.8, 7.6), dpi=100)
    ax = fig.add_axes((0.105, 0.235, 0.67, 0.52))
    x = list(range(1, 8))
    ax.plot(x, separate, color=AMBER, linewidth=1.8, marker="o", markersize=7,
            markeredgecolor=NAVY, markeredgewidth=1.5, zorder=4)
    ax.plot(x, batched, color=JADE, linewidth=1.8, marker="o", markersize=7,
            markeredgecolor=NAVY, markeredgewidth=1.5, zorder=4)
    for value, color, label in ((separate_median, AMBER, "PER-DISPATCH SYNC"),
                                (batch_median, JADE, "ORDERED BATCH")):
        ax.axhline(value, color=color, linewidth=1, linestyle=(0, (4, 5)), alpha=0.65)
        ax.text(7.36, value + 0.065, f"{value:.4f} ms", color=color, fontsize=17,
                fontweight="bold", va="bottom", clip_on=False)
        ax.text(7.36, value - 0.012, "MEDIAN · " + label, color=MUTED, fontsize=8,
                va="top", clip_on=False)
    ax.set_xlim(0.7, 7.13)
    ax.set_ylim(0, max(separate) * 1.1)
    ax.set_xticks(x)
    ax.set_xlabel("Sample order", labelpad=13, fontsize=11)
    ax.set_ylabel("Host wall time per block (ms)", labelpad=18, fontsize=11)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(axis="y", color=STROKE, linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9)

    fig.text(0.059, 0.931, "JADE COMPUTE / MEASURED EXECUTION", color=JADE,
             fontsize=9, fontweight="bold")
    fig.text(0.058, 0.867, "Submission overhead · synthetic MLP", fontsize=25,
             fontweight="bold")
    fig.text(0.059, 0.821, "Seven raw samples per mode · identical native Metal kernels · lower is better",
             color=MUTED, fontsize=12)
    fig.text(0.105, 0.779, "●  Ordered batch", color=JADE, fontsize=11)
    fig.text(0.292, 0.779, "●  Synchronize every dispatch", color=AMBER, fontsize=11)

    device = report["runtime"]["device"]
    fig.text(0.059, 0.125, f"{device}  ·  FP32  ·  6 operations  ·  fixed resident tensors  ·  {repeats} blocks per sample",
             color="#c2d6cf", fontsize=11)
    fig.text(0.059, 0.089, "Excludes compilation, allocation, upload and readback. Raw samples retain the observed timing drift.",
             color=MUTED, fontsize=10)
    fig.text(0.059, 0.055, "Synthetic workload; not a CUDA, vendor-BLAS or complete-model comparison.",
             color=MUTED, fontsize=10)

    title = "Submission overhead · synthetic MLP"
    provenance = f"Source: {source.name}; SHA-256: {hashlib.sha256(raw).hexdigest()}; host_ms/repeats per sample."
    fig.savefig(out / "benchmark-submission.svg", metadata={"Title": title, "Description": provenance,
                "Creator": "scripts/render-presentation.py", "Date": None})
    svg_path = out / "benchmark-submission.svg"
    svg_path.write_text(normalized_text(svg_path.read_text(encoding="utf-8")), encoding="utf-8")
    fig.savefig(out / "benchmark-submission.png", dpi=100, metadata={"Title": title,
                "Description": provenance, "Software": "scripts/render-presentation.py / matplotlib"})
    plt.close(fig)
    with (out / "benchmark-samples.csv").open("w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["sample", "ordered_batch_host_ms_per_block", "per_dispatch_sync_host_ms_per_block"])
        for sample, batch, sync in zip(x, batched, separate):
            writer.writerow([sample, format(batch, ".12g"), format(sync, ".12g")])
    if source.read_bytes() != raw:
        raise RuntimeError("Source report changed during rendering; rerun before publication")

    # Matplotlib embeds DejaVu glyph outlines in the SVG. Preserve the complete
    # font license supplied by this installed distribution alongside the figure.
    font_license_path = Path(matplotlib.get_data_path()) / "fonts/ttf/LICENSE_DEJAVU"
    font_license = font_license_path.read_bytes()
    notice = """# Third-party notices

`benchmark-submission.svg` contains DejaVu Sans glyph outlines, including the
bold face, converted to vector paths by Matplotlib. The corresponding chart
PNG renders the same typefaces. Copyright and licensing rights in those glyphs
remain with their respective holders; the Jade project license does not
supersede the font license below.

Source: the installed Matplotlib distribution's
`mpl-data/fonts/ttf/LICENSE_DEJAVU`. The complete notice is reproduced below;
only line endings and trailing whitespace are normalized.

```text
""" + normalized_text(font_license.decode("utf-8")) + "```\n"
    (out / "THIRD_PARTY_NOTICES.md").write_text(notice, encoding="utf-8")
    return {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_unix_time": report.get("unix_time"),
        "calculation": "Each plotted value is raw sample host_ms divided by its repeats; seven values per mode.",
        "median_ordered_batch_ms": batch_median,
        "median_per_dispatch_sync_ms": separate_median,
        "scope": report["scope"],
        "device": device,
        "benchmark_reexecuted": False,
        "embedded_font": {
            "family": "DejaVu Sans",
            "notice": "assets/THIRD_PARTY_NOTICES.md",
            "license_source": "matplotlib/mpl-data/fonts/ttf/LICENSE_DEJAVU",
            "license_source_sha256": hashlib.sha256(font_license).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/native-metal-mlp.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    visuals = {"hero": (hero(), 1280, 480),
               "social-preview": (hero(640), 1280, 640),
               "architecture": (architecture(), 1000, 480)}
    renderer = None
    for name, (content, width, height) in visuals.items():
        svg, png = out / f"{name}.svg", out / f"{name}.png"
        svg.write_text(normalized_text(content), encoding="utf-8")
        renderer = render_svg(svg, png, width, height)
    provenance = chart(source, out)
    provenance["vector_preview_renderer"] = renderer
    provenance["generator"] = "scripts/render-presentation.py"
    provenance["assets"] = {
        "hero": {"width": 1280, "height": 480},
        "social-preview": {"width": 1280, "height": 640},
        "architecture": {"width": 1000, "height": 480},
        "benchmark-submission": {"width": 1280, "height": 760},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for item in sorted(out.iterdir()):
        if item.is_file():
            print(f"{item.name}: {item.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
