#!/usr/bin/env python3
"""Render final Metal GPU-timing evidence without executing a benchmark.

Requires matplotlib. Normal runs read only reports/metal-advances and write the
three metal-advances assets. An alternative source directory is allowed only
with --validate-only, so development samples cannot silently become final art.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/metal-advances"
OUTPUT = ROOT / "assets"
NAVY, PANEL = "#07141d", "#0c2029"
INK, MUTED, STROKE = "#edf8f4", "#94aaa9", "#244048"
JADE, AMBER, BLUE = "#68eab0", "#e9bc70", "#83b8ef"
INPUTS = (
    ("gemm-512.json", "gemm", {"m": 512, "n": 512, "k": 512}),
    ("gemm-1024.json", "gemm", {"m": 1024, "n": 1024, "k": 1024}),
    ("attention-256.json", "attention", {"heads": 4, "sequence": 256, "head_dim": 64, "causal": False}),
    ("attention-512.json", "attention", {"heads": 4, "sequence": 512, "head_dim": 64, "causal": False}),
    ("attention-causal-512.json", "attention", {"heads": 4, "sequence": 512, "head_dim": 64, "causal": True}),
)
GEMM = (
    ("jade_register_tiled", "Jade register", AMBER),
    ("jade_simdgroup_fp32", "Jade SIMD-group", JADE),
    ("apple_mps_fp32", "Apple MPS", BLUE),
)
ATTENTION = (
    ("materialized_reference", "Scalar materialized", AMBER),
    ("online_simd", "Online SIMD", JADE),
)


def normalized_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_reports(directory: Path) -> list[dict]:
    """Validate figure-specific assumptions and derive every point from GPU time."""
    result = []
    for filename, family, shape in INPUTS:
        path = directory / filename
        raw = path.read_bytes()
        report = json.loads(raw)
        where = f"{filename}: "
        require(report.get("schema_version") == 1, where + "unexpected schema")
        require(report.get("passed") is True, where + "experiment did not pass")
        require(report.get("shape") == shape, where + "shape differs from chart labels")
        require(report.get("runtime", {}).get("device") == "Apple M4 Pro", where + "unexpected device")
        require(report["runtime"].get("backend") == "native-metal", where + "unexpected backend")
        require(report.get("experiment") == ("native_fp32_gemm" if family == "gemm" else "native_fp32_self_attention"), where + "unexpected experiment")
        if family == "gemm":
            require("packed row strides" in report.get("mps_baseline", ""), where + "MPS layout changed")
            require("No AllowReducedPrecision" in report["mps_baseline"], where + "MPS precision changed")
            require(report.get("precision", "").startswith("FP32 inputs, output and accumulation"), where + "precision changed")
        else:
            require(report.get("zero_qk") is False and report.get("input_amplitude") == 1.0, where + "input experiment changed")
        definitions = GEMM if family == "gemm" else ATTENTION
        expected_names = {name for name, _, _ in definitions}
        variants = report["variants"]
        require(len(variants) == len(definitions), where + "variant count changed")
        require({v["name"] for v in variants} == expected_names, where + "variant names changed")
        require(len(report["sample_order"]) == 9, where + "expected nine rounds")
        for order in report["sample_order"]:
            require(len(order) == len(expected_names) and set(order) == expected_names, where + "invalid round order")
        series = {}
        output_count = shape["m"] * shape["n"] if family == "gemm" else shape["heads"] * shape["sequence"] * shape["head_dim"]
        for variant in variants:
            name = variant["name"]
            for stage in ("initial_check", "final_check"):
                check = variant[stage]
                require(check.get("all_finite") is True and check["elements"] == output_count, where + name + " incomplete validation")
            samples = variant["timing"]["samples"]
            require(len(samples) == 9, where + name + " expected nine samples")
            values, repeats = [], []
            dispatch_factor = 3 if name == "materialized_reference" else 1
            for sample in samples:
                repeat = sample["repeats"]
                gpu = sample.get("gpu_ms")
                require(isinstance(repeat, int) and repeat > 0, where + name + " invalid repeat count")
                require(sample["dispatches"] == dispatch_factor * repeat, where + name + " invalid dispatch count")
                require(isinstance(gpu, (int, float)) and math.isfinite(gpu) and gpu > 0, where + name + " GPU time unavailable/invalid")
                # Divide by complete operations, never by the baseline's three
                # dispatches. Host timestamps are intentionally never read.
                values.append(gpu / repeat)
                repeats.append(repeat)
            median = statistics.median(values)
            require(math.isclose(median, variant["timing"]["median_gpu_ms"], rel_tol=1e-10, abs_tol=1e-12), where + name + " median disagrees with raw samples")
            series[name] = {"gpu_ms_per_operation": values, "median_gpu_ms": median, "repeats": repeats}
        result.append({
            "filename": filename, "family": family, "shape": shape,
            "source_path": path, "source_bytes": raw,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_unix_time": report.get("unix_time"), "scope": report["scope"],
            "series": series,
        })
    return result


def draw_panel(ax, reports: list[dict], definitions: tuple, labels: list[str], title: str, subtitle: str) -> None:
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    centers = np.arange(len(reports)) * 1.5
    offsets = np.linspace(-0.32, 0.32, len(definitions)) if len(definitions) == 3 else [-0.22, 0.22]
    jitter = np.linspace(-0.07, 0.07, 9)
    maximum = max(value for report in reports for item in report["series"].values() for value in item["gpu_ms_per_operation"])
    for center, report in zip(centers, reports):
        for offset, (name, _, color) in zip(offsets, definitions):
            item = report["series"][name]
            position, median = center + offset, item["median_gpu_ms"]
            ax.scatter(position + jitter, item["gpu_ms_per_operation"], s=24, color=color,
                       edgecolor=NAVY, linewidth=0.55, alpha=0.85, zorder=3)
            ax.plot([position - 0.105, position + 0.105], [median, median],
                    color=NAVY, linewidth=5, solid_capstyle="round", zorder=4)
            ax.plot([position - 0.105, position + 0.105], [median, median],
                    color=color, linewidth=2.4, solid_capstyle="round", zorder=5)
            ax.annotate(f"{median:.3f}", (position, median), xytext=(0, 11),
                        textcoords="offset points", ha="center", va="bottom",
                        color=color, fontsize=10, fontweight="bold", zorder=6,
                        bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 1.5, "alpha": 0.85})
    ax.set_xlim(-0.65, centers[-1] + 0.65)
    ax.set_ylim(0, maximum * 1.19)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("GPU milliseconds per operation", fontsize=11, labelpad=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.grid(axis="y", color=STROKE, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.text(0, 1.25, title, transform=ax.transAxes, fontsize=18, fontweight="bold", color=INK)
    ax.text(0, 1.16, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED)
    handles = [Line2D([0], [0], marker="o", color=color, label=label, markersize=5, linewidth=1.6)
               for _, label, color in definitions]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.015, 1.015),
              frameon=False, ncol=len(definitions), fontsize=9.4,
              handlelength=1.4, columnspacing=1.2, labelcolor=INK)


def render(reports: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = SOURCE / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    for report in reports:
        require(manifest["reports"].get(report["filename"]) == report["source_sha256"],
                "Source report does not match the finalized benchmark manifest")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "path",
        "svg.hashsalt": "jade-metal-advances-v1", "axes.facecolor": PANEL,
        "figure.facecolor": NAVY, "savefig.facecolor": NAVY, "text.color": INK,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": STROKE,
    })
    fig = plt.figure(figsize=(16, 9.4), dpi=120)
    left = fig.add_axes((0.075, 0.325, 0.385, 0.38))
    right = fig.add_axes((0.565, 0.325, 0.385, 0.38))
    draw_panel(left, reports[:2], GEMM, ["512 × 512 × 512", "1024 × 1024 × 1024"],
               "Dense matrix multiplication", "M × N × K  ·  C = A × B")
    draw_panel(right, reports[2:], ATTENTION, ["S = 256", "S = 512", "S = 512\ncausal"],
               "Self-attention forward", "4 heads  ·  head dimension 64  ·  scale 1/√64")
    fig.text(0.045, 0.95, "JADE COMPUTE / MEASURED EXECUTION", color=JADE, fontsize=10, fontweight="bold")
    fig.text(0.044, 0.897, "Native Metal: kernel comparisons", fontsize=29, fontweight="bold")
    fig.text(0.045, 0.852, "Apple M4 Pro · FP32 · resident tensors · native GPU timestamps · lower is better", fontsize=13, color=MUTED)
    fig.text(0.075, 0.225, "Each dot is one of nine raw samples. Short lines and labels show medians, in milliseconds.", fontsize=12, color=INK)
    fig.text(0.075, 0.19, "GPU time / complete-operation repeats; compilation, allocation, upload and readback are excluded.", fontsize=10.5, color=MUTED)
    fig.text(0.075, 0.153, "GEMM baseline: MPSMatrixMultiplication, packed rows, Float32, reduced precision disabled; other MPS layouts may differ.", fontsize=10.1, color=MUTED)
    fig.text(0.075, 0.118, "Attention baseline: this project's scalar QK → softmax → PV; not optimized FlashAttention, MLX or a complete model.", fontsize=10.1, color=MUTED)
    fig.text(0.075, 0.083, "These measurements compare implementations on one Mac; they do not establish NVIDIA or Chinese-GPU performance.", fontsize=10.1, color=MUTED)
    for x, label, url in (
        (0.075, "Data: reports/metal-advances/", "../reports/metal-advances/"),
        (0.39, "Hashes + medians: metal-advances-provenance.json", "metal-advances-provenance.json"),
        (0.805, "DejaVu font notice", "THIRD_PARTY_NOTICES.md"),
    ):
        fig.text(x, 0.038, label, fontsize=8.8, color=JADE, url=url)
    title = "Jade native Metal GPU timings: GEMM and attention"
    description = "Nine raw GPU-time samples per implementation with medians; Apple M4 Pro, FP32 resident operations. Sources: reports/metal-advances. Provenance: assets/metal-advances-provenance.json."
    svg, png = OUTPUT / "metal-advances.svg", OUTPUT / "metal-advances.png"
    fig.savefig(svg, metadata={"Title": title, "Description": description, "Creator": "scripts/render-metal-results.py", "Date": None})
    svg.write_text(normalized_text(svg.read_text(encoding="utf-8")), encoding="utf-8")
    fig.savefig(png, dpi=120, metadata={"Title": title, "Description": description, "Software": "scripts/render-metal-results.py / matplotlib"})
    plt.close(fig)
    for report in reports:
        require(report["source_path"].read_bytes() == report["source_bytes"], "A source report changed while rendering; regenerate from the finalized reports")
    require(manifest_path.read_bytes() == manifest_raw, "Benchmark manifest changed while rendering")
    font_license = Path(matplotlib.get_data_path()) / "fonts/ttf/LICENSE_DEJAVU"
    require((OUTPUT / "THIRD_PARTY_NOTICES.md").is_file(), "Existing DejaVu font notice is required")
    provenance = {
        "generator": "scripts/render-metal-results.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__, "benchmark_reexecuted": False,
        "device": "Apple M4 Pro", "precision": "FP32", "metric": "GPU milliseconds per complete operation",
        "calculation": "Each point = sample.gpu_ms / sample.repeats; nine points per variant, median computed from all nine; no host times used.",
        "sample_count": sum(len(item["gpu_ms_per_operation"]) for r in reports for item in r["series"].values()),
        "source_manifest": {"path": "reports/metal-advances/manifest.json", "sha256": hashlib.sha256(manifest_raw).hexdigest(), "source_commit": manifest["source_commit"]},
        "sources": [{"path": "reports/metal-advances/" + r["filename"], "sha256": r["source_sha256"],
                     "unix_time": r["source_unix_time"], "shape": r["shape"], "scope": r["scope"], "series": r["series"]} for r in reports],
        "font": {"family": "DejaVu Sans", "notice": "assets/THIRD_PARTY_NOTICES.md",
                 "license_source": "matplotlib/mpl-data/fonts/ttf/LICENSE_DEJAVU", "license_sha256": hashlib.sha256(font_license.read_bytes()).hexdigest()},
        "assets": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (svg, png)},
        "png_dimensions": [1920, 1128],
    }
    (OUTPUT / "metal-advances-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Check input data without writing any assets")
    parser.add_argument("--source-dir", type=Path, default=SOURCE, help="Alternative inputs are accepted only with --validate-only")
    args = parser.parse_args()
    directory = args.source_dir.resolve()
    require(args.validate_only or directory == SOURCE.resolve(), "Rendering requires finalized reports/metal-advances inputs")
    reports = read_reports(directory)
    if args.validate_only:
        print(f"Validated {len(reports)} reports; no assets written.")
        return
    render(reports)
    print("Rendered assets/metal-advances.svg, .png and -provenance.json from finalized reports.")


if __name__ == "__main__":
    main()
