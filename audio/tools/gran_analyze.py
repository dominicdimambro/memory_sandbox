#!/usr/bin/env python3
"""
gran_analyze.py — comparative analysis of .gran granular synthesis banks

Usage:
    python gran_analyze.py file1.gran [file2.gran ...]

Output:
    • per-participant summary table  → stdout + summary.csv
    • pairwise spatial divergence    → stdout + divergence.csv
    • scatter_3d.png
    • distributions.png
    • chroma_profiles.png
    • divergence_heatmap.png         (only when ≥ 2 participants)

Dependencies: numpy, pandas, matplotlib, scipy
"""

import sys
import struct
import os
import math
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3d projection
from scipy.stats import gaussian_kde

# ──────────────────────────────────────────────────────────────────────────────
# Binary format
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 48_000

# GranHeader  — 84 bytes, little-endian, packed
#   char[4]  magic
#   uint32   version
#   uint32   slice_count
#   uint64   corpus_samples
#   char[64] name
HEADER_FMT  = "<4sIIQ64s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 84, f"header size {HEADER_SIZE} ≠ 84"

# GranSliceRec — 92 bytes, little-endian, packed
#   int32    id
#   uint64   corpus_start
#   uint32   length
#   SliceFeatures (7 scalar float32 + chroma_energy[12] float32):
#     float  rms
#     float  tonal_alignment_score
#     float  rolloff_freq
#     float  f0
#     float  pitch_confidence
#     float  spectral_flatness
#     float  gain
#     float  chroma_energy[12]
SLICE_FMT  = "<iQI7f12f"
SLICE_SIZE = struct.calcsize(SLICE_FMT)
assert SLICE_SIZE == 92, f"slice rec size {SLICE_SIZE} ≠ 92"

# Pitch-class names: chroma_energy[0]=A, [1]=A#/Bb … [11]=G#/Ab
_CHROMA_A  = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

# ──────────────────────────────────────────────────────────────────────────────
# Projection functions — mirror grain_visualizer.cpp / slice_store.h exactly
# ──────────────────────────────────────────────────────────────────────────────

def proj_x(tonal):
    return tonal * 0.5

def proj_y(rolloff):
    r = max(float(rolloff), 47.0)
    v = (math.log(r) - 3.850) / (9.393 - 3.850) - 0.5
    return max(-0.5, min(0.5, v))

def proj_z(rms):
    v = math.sqrt(max(float(rms), 0.0) / 0.12)
    return min(v, 1.0) - 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_gran(path: str) -> pd.DataFrame:
    """Return a DataFrame with one row per slice; raises on malformed input."""
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise ValueError("file too short for header")

        magic, version, slice_count, corpus_samples, raw_name = struct.unpack(HEADER_FMT, raw)
        if magic != b"GRAN":
            raise ValueError(f"bad magic {magic!r} (expected b'GRAN')")
        if version != 1:
            raise ValueError(f"unsupported version {version}")

        bank_name = raw_name.rstrip(b"\x00").decode("utf-8", errors="replace")

        # skip raw PCM — we only need features
        f.seek(int(corpus_samples) * 2, 1)   # int16 = 2 bytes

        rows = []
        for _ in range(slice_count):
            raw_rec = f.read(SLICE_SIZE)
            if len(raw_rec) < SLICE_SIZE:
                break
            vals = struct.unpack(SLICE_FMT, raw_rec)
            rec_id, corpus_start, length = vals[0], vals[1], vals[2]
            rms, tonal, rolloff, f0, pitch_conf, flatness, gain = vals[3:10]
            chroma = vals[10:]          # 12 floats, A-indexed

            row = {
                "id":                    rec_id,
                "corpus_start":          corpus_start,
                "length":                length,
                "duration_ms":           length / SAMPLE_RATE * 1000.0,
                "rms":                   rms,
                "tonal_alignment_score": tonal,
                "rolloff_freq":          rolloff,
                "f0":                    f0,
                "pitch_confidence":      pitch_conf,
                "spectral_flatness":     flatness,
                "gain":                  gain,
                # instrument projection
                "proj_x":               proj_x(tonal),
                "proj_y":               proj_y(rolloff),
                "proj_z":               proj_z(rms),
            }
            for ci, cv in enumerate(chroma):
                row[f"chroma_{_CHROMA_A[ci]}"] = cv

            rows.append(row)

    df = pd.DataFrame(rows)
    df["bank_name"]   = bank_name
    df["participant"] = os.path.splitext(os.path.basename(path))[0]
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

SUMMARY_METRICS = ["duration_ms", "rms", "f0", "spectral_flatness", "pitch_confidence"]

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, gdf in df.groupby("participant", sort=True):
        row: dict = {"participant": participant, "grain_count": len(gdf)}
        for col in SUMMARY_METRICS:
            row[f"{col}_mean"] = gdf[col].mean()
            row[f"{col}_std"]  = gdf[col].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("participant")

# ──────────────────────────────────────────────────────────────────────────────
# Pairwise spatial divergence
# ──────────────────────────────────────────────────────────────────────────────

def spatial_divergence(df: pd.DataFrame, n_sample: int = 3000) -> pd.DataFrame:
    """
    For each pair of participants: estimate mean Euclidean distance in 3-D
    projection space by sampling n_sample cross-participant pairs.
    """
    rng          = np.random.default_rng(42)
    participants = sorted(df["participant"].unique())
    pts          = {}
    for p in participants:
        arr = df[df["participant"] == p][["proj_x", "proj_y", "proj_z"]].values
        pts[p] = arr

    rows = []
    for p1, p2 in combinations(participants, 2):
        a, b = pts[p1], pts[p2]
        n = min(n_sample, len(a), len(b))
        if n == 0:
            rows.append({"participant_a": p1, "participant_b": p2, "mean_3d_dist": float("nan")})
            continue
        # independent random draws from each cloud → unbiased estimator of E[dist(X,Y)]
        ia = rng.choice(len(a), n, replace=len(a) < n)
        ib = rng.choice(len(b), n, replace=len(b) < n)
        d  = np.sqrt(((a[ia] - b[ib]) ** 2).sum(axis=1)).mean()
        rows.append({"participant_a": p1, "participant_b": p2, "mean_3d_dist": float(d)})

    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _colors(participants):
    cmap = plt.get_cmap("tab10")
    return {p: cmap(i % 10) for i, p in enumerate(sorted(participants))}

def _save(fig, name: str):
    fig.savefig(name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {name}")

# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: 3-D scatter in instrument projection space
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d(df: pd.DataFrame, colors: dict) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for participant, gdf in df.groupby("participant", sort=True):
        ax.scatter(
            gdf["proj_x"], gdf["proj_y"], gdf["proj_z"],
            label=participant,
            color=colors[participant],
            alpha=0.45, s=14, linewidths=0,
        )

    ax.set_xlabel("X  tonal (×0.5)")
    ax.set_ylabel("Y  rolloff (log-norm)")
    ax.set_zlabel("Z  rms (sqrt-norm)")
    ax.set_title("Grain cloud — instrument projection space", pad=12)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "scatter_3d.png")

# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: per-feature KDE distributions
# ──────────────────────────────────────────────────────────────────────────────

DIST_FEATURES = {
    "duration_ms":        "Duration  (ms)",
    "rms":                "RMS amplitude",
    "f0":                 "f₀  (Hz)",
    "spectral_flatness":  "Spectral flatness",
    "pitch_confidence":   "Pitch confidence",
}

def plot_distributions(df: pd.DataFrame, colors: dict) -> None:
    n   = len(DIST_FEATURES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, DIST_FEATURES.items()):
        for participant, gdf in df.groupby("participant", sort=True):
            vals = gdf[col].dropna().values
            c    = colors[participant]
            if len(vals) < 3 or (vals.max() - vals.min()) < 1e-10:
                ax.axvline(vals.mean() if len(vals) else 0,
                           color=c, label=participant, linewidth=1.5)
                continue
            xs  = np.linspace(vals.min(), vals.max(), 400)
            kde = gaussian_kde(vals, bw_method="scott")
            ax.plot(xs, kde(xs), color=c, label=participant, linewidth=1.8)
            ax.fill_between(xs, kde(xs), alpha=0.12, color=c)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(col, fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    fig.suptitle("Per-feature distributions (KDE)", fontsize=11)
    _save(fig, "distributions.png")

# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: mean chroma energy profiles
# ──────────────────────────────────────────────────────────────────────────────

def plot_chroma(df: pd.DataFrame, colors: dict) -> None:
    participants = sorted(df["participant"].unique())
    n_p      = len(participants)
    bar_w    = 0.80 / max(n_p, 1)
    x        = np.arange(12)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    for i, participant in enumerate(participants):
        gdf = df[df["participant"] == participant]
        means = np.array([gdf[f"chroma_{_CHROMA_A[j]}"].mean() for j in range(12)])
        offset = (i - n_p / 2 + 0.5) * bar_w
        ax.bar(x + offset, means, width=bar_w,
               label=participant, color=colors[participant], alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(_CHROMA_A, fontsize=9)
    ax.set_xlabel("Pitch class")
    ax.set_ylabel("Mean normalized chroma energy")
    ax.set_title("Mean chroma energy profile per participant")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    _save(fig, "chroma_profiles.png")

# ──────────────────────────────────────────────────────────────────────────────
# Plot 4: divergence heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_divergence_heatmap(diverg: pd.DataFrame) -> None:
    participants = sorted(set(diverg["participant_a"]) | set(diverg["participant_b"]))
    n = len(participants)
    if n < 2:
        return

    idx = {p: i for i, p in enumerate(participants)}
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 0.0)
    for _, row in diverg.iterrows():
        i, j = idx[row["participant_a"]], idx[row["participant_b"]]
        mat[i, j] = mat[j, i] = row["mean_3d_dist"]

    fig, ax = plt.subplots(figsize=(max(4, n + 1), max(3, n)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(participants, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(participants, fontsize=8)
    fig.colorbar(im, ax=ax, label="mean 3-D Euclidean distance")
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            label = f"{v:.3f}" if not math.isnan(v) else "—"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if v > (np.nanmax(mat) * 0.6) else "black")
    ax.set_title("Pairwise spatial divergence\n(mean Euclidean dist in projection space)")
    _save(fig, "divergence_heatmap.png")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    dfs = []
    for path in sys.argv[1:]:
        print(f"Parsing  {path}")
        try:
            df = parse_gran(path)
            if df.empty:
                print(f"  warning: 0 slices found — skipping")
                continue
            print(f"  {len(df):4d} slices   bank='{df['bank_name'].iloc[0]}'")
            dfs.append(df)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    if not dfs:
        print("No valid .gran files loaded.", file=sys.stderr)
        sys.exit(1)

    df_all       = pd.concat(dfs, ignore_index=True)
    participants = sorted(df_all["participant"].unique())
    colors       = _colors(participants)

    # ── summary table ─────────────────────────────────────────
    summary = build_summary(df_all)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n" + "=" * 80)
    print("PER-PARTICIPANT SUMMARY")
    print("=" * 80)
    print(summary.to_string())
    print()
    summary.to_csv("summary.csv")
    print("  → summary.csv")

    # ── spatial divergence ────────────────────────────────────
    if len(participants) >= 2:
        diverg = spatial_divergence(df_all)
        print("\nPAIRWISE SPATIAL DIVERGENCE")
        print(diverg.to_string(index=False, float_format="{:.4f}".format))
        diverg.to_csv("divergence.csv", index=False)
        print("  → divergence.csv")
    else:
        diverg = pd.DataFrame()

    # ── plots ─────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_3d(df_all, colors)
    plot_distributions(df_all, colors)
    plot_chroma(df_all, colors)
    if not diverg.empty:
        plot_divergence_heatmap(diverg)

    print("\nDone.")


if __name__ == "__main__":
    main()
