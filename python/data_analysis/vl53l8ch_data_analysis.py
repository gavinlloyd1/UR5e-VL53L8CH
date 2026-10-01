"""
vl53l8ch_data_analysis.py
-------------------------
Analysis utilities for VL53L8CH master "wide" CSV files.

Features
- Region presets on the 8×8 zone grid:
    region='inner36'  -> centered 6×6 (rows 1..6, cols 1..6) → zones rows: 9-14,17-22,25-30,33-38,41-46,49-54
    region='inner16'  -> centered 4×4 (rows 2..5, cols 2..5)
    region='inner4'   -> centered 2×2 (rows 3..4, cols 3..4)
    region='all'      -> all 64 zones (default)
  You may also pass explicit zones=[...]; the final set is the INTERSECTION of the
  preset region and your explicit list.

- CNH bin–sum analysis (precise terminology):
    We use “sum of CNH bin values” for bin-summed quantities.
    * heatmap(...)                          → 8×8 map of sum bins at a location
    * total_cnh_bin_sum_per_location(...)   → sum bins vs. location (optionally by region)
    * cnh_histograms_for_location(...)      → overlay CNH histograms for many zones at one location (legend outside)
    * cnh_histograms_for_zone(...)          → overlay CNH histograms for one zone across many locations (legend outside)
    * compare_region_bin_sums_per_location(...) → single plot comparing 64/36/16/4 zone presets

- Signal-strength analysis (per-zone, averaged over frames):
    * heatmap_signal_strength(...)              → 8×8 map of average signal at a location
    * total_signal_strength_per_location(...)   → summed per-zone average signal vs. location
    * compare_region_signal_strength_per_location(...) → single plot comparing 64/36/16/4 zone presets

Notes
- “save” can be a file path (ending in .png/.pdf) or a directory; if directory, a default filename is used.
- “show=True” displays figures; otherwise plots are closed after saving.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Region helpers (8×8 zones, row-major indexing 0..63)
# --------------------------------------------------------------------

def _zones_for_region(region: str) -> List[int]:
    """Return a list of zone indices for a named region on an 8×8 grid (row-major)."""
    region = (region or "all").lower()
    if region == "all":
        return list(range(64))

    def build_square(r0: int, r1: int, c0: int, c1: int) -> List[int]:
        out = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out.append(r * 8 + c)
        return out

    if region == "inner36":  # 6×6, centered: rows 1..6, cols 1..6
        return build_square(1, 6, 1, 6)
    if region == "inner16":  # 4×4, centered: rows 2..5, cols 2..5
        return build_square(2, 5, 2, 5)
    if region == "inner4":   # 2×2, centered: rows 3..4, cols 3..4
        return build_square(3, 4, 3, 4)

    raise ValueError(f"Unknown region '{region}'. Use one of: all, inner36, inner16, inner4")


def _apply_region_and_zones(candidate_zones: Iterable[int], *, region: str = "all",
                            zones: Optional[Iterable[int]] = None) -> List[int]:
    """Compute final zone list by intersecting a region preset with an optional explicit list."""
    rset: Set[int] = set(_zones_for_region(region))
    if zones is None:
        final = sorted([z for z in candidate_zones if z in rset])
    else:
        zset: Set[int] = set(int(z) for z in zones)
        final = sorted([z for z in candidate_zones if (z in rset and z in zset)])
    return final


# --------------------------------------------------------------------
# Data container and preprocessing
# --------------------------------------------------------------------

@dataclass
class AnalysisState:
    input_csv: Path
    df: pd.DataFrame                 # raw wide dataframe
    long: pd.DataFrame               # long CNH: movement_value, zone, bin, value
    avg_loc_zone_bin: pd.DataFrame   # avg over frames: movement_value, zone, bin, value
    sum_loc_zone: pd.DataFrame       # sum bins per (movement_value, zone)
    signal_loc_zone: pd.DataFrame    # average signal strength per (movement_value, zone)
    movement_values: list            # sorted unique movement_value list
    zones: list                      # sorted unique zone list
    bins: list                       # sorted unique bin list


def _extract_cnh_long(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form CNH table with columns: movement_value, zone, bin, value."""
    if "movement_value" not in df.columns:
        raise ValueError("Expected column 'movement_value' in master CSV.")
    pat = re.compile(r"^cnh__hist_bin_(\d+)_a(\d+)$")
    mv = df["movement_value"].values
    frames = []
    for c in df.columns:
        m = pat.match(c)
        if not m:
            continue
        bin_idx = int(m.group(1))
        zone = int(m.group(2))
        frames.append(pd.DataFrame({
            "movement_value": mv,
            "zone": zone,
            "bin": bin_idx,
            "value": df[c].values
        }))
    if not frames:
        raise ValueError("No CNH histogram columns found (expected cnh__hist_bin_{bin}_a{zone}).")
    long = pd.concat(frames, ignore_index=True)
    return long


def _average_over_frames(long: pd.DataFrame) -> pd.DataFrame:
    """Average CNH over frames for each (movement_value, zone, bin)."""
    return (long
            .groupby(["movement_value", "zone", "bin"], as_index=False)["value"]
            .mean())


def _sum_per_zone_location(avg_loc_zone_bin: pd.DataFrame) -> pd.DataFrame:
    """Sum over bins to get sum bins per (movement_value, zone)."""
    return (avg_loc_zone_bin
            .groupby(["movement_value", "zone"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "sum_bins"}))


def _discover_signal_strength_columns(df: pd.DataFrame) -> dict[int, str]:
    """
    Discover per-zone signal strength columns.
    Returns {zone_index: column_name}. Raises ValueError if none found.

    Supports both zone suffix styles:
      - ..._a{zone}  (e.g., signal_per_spad_a53)
      - ..._z{zone}  (e.g., signal_per_spad_z53)  <-- your case
    And multiple naming families from EVK dumps.
    """
    candidates: dict[int, str] = {}

    # Prefer specific, known-good patterns first
    patterns = [
        # signal_per_spad (kcps optional)
        re.compile(r'^(?:cnh__)?signal_per_spad(?:_kcps)?_(?:a|z)(\d+)$', re.IGNORECASE),

        # generic "signal"/"signal_strength"
        re.compile(r'^(?:cnh__)?signal(?:_strength)?_(?:a|z)(\d+)$', re.IGNORECASE),
        re.compile(r'^sig(?:nal)?(?:_strength)?_(?:a|z)(\d+)$', re.IGNORECASE),

        # peak signal rate / signal kcps variants
        re.compile(r'^(?:cnh__)?peak_signal(?:_rate)?(?:_kcps(?:_spad)?)?_(?:a|z)(\d+)$', re.IGNORECASE),
        re.compile(r'^(?:cnh__)?signal_(?:kcps|kcps_spad|rate_kcps|rate_kcps_spad)_(?:a|z)(\d+)$', re.IGNORECASE),

        # Allow double-underscore around the suffix in some exports
        re.compile(r'^(?:cnh__)?signal(?:_strength)?__?(?:a|z)(\d+)$', re.IGNORECASE),
        re.compile(r'^sig(?:nal)?(?:_strength)?__?(?:a|z)(\d+)$', re.IGNORECASE),
    ]

    for col in df.columns:
        for pat in patterns:
            m = pat.match(col)
            if m:
                z = int(m.group(1))
                candidates.setdefault(z, col)  # keep first match per zone
                break

    # Fallback: any "*signal*" field ending with _a{zone} or _z{zone},
    # excluding things that are clearly not signal.
    if not candidates:
        generic = re.compile(r'.*signal.*_(?:a|z)(\d+)$', re.IGNORECASE)
        exclude = re.compile(r'ambient|sigma|distance|spad(?:s|_count)?', re.IGNORECASE)
        for col in df.columns:
            if exclude.search(col):
                continue
            m = generic.match(col)
            if m:
                z = int(m.group(1))
                candidates.setdefault(z, col)

    if not candidates:
        hints = [c for c in df.columns if re.search(r'_(?:a|z)\d+$', c)]
        hints = hints[:20]
        raise ValueError(
            "No per-zone signal strength columns found. "
            "Looked for variants like 'signal_per_spad_z0', 'signal_strength_a0', "
            "'peak_signal_rate_kcps_z0', etc. "
            f"Example zone-suffixed columns seen: {hints}. "
            "Update _discover_signal_strength_columns(...) patterns if needed."
        )
    return candidates


def _avg_signal_strength_per_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average signal strength per (movement_value, zone).
    Returns DataFrame with columns ['movement_value','zone','signal'].
    """
    sig_cols = _discover_signal_strength_columns(df)
    mv = df["movement_value"].values
    frames = []
    for zone, col in sig_cols.items():
        frames.append(pd.DataFrame({
            "movement_value": mv,
            "zone": int(zone),
            "signal": df[col].values
        }))
    long = pd.concat(frames, ignore_index=True)
    avg = (long.groupby(["movement_value", "zone"], as_index=False)["signal"].mean())
    return avg


def load_analysis(input_csv: str | Path) -> AnalysisState:
    """Load and preprocess a master wide CSV for CNH analysis."""
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)

    long = _extract_cnh_long(df)
    avg = _average_over_frames(long)
    sum_lz = _sum_per_zone_location(avg)
    sig_lz = _avg_signal_strength_per_location(df)

    movement_values = sorted(avg["movement_value"].unique().tolist())
    zones = sorted(avg["zone"].unique().tolist())
    bins = sorted(avg["bin"].unique().tolist())

    return AnalysisState(
        input_csv=input_csv,
        df=df,
        long=long,
        avg_loc_zone_bin=avg,
        sum_loc_zone=sum_lz,
        signal_loc_zone=sig_lz,
        movement_values=movement_values,
        zones=zones,
        bins=bins,
    )


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

def _resolve_save_path(save: Optional[str | Path], default_filename: str, default_dir: Path) -> Optional[Path]:
    if save is None:
        return None
    save = Path(save)
    if save.is_dir() or (str(save).endswith(("/", "\\")) and not str(save).lower().endswith(('.png', '.pdf'))):
        default_dir = save
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / default_filename
    save.parent.mkdir(parents=True, exist_ok=True)
    return save


def _tag_for_region_zones(region: str = "all", zones: Optional[Iterable[int]] = None) -> str:
    """
    Build a short filename tag from region/zones so that zone-filtered plots don't
    overwrite region-only plots when 'save' points to a directory.
    """
    if zones:
        zs = sorted({int(z) for z in zones})
        if len(zs) == 1:
            return f"z{zs[0]}"
        if len(zs) <= 8:
            return "z" + "_".join(str(z) for z in zs)
        return f"z{zs[0]}-{zs[-1]}_{len(zs)}zones"
    return (region or "all")


def pick_nearest_movement_value(an: AnalysisState, value: float):
    """Return the nearest available movement_value to 'value' in the dataset."""
    return min(an.movement_values, key=lambda x: abs(x - float(value)))


# --------------------------------------------------------------------
# CNH bin–sum plots
# --------------------------------------------------------------------

def heatmap(
    an: AnalysisState,
    movement_value: float | int,
    *,
    region: str = "all",
    zones: Optional[Iterable[int]] = None,
    save: Optional[str | Path] = None,
    show: bool = False,
):
    """
    8×8 heatmap of the **sum of CNH bin values** for a given movement_value.
    """
    mv = pick_nearest_movement_value(an, movement_value)
    zslice = an.sum_loc_zone[an.sum_loc_zone["movement_value"] == mv].copy()
    allowed = set(_apply_region_and_zones(an.zones, region=region, zones=zones))
    zslice = zslice[zslice["zone"].isin(allowed)]

    img = np.full((8, 8), np.nan)
    for _, r in zslice.iterrows():
        y, x = divmod(int(r["zone"]), 8)
        img[y, x] = r["sum_bins"]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img, aspect="equal")
    fig.colorbar(im, ax=ax, label="Sum of CNH bin values")
    ax.set_title(f"CNH Bin Sum Heatmap @ {mv}  [{region}]")
    ax.set_xlabel("X (zone column)")
    ax.set_ylabel("Y (zone row)")
    fig.tight_layout()

    tag = _tag_for_region_zones(region, zones)
    outpath = _resolve_save_path(save, f"cnh_bin_sum_heatmap_{mv}_{tag}.png", an.input_csv.parent / "analysis")
    if outpath:
        fig.savefig(outpath, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, mv


def total_cnh_bin_sum_per_location(
    an: AnalysisState,
    *,
    region: str = "all",
    zones: Optional[Iterable[int]] = None,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> pd.DataFrame:
    """
    Sum CNH bins vs. location for the selected region/zones.
    Returns DataFrame ['movement_value', 'sum_bins'].
    """
    df = an.sum_loc_zone
    allowed = set(_apply_region_and_zones(an.zones, region=region, zones=zones))
    df = df[df["zone"].isin(allowed)]
    sum_loc = df.groupby("movement_value", as_index=False)["sum_bins"].sum()

    if save is not None or show:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sum_loc["movement_value"], sum_loc["sum_bins"], marker="o")
        ax.set_xlabel("Location (movement_value)")
        ax.set_ylabel("Total sum of CNH bin values")
        ax.set_title( f"Total CNH Bin Sum per Location [{'zones=' + ','.join(map(str, zones)) if zones is not None else region}]")
        fig.tight_layout()
        tag = _tag_for_region_zones(region, zones)
        outpath = _resolve_save_path(save, f"cnh_bin_sum_per_location_{tag}.png", an.input_csv.parent / "analysis")
        if outpath:
            fig.savefig(outpath, dpi=160)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return sum_loc


def cnh_histograms_for_location(
    an: AnalysisState,
    movement_value: float,
    *,
    region: str = "all",
    zones: Optional[Iterable[int]] = None,
    normalize: bool = False,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> pd.DataFrame:
    """
    Overlay CNH histograms for multiple zones at a single location.
    Legend is placed outside (below) to avoid covering lines.
    """
    mv = pick_nearest_movement_value(an, movement_value)
    slice_loc = an.avg_loc_zone_bin[an.avg_loc_zone_bin["movement_value"] == mv]
    allowed = set(_apply_region_and_zones(an.zones, region=region, zones=zones))
    slice_loc = slice_loc[slice_loc["zone"].isin(allowed)]

    labels = []
    fig, ax = plt.subplots(figsize=(10, 4))
    for z in sorted(slice_loc["zone"].unique().tolist()):
        d = slice_loc[slice_loc["zone"] == z].sort_values("bin")
        y = d["value"].values
        if normalize:
            s = np.sum(y)
            if s > 0:
                y = y / s
        ax.plot(d["bin"].values, y, alpha=0.35, linewidth=1.0, label=f"z{z}")
        labels.append(f"z{z}")

    ax.set_xlabel("CNH bin")
    ax.set_ylabel("CNH value" + (" (normalized per curve)" if normalize else ""))
    ax.set_title(f"CNH at Location {mv}  [{region}]")
    if labels:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=min(8, max(1, len(labels)//4)), fontsize=8, frameon=False)
        fig.subplots_adjust(bottom=0.25)
    fig.tight_layout()

    tag = _tag_for_region_zones(region, zones)
    outpath = _resolve_save_path(save, f"cnh_at_location_{mv}_{tag}.png", an.input_csv.parent / "analysis")
    if outpath:
        fig.savefig(outpath, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return slice_loc.copy()


def cnh_histograms_for_zone(
    an: AnalysisState,
    zone: int,
    *,
    locations: Optional[Iterable[float]] = None,
    normalize: bool = False,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> pd.DataFrame:
    """
    Overlay CNH histograms for a single zone across selected (or all) locations.
    Legend is placed outside (below).
    """
    zslice = an.avg_loc_zone_bin[an.avg_loc_zone_bin["zone"] == int(zone)]
    if locations is None:
        locs = an.movement_values
    else:
        locs = [pick_nearest_movement_value(an, v) for v in locations]

    wide = zslice.pivot(index="bin", columns="movement_value", values="value").sort_index()
    cols = [c for c in wide.columns if c in set(locs)]
    wide = wide[cols]

    fig, ax = plt.subplots(figsize=(10, 4))
    for mv_val in wide.columns:
        y = wide[mv_val].values
        if normalize:
            s = np.sum(y)
            if s > 0:
                y = y / s
        ax.plot(wide.index.values, y, alpha=0.8, linewidth=1.2, label=str(mv_val))

    ax.set_xlabel("CNH bin")
    ax.set_ylabel("CNH value" + (" (normalized per curve)" if normalize else ""))
    ttl = f"Zone {zone}: CNH Histograms across Locations"
    if normalize:
        ttl += " (normalized)"
    ax.set_title(ttl)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(6, max(1, len(wide.columns)//2)), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    outpath = _resolve_save_path(save, f"zone{zone}_histograms_across_locations.png", an.input_csv.parent / "analysis")
    if outpath:
        fig.savefig(outpath, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return wide


def compare_region_bin_sums_per_location(
    an: AnalysisState,
    *,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> pd.DataFrame:
    """
    One plot comparing sum CNH bins per location for 64/36/16/4 zone presets.
    Returns tidy DataFrame ['movement_value','sum_bins','region'].
    """
    regions: List[Tuple[str, str]] = [
        ("all", "64 zones"),
        ("inner36", "36 zones"),
        ("inner16", "16 zones"),
        ("inner4", "4 zones"),
    ]
    frames = []
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, label in regions:
        df = total_cnh_bin_sum_per_location(an, region=key, save=None, show=False)
        tmp = df.copy()
        tmp["region"] = label
        frames.append(tmp)
        ax.plot(df["movement_value"], df["sum_bins"], marker="o", label=label)

    ax.set_xlabel("Location (movement_value)")
    ax.set_ylabel("Total sum of CNH bin values")
    ax.set_title("Total sum CNH bins per location: region comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    outpath = _resolve_save_path(save, "cnh_bin_sum_per_location_regions.png", an.input_csv.parent / "analysis")
    if outpath:
        fig.savefig(outpath, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------
# Signal-strength plots (per-zone averages over frames)
# --------------------------------------------------------------------

def heatmap_signal_strength(
    an: AnalysisState,
    movement_value: float | int,
    *,
    region: str = "all",
    zones: Optional[Iterable[int]] = None,
    save: Optional[str | Path] = None,
    show: bool = False,
):
    """
    8×8 heatmap of **average signal strength** per zone for a given movement_value.
    """
    mv = pick_nearest_movement_value(an, movement_value)
    zslice = an.signal_loc_zone[an.signal_loc_zone["movement_value"] == mv].copy()
    allowed = set(_apply_region_and_zones(an.zones, region=region, zones=zones))
    zslice = zslice[zslice["zone"].isin(allowed)]

    img = np.full((8, 8), np.nan)
    for _, r in zslice.iterrows():
        y, x = divmod(int(r["zone"]), 8)
        img[y, x] = r["signal"]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img, aspect="equal")
    fig.colorbar(im, ax=ax, label="Average signal strength")
    ax.set_title(f"Signal Strength Heatmap @ {mv}  [{region}]")
    ax.set_xlabel("X (zone column)")
    ax.set_ylabel("Y (zone row)")
    fig.tight_layout()

    tag = _tag_for_region_zones(region, zones)
    outpath = _resolve_save_path(save, f"signal_strength_heatmap_{mv}_{tag}.png", an.input_csv.parent / "analysis")
    if outpath:
        fig.savefig(outpath, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, mv


def total_signal_strength_per_location(
    an: AnalysisState,
    *,
    region: str = "all",
    zones: Optional[Iterable[int]] = None,
    save: Optional[str | Path] = None,
    show: bool = False,
    transition_detection: Optional[dict | bool] = None,
    transition_kwargs: Optional[dict] = None,
    plot_smoothed: bool = False,           # <— overlay smoothed values (optional)
    smooth_window: int = 5,                # <— smoothing window for overlay
    smooth_method: str = "median",         # <— "median" or "mean"
) -> pd.DataFrame:
    """
    Total **average signal strength** vs. location (sums per-zone averages across selected zones).
    Returns DataFrame ['movement_value','signal'].

    Args (optional):
        plot_smoothed: If True, overlay the smoothed series used for detection-style views.
        smooth_window: Rolling window (odd int recommended) for the overlay smoothing.
        smooth_method: 'median' (robust, default) or 'mean' for the overlay smoothing.
        transition_detection: If truthy, draws offset-yield style bounds using detect_transition_bounds_offset_yield.
    """
    df = an.signal_loc_zone
    allowed = set(_apply_region_and_zones(an.zones, region=region, zones=zones))
    df = df[df["zone"].isin(allowed)]
    sig_loc = df.groupby("movement_value", as_index=False)["signal"].sum()

    if save is not None or show:
        fig, ax = plt.subplots(figsize=(8, 4))

        # Raw summed signal
        ax.plot(sig_loc["movement_value"], sig_loc["signal"], marker="o", label="Raw")

        # Optional: smoothed overlay for visibility (uses same helper as detector)
        if plot_smoothed:
            x_vals = sig_loc["movement_value"].values
            y_vals = sig_loc["signal"].values
            y_s    = _rolling_smooth(y_vals, window=smooth_window, method=smooth_method)
            ax.plot(x_vals, y_s, linewidth=2, alpha=0.8,
                    label=f"Smoothed ({smooth_method}, w={smooth_window})")

        # Optional: compute and overlay transition bounds (offset-yield method)
        if transition_detection:
            try:
                params = transition_kwargs or {}
                det = detect_transition_bounds_offset_yield(
                    sig_loc["movement_value"].values,
                    sig_loc["signal"].values,
                    **params
                )
                if det:
                    th_lo = det["theta_lo"]; th_hi = det["theta_hi"]; W = det["width"]
                    ax.axvline(th_lo, linestyle="--", color="tab:red", alpha=0.7, label="Lower bound")
                    ax.axvline(th_hi, linestyle="--", color="tab:green", alpha=0.7, label="Upper bound")
                    ax.fill_betweenx([sig_loc["signal"].min(), sig_loc["signal"].max()],
                                     th_lo, th_hi, alpha=0.08, color="tab:blue")
                    # annotate width
                    midy = 0.05 * (sig_loc["signal"].max() - sig_loc["signal"].min()) + sig_loc["signal"].min()
                    ax.annotate(f"width ≈ {W:.2f}°",
                                xy=((th_lo+th_hi)/2, midy), xytext=(0, -20),
                                textcoords="offset points", ha="center", va="top",
                                bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.6))
            except Exception:
                pass

        ax.set_xlabel("Location (movement_value)")
        ax.set_ylabel("Total average signal strength")
        ax.set_title(f"Total Signal Strength per Location [{'zones=' + ','.join(map(str, zones)) if zones is not None else region}]")
        ax.legend(loc="best")
        fig.tight_layout()

        tag = _tag_for_region_zones(region, zones)
        outpath = _resolve_save_path(save, f"signal_strength_per_location_{tag}.png", an.input_csv.parent / "analysis")
        if outpath:
            fig.savefig(outpath, dpi=160)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return sig_loc


def compare_region_signal_strength_per_location(
    an: AnalysisState,
    *,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> pd.DataFrame:
    """
    One plot comparing total average signal strength per location for 64/36/16/4 zone presets.
    Returns tidy DataFrame ['movement_value','signal','region'].
    """
    regions: List[Tuple[str, str]] = [
        ("all", "64 zones"),
        ("inner36", "36 zones"),
        ("inner16", "16 zones"),
        ("inner4", "4 zones"),
    ]
    frames = []
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, label in regions:
        df = total_signal_strength_per_location(an, region=key, save=None, show=False)
        tmp = df.copy()
        tmp["region"] = label
        frames.append(tmp)
        ax.plot(df["movement_value"], df["signal"], marker="o", label=label)

    ax.set_xlabel("Location (movement_value)")
    ax.set_ylabel("Total average signal strength")
    ax.set_title("Total signal strength per location: region comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    outpath = _resolve_save_path(save, "signal_strength_per_location_regions.png", an.input_csv.parent / "analysis")
    if outpath:
        fig.savefig(outpath, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return pd.concat(frames, ignore_index=True)


def export_roll_cnh_heatmaps(
    an: AnalysisState,
    expected_positions=(180, 90, 0, -90),   # matches your roll_stepper signs
    *,
    region="all",
    zones=None
):
    """
    Save one CNH bin-sum heatmap for each roll location.
    Each expected angle is snapped to the nearest movement_value in the CSV.
    """
    # Snap + dedupe while preserving order
    ordered = []
    for v in expected_positions:
        mv = pick_nearest_movement_value(an, v)
        if mv not in ordered:
            ordered.append(mv)

    out_dir = an.input_csv.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mv in ordered:
        heatmap(
            an,
            movement_value=mv,
            region=region,
            zones=zones,
            save=out_dir,   # directory → auto filename: cnh_bin_sum_heatmap_{mv}_{tag}.png
            show=False,
        )
    return ordered


def export_roll_signal_heatmaps(
    an: AnalysisState,
    expected_positions=(180, 90, 0, -90),   # matches your roll_stepper signs
    *,
    region="all",
    zones=None
):
    """
    Save one signal-strength heatmap for each roll location.
    Each expected angle is snapped to the nearest movement_value in the CSV.
    """
    # Snap + dedupe while preserving order
    ordered = []
    for v in expected_positions:
        mv = pick_nearest_movement_value(an, v)
        if mv not in ordered:
            ordered.append(mv)

    out_dir = an.input_csv.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mv in ordered:
        heatmap_signal_strength(
            an,
            movement_value=mv,
            region=region,
            zones=zones,
            save=out_dir,   # directory → auto filename: signal_strength_heatmap_{mv}_{tag}.png
            show=False,
        )
    return ordered



# --------------------------------------------------------------------
# Transition detection (offset-yield style)
# --------------------------------------------------------------------

def _rolling_smooth(y: np.ndarray, window: int = 5, method: str = "median") -> np.ndarray:
    """Lightweight smoothing using pandas rolling window; returns np.ndarray of same length."""
    s = pd.Series(y, dtype="float64")
    if window is None or window <= 1:
        return s.values.astype(float)
    if method == "mean":
        out = s.rolling(window=window, center=True, min_periods=1).mean()
    else:
        out = s.rolling(window=window, center=True, min_periods=1).median()
    return out.values.astype(float)


def _robust_line_fit(x: np.ndarray, y: np.ndarray, *, max_iter: int = 3, mad_thresh: float = 3.5):
    """
    Simple robust line fit: iterate least-squares with MAD-based outlier rejection.
    Returns (slope, intercept, sigma, mask) where sigma is 1.4826 * MAD of residuals after final fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 2:
        return 0.0, float(np.nanmean(y) if y.size else 0.0), float("nan"), mask

    keep = np.ones_like(x, dtype=bool)
    a = b = 0.0
    for _ in range(max_iter):
        if keep.sum() < 2:
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
        resid = y - (a * x + b)
        med = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - med))
        scale = 1.4826 * mad if mad > 0 else np.std(resid[keep]) if keep.sum()>1 else 0.0
        if scale <= 0:
            break
        new_keep = np.abs(resid - med) <= mad_thresh * scale
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep
    # final sigma
    resid = y - (a * x + b)
    med = np.median(resid[keep]) if keep.any() else 0.0
    mad = np.median(np.abs(resid[keep] - med)) if keep.any() else 0.0
    sigma = 1.4826 * mad if mad > 0 else (np.std(resid[keep]) if keep.sum()>1 else 0.0)
    return a, b, sigma, keep


def detect_transition_bounds_offset_yield(
    x: np.ndarray,
    y: np.ndarray,
    *,
    left_window: Tuple[float, float] = (-25.0, -5.0),
    right_window: Tuple[float, float] = (5.0, 25.0),
    k: float = 3.0,
    r: float = 0.03,
    min_streak: int = 3,
    smooth_window: int = 5,
    smooth_method: str = "median",
    debug: bool = False,
):
    """
    Data-driven transition detection inspired by offset-yield:
      1) Fit robust lines on low and high 'flat' windows
      2) Choose vertical offset delta = max(k*noise, r*step)
      3) Lower bound = first forward crossing of (left baseline + delta)
         Upper bound = first backward crossing of (right baseline - delta)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sort by x to ensure monotone traversal
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Optional smoothing (for detection only)
    y_s = _rolling_smooth(y, window=smooth_window, method=smooth_method)

    # Window masks for baseline fits
    Lmask = (x >= left_window[0]) & (x <= left_window[1])
    Rmask = (x >= right_window[0]) & (x <= right_window[1])

    if Lmask.sum() < max(5, min_streak) or Rmask.sum() < max(5, min_streak):
        return ({"reason": "insufficient_window_points",
                 "L_count": int(Lmask.sum()), "R_count": int(Rmask.sum())}
                if debug else None)

    # Robust linear fits on the two plateaus
    aL, bL, sigmaL, _ = _robust_line_fit(x[Lmask], y_s[Lmask])
    aR, bR, sigmaR, _ = _robust_line_fit(x[Rmask], y_s[Rmask])

    # Step estimate via predictions at window centers (robust to mild slopes)
    xL_mid = 0.5 * (left_window[0] + left_window[1])
    xR_mid = 0.5 * (right_window[0] + right_window[1])
    yL_mid = aL * xL_mid + bL
    yR_mid = aR * xR_mid + bR
    step = yR_mid - yL_mid

    # Vertical offset: noise-aware with small fractional backstop
    noise = np.sqrt((sigmaL**2 + sigmaR**2) / 2.0)
    delta = max(k * noise, r * abs(step))

    # ---------- Lower bound (forward crossing of left baseline + delta) ----------
    start_idx = np.searchsorted(x, left_window[1], side="left")

    def residual_L(idx: int) -> float:
        return y_s[idx] - (aL * x[idx] + bL)

    lo_idx = None
    streak = 0
    for i in range(start_idx, len(x)):
        if residual_L(i) > delta:
            streak += 1
            if streak >= min_streak:
                lo_idx = i - (min_streak - 1)
                break
        else:
            streak = 0

    if lo_idx is None or lo_idx <= 0:
        return ({"reason": "no_forward_crossing",
                 "delta": float(delta),
                 "max_residual_L": float(np.max(y_s[start_idx:] - (aL * x[start_idx:] + bL))) if start_idx < len(x) else float("nan")}
                if debug else None)

    # Interpolate to the exact crossing with residual == delta
    i0 = max(lo_idx - 1, 0)
    r0 = residual_L(i0)
    r1 = residual_L(lo_idx)
    if r1 == r0:
        theta_lo = x[lo_idx]
    else:
        t = (delta - r0) / (r1 - r0)
        theta_lo = x[i0] + t * (x[lo_idx] - x[i0])

    # ---------- Upper bound (backward crossing of right baseline - delta) ----------
    end_idx = np.searchsorted(x, left_window[1], side="left")

    def residual_R(idx: int) -> float:
        return (aR * x[idx] + bR) - y_s[idx]

    hi_idx = None
    streak = 0
    for i in range(len(x) - 1, end_idx - 1, -1):
        if residual_R(i) > delta:
            streak += 1
            if streak >= min_streak:
                hi_idx = i
                break
        else:
            streak = 0

    if hi_idx is None or hi_idx >= len(x):
        return ({"reason": "no_backward_crossing",
                 "delta": float(delta),
                 "max_residual_R": float(np.max((aR * x[end_idx:] + bR) - y_s[end_idx:])) if end_idx < len(x) else float("nan")}
                if debug else None)

    # Interpolate to exact crossing
    j1 = min(hi_idx + 1, len(x) - 1)
    r0 = residual_R(hi_idx)
    r1 = residual_R(j1)
    if r1 == r0:
        theta_hi = x[hi_idx]
    else:
        t = (delta - r0) / (r1 - r0)
        theta_hi = x[hi_idx] + t * (x[j1] - x[hi_idx])

    width = theta_hi - theta_lo
    if not np.isfinite(width) or width <= 0:
        return ({"reason": "nonpositive_width",
                 "theta_lo": float(theta_lo), "theta_hi": float(theta_hi)}
                if debug else None)

    return {
        "theta_lo": float(theta_lo),
        "theta_hi": float(theta_hi),
        "width": float(width),
        "delta": float(delta),
        "step": float(step),
        "sigma_L": float(sigmaL),
        "sigma_R": float(sigmaR),
        "left_fit": (float(aL), float(bL)),
        "right_fit": (float(aR), float(bR)),
        "x_sorted": x.tolist(),
        "y_smooth": y_s.tolist(),
    }



# --------------------------------------------------------------------
# Sigmoid helpers — Gaussian CDF (lmfit)
# --------------------------------------------------------------------

# These helpers provide an alternative sigmoid model based on the Gaussian CDF (erf),
# and use the 'lmfit' package for non-linear least-squares. They are kept separate
# so you can compare different edge models and fitting toolchains.

try:
    from lmfit import Model
except Exception:
    Model = None

from scipy.special import erf, erfinv


def gauss_cdf(x, x0, sigma, y_lo, y_hi):
    """
    Gaussian-CDF (error-function) ramp:
        y(x) = y_lo + (y_hi - y_lo) * 0.5 * (1 + erf((x - x0) / (sqrt(2) * sigma)))
    """
    x = np.asarray(x, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-12)
    t = (x - x0) / (np.sqrt(2.0) * sigma)
    return y_lo + (y_hi - y_lo) * 0.5 * (1.0 + erf(t))


def _initial_guess_gauss_cdf(x, y):
    """
    Robust initial guesses for (x0, sigma, y_lo, y_hi) that work for
    both increasing *and* decreasing edges.

    Strategy:
      • plateaus from medians of the first/last ~10% of samples (by x)
      • smoothed series → find all mid-level crossings; pick the one
        with the largest |dy/dx| (the real transition)
      • local 10–90 crossings around that same edge → sigma
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 5:
        return dict(x0=float(np.nanmedian(x) if x.size else 0.0),
                    sigma=float(np.nanstd(x) or 1.0),
                    y_lo=float(np.nanmin(y) if y.size else 0.0),
                    y_hi=float(np.nanmax(y) if y.size else 1.0))

    # sort by x
    o = np.argsort(x); x, y = x[o], y[o]

    # plateau levels from ends
    k = max(3, int(0.10 * len(y)))
    yL = float(np.nanmedian(y[:k]))
    yR = float(np.nanmedian(y[-k:]))
    y_hi = max(yL, yR)
    y_lo = min(yL, yR)

    # smoothed signal and mid level
    y_s = _rolling_smooth(y, window=5, method="median")
    mid = 0.5 * (y_hi + y_lo)

    # find crossings of mid level
    d = y_s - mid
    cross_idx = np.where(d[:-1] * d[1:] <= 0)[0]
    if cross_idx.size == 0:
        # fallback: closest point to mid
        i = int(np.nanargmin(np.abs(d)))
        x0 = float(x[i])
        # crude width from global range
        C = 2.0 * np.sqrt(2.0) * float(erfinv(0.8))
        sigma = (np.nanstd(x) or 1.0) / 4.0
        return dict(x0=x0, sigma=float(max(1e-6, sigma)), y_lo=y_lo, y_hi=y_hi)

    # pick the crossing with largest local |dy/dx|
    g = np.abs(np.gradient(y_s, x))
    best = int(cross_idx[np.argmax(g[cross_idx + 1])])

    # interpolate x at mid level
    def _x_at_level(i, level):
        x0i, x1i = x[i], x[i+1]
        y0i, y1i = y_s[i], y_s[i+1]
        return float(x0i + (level - y0i) * (x1i - x0i) / (y1i - y0i + 1e-12))

    x0 = _x_at_level(best, mid)

    # 10–90 levels and local crossings in a small window around `best`
    y10 = y_lo + 0.10 * (y_hi - y_lo)
    y90 = y_lo + 0.90 * (y_hi - y_lo)
    j0 = max(0, best - 10)
    j1 = min(len(x) - 2, best + 10)

    def _local_cross(level):
        dd = (y_s[j0:j1+2] - level)
        jj = np.where(dd[:-1] * dd[1:] <= 0)[0]
        if jj.size:
            j = int(jj[np.argmin(np.abs(dd[jj]))])  # nearest segment
            return _x_at_level(j0 + j, level)
        # fallback: global interpolation on smoothed curve
        return float(np.interp(level, y_s, x))

    x10 = _local_cross(y10)
    x90 = _local_cross(y90)
    lo, hi = (min(x10, x90), max(x10, x90))

    # sigma from 10–90 width
    C = 2.0 * np.sqrt(2.0) * float(erfinv(0.8))
    sigma = max(1e-6, (hi - lo) / C)

    return dict(x0=float(x0), sigma=float(sigma), y_lo=float(y_lo), y_hi=float(y_hi))


def _x_filter_mask(x: np.ndarray,
                   *,
                   x_range: Optional[Tuple[float, float]] = None,
                   exclude: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """Build a boolean mask to keep only desired x-range and drop excluded intervals."""
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if x_range is not None:
        lo, hi = float(x_range[0]), float(x_range[1])
        m &= (x >= lo) & (x <= hi)
    if exclude:
        for a, b in exclude:
            a, b = float(a), float(b)
            if a > b:
                a, b = b, a
            m &= ~((x >= a) & (x <= b))
    return m


def _steepest_x(x: np.ndarray, y: np.ndarray, smooth_window: int = 5, smooth_method: str = "median") -> float:
    """
    Return the x-position of the steepest slope (max |dy/dx|), after light smoothing.
    Used as an initial guess for the edge center.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return float(np.nanmedian(x) if x.size else 0.0)

    # light, robust smoothing to avoid chasing noise
    y_s = _rolling_smooth(y, window=smooth_window, method=smooth_method)
    # gradient w.r.t. x to handle nonuniform spacing
    dy_dx = np.gradient(y_s, x)
    idx = int(np.nanargmax(np.abs(dy_dx)))
    return float(x[idx])


def _sigma_from_10_90_width(width):
    # width_10_90 = 2*sqrt(2)*sigma*erfinv(0.8)
    return float(width) / (2.0 * np.sqrt(2.0) * float(erfinv(0.8)))


def fit_gauss_cdf_lmfit(
    x, y, *,
    robust=True,
    min_width=None,              # None -> auto = 2 * median Δx
    use_auto_fit_band=True,      # fit only around the edge (like the logistic helper)
    band_expand_points=6,        # ~ a dozen points total
    x_range: Optional[Tuple[float, float]] = None,           # <-- added
    exclude: Optional[List[Tuple[float, float]]] = None,     # <-- added
):
    """
    Fit the Gaussian-CDF edge with lmfit, with two practical guards:
      - sigma has a floor set by a minimum 10–90% width (from sampling)
      - the fit is restricted to a band around the steepest slope
    You can also constrain the domain via x_range and/or exclude sub-intervals.
    """
    if Model is None:
        raise RuntimeError("lmfit is not available to perform Gaussian-CDF fitting.")

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Clip/exclude x before anything else
    keep = _x_filter_mask(x, x_range=x_range, exclude=exclude)
    x, y = x[keep], y[keep]
    if x.size < 5:
        raise RuntimeError("Not enough points remain after x filtering for a stable fit.")

    # Initial guesses and sigma floor from sampling
    p0 = _initial_guess_gauss_cdf(x, y)
    x0_guess = p0["x0"]

    dx_med = float(np.median(np.diff(np.sort(x)))) if x.size > 1 else 1.0
    if min_width is None:
        min_width = max(2.0 * dx_med, 1e-6)   # ≈ two samples
    sigma_min = max(1e-6, _sigma_from_10_90_width(min_width))

    # Optional: restrict fit to a band around x0
    if use_auto_fit_band:
        order = np.argsort(x)
        xs = x[order]; ys = y[order]
        idx0 = int(np.searchsorted(xs, x0_guess))
        i0 = max(0, idx0 - band_expand_points)
        i1 = min(len(xs) - 1, idx0 + band_expand_points)
        mask_sorted = np.zeros_like(xs, dtype=bool); mask_sorted[i0:i1+1] = True
        mask = np.zeros_like(x, dtype=bool); mask[order] = mask_sorted
        x_fit, y_fit = x[mask], y[mask]
    else:
        x_fit, y_fit = x, y

    # Build and fit model
    mod = Model(gauss_cdf)
    params = mod.make_params(**p0)
    params['sigma'].min = sigma_min
    params['x0'].min = float(np.min(x)) - 0.5 * abs(np.ptp(x))
    params['x0'].max = float(np.max(x)) + 0.5 * abs(np.ptp(x))

    fit_method = 'least_squares' if robust else 'leastsq'
    fit_kws = dict(loss='soft_l1', f_scale=1.0) if robust else None

    result = mod.fit(
        np.asarray(y_fit, float),
        params,
        x=np.asarray(x_fit, float),
        method=fit_method,
        fit_kws=fit_kws,
        nan_policy='omit',
    )

    xv = float(result.params['x0'].value)
    sv = float(abs(result.params['sigma'].value))
    ylo = float(result.params['y_lo'].value)
    yhi = float(result.params['y_hi'].value)

    a = np.sqrt(2.0) * sv * float(erfinv(0.8))
    x10 = xv - a
    x90 = xv + a
    width = x90 - x10
    slope = (yhi - ylo) / (np.sqrt(2.0 * np.pi) * sv)

    metrics = dict(x0=xv, sigma=sv, y_lo=ylo, y_hi=yhi,
                   x10=x10, x90=x90, width_10_90=width, slope_at_x0=slope,
                   sigma_min=sigma_min, dx_median=dx_med, used_band=bool(use_auto_fit_band),
                   band_expand_points=int(band_expand_points),
                   x_filter=dict(x_range=x_range, exclude=exclude))
    return result, metrics


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

if __name__ == "__main__":
    # for Windows
    #DEFAULT_INPUT = r"C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/shared/Koerner, Lucas J.'s files - lloyd_gavin/data/experiment_20250814_004115/yaw_step_20250814_004115__wide.csv"
    #DEFAULT_INPUT = r"C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_summer/shared/Koerner, Lucas J.'s files - lloyd_gavin/data/experiment_20250821_220820/yaw_step_20250821_220820__wide.csv"
    #DEFAULT_INPUT = r"C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_summer/shared/Koerner, Lucas J.'s files - lloyd_gavin/data/experiment_20250901_214559/roll_step_20250901_214559__wide.csv"
    #DEFAULT_INPUT = r"C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_summer/shared/Koerner, Lucas J.'s files - lloyd_gavin/data/experiment_20251009_234705/roll_step_20251009_234705__wide.csv"
    DEFAULT_INPUT = r"C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_summer/shared/Koerner, Lucas J.'s files - lloyd_gavin/data/experiment_20251013_232418/roll_step_20251013_232418__wide.csv"


    # for Mac
    #DEFAULT_INPUT = Path("/Users/gavinlloyd/Library/CloudStorage/OneDrive-UniversityofSt.Thomas/2025_Summer/shared/Koerner, Lucas J.'s files - lloyd_gavin/data/experiment_20250814_004115/yaw_step_20250814_004115__wide.csv")

    try:
        an = load_analysis(DEFAULT_INPUT)
        
        # --- ROLL HEATMAPS EVERY 45° ---

        angles_45deg = list(range(-180, 181, 45))  # [-180, -135, -90, ..., 180]

        # Signal-strength heatmaps at 45° increments
        export_roll_signal_heatmaps(an, expected_positions=angles_45deg, region="all", zones=None)

        # --- CNH histogram for zone 28 at 0° roll ---
        cnh_histograms_for_location(an, movement_value=0, zones=[28], normalize=False, save=an.input_csv.parent / "analysis/cnh_zone28_at_0deg.pdf")
        
        # CNH histograms every 45° for zones 24–31 (PDFs)
        angles_45 = list(range(-135, 181, 45))
        for i in range(24, 32):  # 24..31 inclusive
            cnh_histograms_for_zone(an, zone=i, locations=angles_45, normalize=False, save=an.input_csv.parent / f"analysis/cnh_zone{i}_every_45deg.pdf")

        '''
        # Basic plots (optional)
        heatmap(an, movement_value=0, region="all", save=an.input_csv.parent / "analysis")
        heatmap_signal_strength(an, movement_value=0, region="all", save=an.input_csv.parent / "analysis")

        total_cnh_bin_sum_per_location(an, region="all", save=an.input_csv.parent / "analysis")
        total_cnh_bin_sum_per_location(an, region="inner36", save=an.input_csv.parent / "analysis")
        total_cnh_bin_sum_per_location(an, region="inner16", save=an.input_csv.parent / "analysis")
        total_cnh_bin_sum_per_location(an, region="inner4", save=an.input_csv.parent / "analysis")
        total_cnh_bin_sum_per_location(an, zones=[27], save=an.input_csv.parent / "analysis")
        compare_region_bin_sums_per_location(an, save=an.input_csv.parent / "analysis")

        total_signal_strength_per_location(an, region="all", save=an.input_csv.parent / "analysis")
        total_signal_strength_per_location(an, region="inner36", save=an.input_csv.parent / "analysis")
        total_signal_strength_per_location(an, region="inner16", save=an.input_csv.parent / "analysis")
        total_signal_strength_per_location(an, region="inner4", save=an.input_csv.parent / "analysis")
        total_signal_strength_per_location(an, zones=[27], save=an.input_csv.parent / "analysis")
        compare_region_signal_strength_per_location(an, save=an.input_csv.parent / "analysis")

        cnh_histograms_for_location(an, movement_value=0, zones=[24, 25, 26, 27, 28, 29, 30, 31], save=an.input_csv.parent / "analysis")
        cnh_histograms_for_zone(an, zone=27, locations=[-15, -7, 0, 7, 15], normalize=False, save=an.input_csv.parent / "analysis")

        print("[analysis] Finished example run. Outputs under:", an.input_csv.parent / "analysis")

        # ---- Gaussian CDF (lmfit): estimate 10–90% width on a single-zone sum ----
        sig_loc = total_signal_strength_per_location(an, zones=[27], save=None, show=False)
        x = sig_loc["movement_value"].values
        y = sig_loc["signal"].values

        try:
            # Guardrails are inside fit_gauss_cdf_lmfit (sigma floor + edge band)
            result, metrics = fit_gauss_cdf_lmfit(
                x, y,
                robust=True,
                x_range=(-10, 15),          # <— keep only this span
                # exclude=[(2, 4)]          # <— optional holes, if you ever want them
            )

            print("10–90 edges (CDF):", (metrics['x10'], metrics['x90']), "  width:", metrics['width_10_90'])

            # Plot the fit and mark 10%/90% edges
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 400)
            yhat = gauss_cdf(xs, metrics['x0'], metrics['sigma'], metrics['y_lo'], metrics['y_hi'])

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, 'o-', label="Raw")
            ax.plot(xs, yhat, '-', label="Gaussian CDF fit")
            ax.axvline(metrics['x10'], ls='--', color='tab:blue', alpha=0.7, label="10% edge")
            ax.axvline(metrics['x90'], ls='--', color='tab:blue', alpha=0.7, label="90% edge")
            midx = 0.5 * (metrics['x10'] + metrics['x90'])
            midy = gauss_cdf(midx, metrics['x0'], metrics['sigma'], metrics['y_lo'], metrics['y_hi'])
            ax.annotate(f"width ≈ {metrics['width_10_90']:.2f}",
                        xy=(midx, midy),
                        xytext=(0, -25), textcoords="offset points", ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.7))
            ax.set_xlabel("Location (movement_value)")
            ax.set_ylabel("Total average signal strength")
            ax.set_title("Gaussian CDF fit with 10–90% width")
            ax.legend(loc="best")
            fig.tight_layout()

            outpath = _resolve_save_path(an.input_csv.parent / "analysis", "signal_cdf_fit.png", an.input_csv.parent / "analysis")
            if outpath:
                fig.savefig(outpath, dpi=160)
            plt.show()
            plt.close(fig)
        except Exception as e:
            print("Gaussian-CDF fit failed:", e)
        '''

    except Exception as e:
        print('[analysis] Example run failed:', e)
        print("[analysis] Skipped example run due to:", e)
        print("Edit DEFAULT_INPUT in __main__ or import and call functions directly.")
