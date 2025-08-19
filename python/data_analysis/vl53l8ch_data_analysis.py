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
) -> pd.DataFrame:
    """
    Total **average signal strength** vs. location (sums per-zone averages across selected zones).
    Returns DataFrame ['movement_value','signal'].
    """
    df = an.signal_loc_zone
    allowed = set(_apply_region_and_zones(an.zones, region=region, zones=zones))
    df = df[df["zone"].isin(allowed)]
    sig_loc = df.groupby("movement_value", as_index=False)["signal"].sum()

    if save is not None or show:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sig_loc["movement_value"], sig_loc["signal"], marker="o")
        ax.set_xlabel("Location (movement_value)")
        ax.set_ylabel("Total average signal strength")
        ax.set_title(f"Total Signal Strength per Location [{'zones=' + ','.join(map(str, zones)) if zones is not None else region}]")
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


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Set your default input path, then run:
    #   python vl53l8ch_data_analysis.py
    DEFAULT_INPUT = r"C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/shared/Koerner, Lucas J.'s files - lloyd_gavin/data/experiment_20250814_004115/yaw_step_20250814_004115__wide.csv"

    try:
        an = load_analysis(DEFAULT_INPUT)

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

    except Exception as e:
        print("[analysis] Skipped example run due to:", e)
        print("Edit DEFAULT_INPUT in __main__ or import and call functions directly.")
