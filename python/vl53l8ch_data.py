"""
vl53l8ch_data.py
----------------
Helper functions for managing, logging, and ingesting VL53L8CH experiment data.

Functions:
    get_new_log_folder(base_path, before_set):
        Detects newly created "log__" folders in the given base path that were not
        present before a logging cycle began. Returns the most recent matching folder path.

    find_data_csv(folder_path):
        Returns the path to the newest `data_*.csv` file in the given folder, or None.

    find_info_csv(folder_path):
        Returns the path to the newest `info_*.csv` file in the given folder, or None.

    find_data_and_info_csv(folder_path, timeout_s=10, poll_s=0.25):
        Waits (polling) until both a data CSV and an info CSV are present in the folder,
        or until timeout. Returns a tuple (data_csv_path, info_csv_path).

    write_pose_log_header(csv_path, movement_label):
        Creates or overwrites `pose_log.csv` with the correct header row for a run.

    log_pose_to_csv(csv_path, pose_index, movement_label, movement_value,
                    pose_vector, csv_file_path, info_csv_path=None, log_folder=None):
        Appends a row to `pose_log.csv` containing pose/movement metadata, filenames,
        and log folder path. Assumes the header is already written.

    _read_info_metadata(info_csv_path):
        Reads GUI-generated `info_*.csv` and returns a dict of extracted metadata.
        Supports key/value pairs, single-row tables, and multi-row JSON serialization.

    _append_manifest_row(manifest_path, row_dict):
        Appends or updates a single manifest entry (CSV or Parquet) keyed by run_id.

    ingest_run_to_pandas(csv_path, experiment_id, pose_vector, movement_label,
                         movement_value, preset=None, parquet_dir=None, manifest_path=None,
                         tz="America/Chicago", info_csv_path=None):
        Loads an EVK-generated data CSV into a wide pandas DataFrame, attaches run
        metadata (including parsed info CSV fields if available), optionally writes
        a Parquet copy (with Snappy compression when available), and updates a manifest.
        Adds a quick schema hash for change tracking.

    derive_run_id(abs_csv_path):
        Generates a stable run_id from file path, size, and mtime.

    parse_time_columns(df, tz="America/Chicago"):
        Best-effort parsing and timezone localization of common EVK time columns.

    attach_metadata(df, meta, front=True):
        Inserts metadata columns into a DataFrame, optionally moving them to the front.

    save_parquet(df, parquet_dir, run_id):
        Saves a DataFrame to Parquet (Snappy if available) and returns the output path.

    update_manifest(manifest_path, row):
        Creates or updates a manifest file, supporting CSV and Parquet formats.

    hash_column_names(cols):
        Returns a short hash of column names for quick schema comparison.

Key details:
    • Designed for post-processing and organization of VL53L8CH sensor data.
    • Tracks association between robot poses and both sensor data files and GUI metadata files.
    • Allows resetting `pose_log.csv` headers between runs for consistent schema.
    • Ingestion functions preserve the raw EVK spreadsheet layout (wide format) while
      prepending metadata columns for filtering/merging.
    • Parquet writing includes compression fallback for maximum compatibility.
    • Manifest handling supports schema change detection via column name hashing.
"""


import time
import os
import csv
import hashlib
import pandas as pd
from datetime import datetime


# -------------------------------------------------------------------
# DATA INDEXING HELPERS
# -------------------------------------------------------------------

def get_new_log_folder(base_path, before_set):
    """Return the path to the newest 'log__' folder that wasn't in before_set."""
    after_set = set(os.listdir(base_path))
    new_items = sorted(after_set - before_set)
    candidates = []
    for name in new_items:
        if name.startswith("log__"):
            full = os.path.join(base_path, name)
            if os.path.isdir(full):
                candidates.append(full)
    if not candidates:
        return None
    # pick most recent by mtime (safer than name sort if clock or naming varies)
    return max(candidates, key=lambda p: os.path.getmtime(p))


def find_data_csv(folder_path):
    """Return the path to the newest data_*.csv file inside folder_path."""
    try:
        entries = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    except FileNotFoundError:
        return None
    matches = [p for p in entries
               if os.path.isfile(p)
               and os.path.basename(p).startswith("data_")
               and p.endswith(".csv")
               and os.path.getsize(p) > 0]  # skip zero-byte partials
    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(p))


def find_info_csv(folder_path):
    """Return the path to the newest info_*.csv file inside folder_path."""
    try:
        entries = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    except FileNotFoundError:
        return None
    matches = [p for p in entries
               if os.path.isfile(p)
               and os.path.basename(p).startswith("info_")
               and p.endswith(".csv")
               and os.path.getsize(p) > 0]  # skip zero-byte partials
    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(p))


def find_data_and_info_csv(folder_path, timeout_s=10, poll_s=0.25):
    """Wait up to timeout_s for BOTH data_*.csv and info_*.csv; return (data, info) or (None, None)."""
    def _is_stable_file(path, prev_size=None):
        try:
            size = os.path.getsize(path)
            return (size > 0 and prev_size is not None and size == prev_size), size
        except OSError:
            return False, None

    end = time.time() + timeout_s
    data_path = info_path = None
    prev_sizes = {}
    while time.time() < end:
        data_path = data_path or find_data_csv(folder_path)
        info_path = info_path or find_info_csv(folder_path)
        if data_path and info_path:
            stable_d, sz_d = _is_stable_file(data_path, prev_sizes.get(data_path))
            stable_i, sz_i = _is_stable_file(info_path, prev_sizes.get(info_path))
            prev_sizes[data_path] = sz_d
            prev_sizes[info_path] = sz_i
            if stable_d and stable_i:
                return data_path, info_path
        time.sleep(poll_s)
    return data_path, info_path  # may be None(s)


# Centralized header helpers (avoid duplication and drift)
POSE_LOG_FIXED_PREFIX = ["pose_index", "timestamp"]
POSE_LOG_SUFFIX = ["x", "y", "z", "rx", "ry", "rz", "data_file", "info_file", "log_folder"]

def pose_log_header(movement_label):
    return POSE_LOG_FIXED_PREFIX + [movement_label] + POSE_LOG_SUFFIX


def write_pose_log_header(csv_path, movement_label):
    """Write (or overwrite) the header for pose_log.csv."""
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(pose_log_header(movement_label))


def _first_row(csv_path):
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            return f.readline().strip()
    except FileNotFoundError:
        return ""


def log_pose_to_csv(csv_path, pose_index, movement_label, movement_value, pose_vector,
                    csv_file_path, info_csv_path=None, log_folder=None):
    """Append pose + movement + pose vector + filenames/paths to master CSV."""
    # sanity: file must already have a header (we no longer write it here)
    header_line = _first_row(csv_path)
    if not header_line:
        raise RuntimeError("pose_log.csv has no header. Call write_pose_log_header(...) before logging.")
    if movement_label not in header_line.split(","):
        raise RuntimeError(f"pose_log.csv header mismatch (missing '{movement_label}'). Did you rewrite the header?")

    timestamp = datetime.now().isoformat(timespec='seconds')

    data_file_base = os.path.basename(csv_file_path) if csv_file_path else ""
    info_file_base = os.path.basename(info_csv_path) if info_csv_path else ""
    log_folder_abs = os.path.abspath(log_folder) if log_folder else ""

    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            pose_index, timestamp, movement_value,
            *pose_vector,
            data_file_base,
            info_file_base,
            log_folder_abs
        ])

    print(f"Logged pose {pose_index} to {os.path.basename(csv_path)}")



# -------------------------------------------------------------------
# PANDAS INGEST HELPERS
# -------------------------------------------------------------------

def _read_info_metadata(info_csv_path):
    """
    Best-effort parser for the GUI 'info_*.csv'.
    - If it looks like key/value (2 columns), flatten to a dict.
    - If it's a single-row table, flatten columns with 'info_' prefix.
    - If multi-row, store JSON into 'info_table_json'.
    """

    if not info_csv_path or not os.path.exists(info_csv_path):
        return {}

    info = pd.read_csv(info_csv_path, engine="python", encoding="utf-8", on_bad_lines="skip")
    md = {}

    if info.shape[1] == 2:
        # key/value style
        kcol, vcol = info.columns[:2]
        try:
            vals = info[vcol]
            md = {str(k): (None if pd.isna(v) else v) for k, v in zip(info[kcol].astype(str), vals)}
        except Exception:
            md = {}
    else:
        if len(info) == 1:
            md = {f"info_{c}": info.iloc[0][c] for c in info.columns}
        else:
            md = {"info_table_json": info.to_json(orient="records")}

    # Normalize a few common fields if they happen to exist
    ren = {
        "GUI Version": "gui_version",
        "Preset": "preset_name",
        "CNH Start Bin": "cnh_start_bin",
        "CNH Num Bins": "cnh_num_bins",
        "CNH Sub Sample": "cnh_sub_sample",
        "Capture Start": "capture_started_at_local",
    }
    for old, new in ren.items():
        if old in md:
            md[new] = md.pop(old)

    return md


def _append_manifest_row(manifest_path, row_dict):
    
    ext = os.path.splitext(manifest_path)[1].lower()
    exists = os.path.exists(manifest_path)

    if exists:
        try:
            if ext == ".parquet":
                mdf = pd.read_parquet(manifest_path)
            else:
                mdf = pd.read_csv(manifest_path)
        except Exception:
            mdf = pd.DataFrame()  # recover from corrupted manifest
        # update or append (only filter if 'run_id' exists)
        if "run_id" in mdf.columns:
            mdf = mdf[mdf["run_id"] != row_dict["run_id"]]
        mdf = pd.concat([mdf, pd.DataFrame([row_dict])], ignore_index=True)
    else:
        mdf = pd.DataFrame([row_dict])

    if ext == ".parquet":
        mdf.to_parquet(manifest_path, index=False)
    else:
        mdf.to_csv(manifest_path, index=False)


def ingest_run_to_pandas(
    csv_path,
    experiment_id,
    pose_vector,
    movement_label,
    movement_value,
    preset=None,
    parquet_dir=None,
    manifest_path=None,
    tz="America/Chicago",
    info_csv_path=None,
):
    """
    Load a single EVK-generated CSV into pandas (wide format), attach metadata columns,
    and optionally save a Parquet copy and update a manifest.

    Parameters
    ----------
    csv_path : str
        Path to the EVK-generated data_*.csv.
    experiment_id : str
        Identifier for the experiment/session (e.g., UUID or human-readable name).
    pose_vector : list[float]
        [x, y, z, rx, ry, rz] with units meters and radians.
    movement_label : str
        Name of the movement/sweep parameter (e.g., "yaw_deg").
    movement_value : Any
        Value of the movement parameter at capture (e.g., -20.0).
    preset : dict, optional
        CNH/logging settings applied (e.g., {"preset_name": "...", "cnh_start_bin": 4, ...}).
        Only simple scalar values are attached as columns (strings, ints, floats, bools).
    parquet_dir : str | None
        If provided, writes a Parquet copy to {parquet_dir}/{run_id}.parquet.
    manifest_path : str | None
        If provided, creates/updates a manifest (CSV or Parquet based on extension).
    tz : str
        Local timezone name for host datetime localization (default: "America/Chicago").
    info_csv_path : str | None
        Optional path to the GUI 'info_*.csv' to extract run-level metadata.

    Returns
    -------
    dict
        {
          "run_id": str,
          "df": pandas.DataFrame,       # the full wide DataFrame (metadata columns first)
          "parquet_path": Optional[str],
          "manifest_row": Optional[dict],
        }

    Notes
    -----
    - This function is idempotent with respect to a given CSV file path, size, and mtime.
      Calling it again for the same CSV yields the same run_id and updates (not duplicates)
      the manifest entry.
    - Measurement columns from the EVK are left exactly as-is. No reshaping is performed.
    """
    
    abs_csv = os.path.abspath(csv_path)
    if not os.path.exists(abs_csv):
        raise FileNotFoundError(f"CSV not found: {abs_csv}")

    run_id = derive_run_id(abs_csv)

    # Read the CSV "as-is" (wide) with a forgiving engine
    df = pd.read_csv(abs_csv, engine="python", encoding="utf-8", on_bad_lines="skip")

    # Best-effort parse a few time fields, honoring tz
    try:
        df = parse_time_columns(df, tz=tz)
    except Exception:
        # non-fatal
        pass

    # Provenance / metadata
    log_folder = os.path.abspath(os.path.dirname(abs_csv))
    info_abs = os.path.abspath(info_csv_path) if info_csv_path else ""

    meta = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "movement_label": movement_label,
        "movement_value": movement_value,
        "x_m": pose_vector[0] if len(pose_vector) > 0 else None,
        "y_m": pose_vector[1] if len(pose_vector) > 1 else None,
        "z_m": pose_vector[2] if len(pose_vector) > 2 else None,
        "rx_rad": pose_vector[3] if len(pose_vector) > 3 else None,
        "ry_rad": pose_vector[4] if len(pose_vector) > 4 else None,
        "rz_rad": pose_vector[5] if len(pose_vector) > 5 else None,
        "log_folder": log_folder,
        "data_csv": abs_csv,
        "info_csv": info_abs,
        "data_sha1": hashlib.sha1(open(abs_csv, "rb").read()).hexdigest()[:12],
    }

    # If we have an info CSV, add its checksum and parsed fields
    if info_abs and os.path.exists(info_abs):
        meta["info_sha1"] = hashlib.sha1(open(info_abs, "rb").read()).hexdigest()[:12]
        try:
            meta.update(_read_info_metadata(info_abs))
        except Exception as e:
            print(f"Warning: could not parse info CSV metadata: {e}")

    # If a preset dict was passed, copy only simple scalar values
    if isinstance(preset, dict):
        for k, v in preset.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                meta[k] = v

    # Prepend metadata columns (keep EVK data intact)
    for k, v in reversed(list(meta.items())):
        df.insert(0, k, v)

    # Optional Parquet
    parquet_path = None
    if parquet_dir:
        os.makedirs(parquet_dir, exist_ok=True)
        parquet_path = os.path.join(parquet_dir, f"{run_id}.parquet")
        try:
            df.to_parquet(parquet_path, compression="snappy", index=False)
        except Exception:
            df.to_parquet(parquet_path, index=False)

    # Optional manifest row (include a quick schema fingerprint)
    manifest_row = {
        **meta,
        "parquet_path": parquet_path or "",
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "schema_hash": hash_column_names(df.columns),  # NEW, you already defined it
    }
    if manifest_path:
        _append_manifest_row(manifest_path, manifest_row)

    return {"run_id": run_id, "df": df, "parquet_path": parquet_path, "manifest_row": manifest_row}


def derive_run_id(abs_csv_path):
    """Create a stable run_id from absolute path, file size, and mtime."""
    st = os.stat(abs_csv_path)
    mtime_part = getattr(st, "st_mtime_ns", int(st.st_mtime))
    key = f"{abs_csv_path}|{st.st_size}|{mtime_part}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def parse_time_columns(df, tz="America/Chicago"):
    """
    Best-effort parse of common EVK time columns.

    - If a human-readable host datetime column exists (e.g., ".HostDateTime"), produce
      `host_datetime_local` (tz-aware). The original column is preserved.
    - If a numeric host timestamp exists (e.g., ".HostTimestamp" or "time_stamp"),
      produce `host_timestamp_utc` as UTC.

    This function is lenient: if a column isn't present or can't be parsed, it skips it.
    """
    
    try:
        from zoneinfo import ZoneInfo  # Python 3.9+
        tzinfo = ZoneInfo(tz)
    except Exception:
        tzinfo = None

    # Human-readable local time columns
    for col in (".HostDateTime", "HostDateTime", "host_datetime", "host_date_time"):
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            # Localize if naive
            if tzinfo is not None:
                try:
                    if getattr(dt.dt, "tz", None) is None:
                        dt = dt.dt.tz_localize(tzinfo)
                except Exception:
                    # some pandas versions use dt.tz is None; keep best-effort
                    dt = dt.dt.tz_localize(tzinfo)
            df["host_datetime_local"] = dt
            break  # only take the first matching column

    # Numeric timestamp columns (epoch-like)
    for col in (".HostTimestamp", "HostTimestamp", "time_stamp", "timestamp"):
        if col in df.columns:
            ser = pd.to_numeric(df[col], errors="coerce")
            # Heuristic unit detection
            maxv = ser.max(skipna=True)
            if pd.isna(maxv):
                continue
            if maxv > 1e18:
                unit = "ns"
            elif maxv > 1e15:
                unit = "us"
            elif maxv > 1e11:
                unit = "ms"
            else:
                unit = "s"
            ts = pd.to_datetime(ser, unit=unit, errors="coerce", utc=True)
            df["host_timestamp_utc"] = ts
            break

    return df


def attach_metadata(df, meta, front=True):
    """
    Insert metadata columns (e.g., run_id, pose, movement) into the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
    meta : dict
        Column -> value mapping to insert.
    front : bool
        If True, move these columns to the front of the DataFrame.

    Returns
    -------
    pandas.DataFrame
    """

    for k, v in meta.items():
        df[k] = v
    if front:
        front_cols = list(meta.keys())
        rest = [c for c in df.columns if c not in front_cols]
        df = df.loc[:, front_cols + rest]
    return df


def save_parquet(df, parquet_dir, run_id):
    """
    Save a DataFrame to {parquet_dir}/{run_id}.parquet with Snappy compression when available.
    Returns the full path to the written file.
    """
    os.makedirs(parquet_dir, exist_ok=True)
    out_path = os.path.join(parquet_dir, f"{run_id}.parquet")
    try:
        df.to_parquet(out_path, compression="snappy", index=False)
    except Exception:
        # Fallback if snappy/pyarrow isn't available
        df.to_parquet(out_path, index=False)
    return out_path


def update_manifest(manifest_path, row):
    """
    Create or update a manifest of ingested runs.

    - If `manifest_path` ends with ".parquet", the manifest is stored as Parquet; otherwise CSV.
    - Upserts on `run_id`: replaces any existing row with the same run_id, then appends the new one.
    """

    ext = os.path.splitext(manifest_path)[1].lower()
    exists = os.path.exists(manifest_path)

    if exists:
        try:
            if ext == ".parquet":
                mdf = pd.read_parquet(manifest_path)
            else:
                mdf = pd.read_csv(manifest_path)
        except Exception:
            # Corrupt/unreadable: start fresh but don't crash the run
            mdf = pd.DataFrame()
    else:
        mdf = pd.DataFrame()

    # Drop any existing row with this run_id (upsert)
    if "run_id" in mdf.columns and not mdf.empty:
        mdf = mdf[mdf["run_id"] != row["run_id"]]

    mdf = pd.concat([mdf, pd.DataFrame([row])], ignore_index=True)

    try:
        if ext == ".parquet":
            mdf.to_parquet(manifest_path, compression="snappy", index=False)
        else:
            mdf.to_csv(manifest_path, index=False)
    except Exception:
        # Final fallback write without compression
        if ext == ".parquet":
            mdf.to_parquet(manifest_path, index=False)
        else:
            mdf.to_csv(manifest_path, index=False)


def hash_column_names(cols):
    """Compact hash of column names for quick schema comparisons in the manifest."""
    h = hashlib.md5()
    for c in cols:
        h.update(str(c).encode("utf-8"))
    return h.hexdigest()[:12]
