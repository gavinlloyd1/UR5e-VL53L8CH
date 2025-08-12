"""
vl53l8ch_data.py
----------------
Helper functions for managing and recording VL53L8CH experiment data logs.

Functions:
    get_new_log_folder(base_path, before_set):
        Detects newly created "log__" folders in the given base path that were not present
        before a logging cycle began. Returns the newest matching folder path.

    find_data_csv(folder_path):
        Searches the specified folder for a `data_*.csv` file and returns its full path
        if found, otherwise returns None.

    log_pose_to_csv(csv_path, pose_index, movement_label, movement_value, pose_vector, csv_file_path):
        Appends experiment metadata (pose index, timestamp, movement parameter, pose vector)
        and the associated CSV filename to a master log CSV file. Automatically writes a
        header if the file is new or empty.

    ingest_run_to_pandas(csv_path, experiment_id, pose_vector, movement_label, movement_value, preset=None,
                         parquet_dir=None, manifest_path=None, tz="America/Chicago"):
        Load the EVK CSV into a wide pandas DataFrame (no reshaping), attach experiment metadata,
        optionally save a Parquet copy, and update a simple manifest. Returns a dict with
        the run_id, DataFrame, and any written paths.

Key details:
    • Designed for post-processing and organization of sensor data.
    • Tracks association between robot poses and sensor data files.
    • Ensures consistent CSV logging format across multiple experiments.
    • Optional pandas ingestion keeps a 1:1 wide copy of the original spreadsheet and
      adds metadata columns for easy filtering/concatenation later.
"""


import os
import csv
import hashlib
from datetime import datetime


# -------------------------------------------------------------------
# DATA INDEXING HELPERS (existing)
# -------------------------------------------------------------------

def get_new_log_folder(base_path, before_set):
    """Return the path to the newest 'log__' folder that wasn't in before_set."""
    after_set = set(os.listdir(base_path))
    new_folders = after_set - before_set
    new_folders = [f for f in new_folders if f.startswith("log__")]
    if not new_folders:
        return None
    latest_folder = sorted(new_folders)[-1]
    return os.path.join(base_path, latest_folder)


def find_data_csv(folder_path):
    """Return the path to the data_*.csv file inside folder_path."""
    files = os.listdir(folder_path)
    for file in files:
        if file.startswith("data_") and file.endswith(".csv"):
            return os.path.join(folder_path, file)
    return None


def log_pose_to_csv(csv_path, pose_index, movement_label, movement_value, pose_vector, csv_file_path):
    """Append pose + movement + pose vector + filename to master CSV."""
    timestamp = datetime.now().isoformat(timespec='seconds')

    # Only write header if file doesn't exist or is empty
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    header = ["pose_index", "timestamp", movement_label, "x", "y", "z", "rx", "ry", "rz", "data_file"]

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([pose_index, timestamp, movement_value, *pose_vector, os.path.basename(csv_file_path)])
    print(f"Logged pose {pose_index} to {os.path.basename(csv_path)}")



# -------------------------------------------------------------------
# PANDAS INGEST HELPERS (new, additive)
# -------------------------------------------------------------------

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
    try:
        import pandas as pd  # imported here so module still works if pandas isn't installed
    except ImportError as e:
        raise RuntimeError("pandas is required to use ingest_run_to_pandas().") from e

    abs_csv = os.path.abspath(csv_path)
    if not os.path.exists(abs_csv):
        raise FileNotFoundError(f"CSV not found: {abs_csv}")

    run_id = derive_run_id(abs_csv)

    # Read the CSV "as-is" (wide) with a forgiving engine
    df = pd.read_csv(abs_csv, engine="python")

    # Best-effort parsing of time columns (keeps originals)
    df = parse_time_columns(df, tz=tz)

    # Attach metadata columns (front of the DataFrame)
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
    }
    if isinstance(preset, dict):
        for k, v in preset.items():
            if isinstance(v, (str, int, float, bool)) and k not in meta:
                # namespace preset fields to avoid collisions
                meta[f"preset_{k}"] = v

    df = attach_metadata(df, meta, front=True)

    # Optionally persist to Parquet
    parquet_path = None
    if parquet_dir:
        parquet_path = save_parquet(df, parquet_dir, run_id)

    # Optionally update a manifest
    manifest_row = None
    if manifest_path:
        st = os.stat(abs_csv)
        manifest_row = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "csv_path": abs_csv,
            "parquet_path": parquet_path,
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "file_size": int(st.st_size),
            "mtime_local": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "movement_label": movement_label,
            "movement_value": movement_value,
            "x_m": meta["x_m"],
            "y_m": meta["y_m"],
            "z_m": meta["z_m"],
            "rx_rad": meta["rx_rad"],
            "ry_rad": meta["ry_rad"],
            "rz_rad": meta["rz_rad"],
            "schema_hash": hash_column_names(df.columns),
        }
        update_manifest(manifest_path, manifest_row)

    return {
        "run_id": run_id,
        "df": df,
        "parquet_path": parquet_path,
        "manifest_row": manifest_row,
    }


def derive_run_id(csv_path):
    """
    Create a deterministic run_id from absolute path, file size, and mtime.

    This keeps the ID stable for retries while still changing if the file changes.
    """
    st = os.stat(csv_path)
    h = hashlib.md5()
    h.update(os.path.abspath(csv_path).encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    # use integer seconds to be stable across filesystems with coarse mtime resolution
    h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def parse_time_columns(df, tz="America/Chicago"):
    """
    Best-effort parse of common EVK time columns.

    - If a human-readable host datetime column exists (e.g., ".HostDateTime"), produce
      `host_datetime_local` (tz-aware). The original column is preserved.
    - If a numeric host timestamp exists (e.g., ".HostTimestamp" or "time_stamp"),
      produce `host_timestamp_utc` as UTC.

    This function is lenient: if a column isn't present or can't be parsed, it skips it.
    """
    import pandas as pd
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
    import pandas as pd  # local import for consistency with optional dependency pattern
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
    import pandas as pd
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
