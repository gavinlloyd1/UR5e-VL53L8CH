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

Key details:
    • Designed for post-processing and organization of sensor data.
    • Tracks association between robot poses and sensor data files.
    • Ensures consistent CSV logging format across multiple experiments.
"""


import os
import csv
from datetime import datetime


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