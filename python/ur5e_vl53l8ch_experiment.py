"""
ur5e_vl53l8ch_experiment.py
---------------------------
General experiment runner for coordinating a UR5e robotic arm with a
VL53L8CH time-of-flight (ToF) sensor for automated data collection.

Features:
    • Provides reusable experiment helper functions (e.g., yaw stepping).
    • Supports precise UR5e motion control via the UR5eController class.
    • Automates data capture through the VL53L8CH GUI.
    • Logs robot poses and associates them with collected sensor datasets.
    • Stores results in a master CSV log for later analysis.

Current Implementation:
    - yaw_stepper(): Sweeps robot yaw in fixed-degree increments and logs data.

Future Extensions:
    • Additional movement patterns (pitch sweeps, XYZ translation grids, etc.).
    • Multi-sensor experiment coordination.
    • Configurable motion and logging parameters from external files.

Key Components:
    UR5eController           - Motion and pose control for the UR5e robot.
    vl53l8ch_gui_automation  - Automates the VL53L8CH GUI for data logging.
    vl53l8ch_data            - Detects new log folders and writes master CSV logs.

Usage:
    Run this script directly to perform a configured experiment.
    Adjust constants (IP, TCP_M, PAYLOAD_KG, etc.) for your setup.
"""


import os
import time
from datetime import datetime


# -------------------------------------------------------------------
# LOCAL IMPORTS
# -------------------------------------------------------------------

from ur5e_control import UR5eController
from vl53l8ch_gui_automation import data_logging_cycle, vl53l8ch_gui_startup
from vl53l8ch_data import get_new_log_folder, find_data_csv, log_pose_to_csv, ingest_run_to_pandas


# -------------------------------------------------------------------
# UR5e CONFIGURATION
# -------------------------------------------------------------------

# for ur5e_control.py
IP = "10.219.1.138"             # UR5e IP address
TCP_M = (0, 0, 0.150, 0, 0, 0)  # Tool center point [m]
PAYLOAD_KG = 0.1
MAX_STARTUP_ATTEMPTS = 5


# -------------------------------------------------------------------
# VL53L8CH CONFIGURATION
# -------------------------------------------------------------------

# for vl53l8ch_gui_auto.py
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "gui_images") # Folder containing GUI button images
IMAGE_TIMEOUT = 5
LOGGING_TIMEOUT = 300

# for vl53l8ch_data.py
DATA_ROOT = r"C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/GUIs/MZAI_EVK_v1.0.1/data"
CSV_LOG_PATH = os.path.join(DATA_ROOT, "pose_log.csv")


# -------------------------------------------------------------------
# EXPERIMENT HELPERS
# -------------------------------------------------------------------

def yaw_stepper(robot: UR5eController, edge_deg: float, step_deg: float = 1.0, movement_label="yaw_deg"):
    """
    Rotates the UR5e robot's tool yaw in step_deg increments between -edge_deg and +edge_deg,
    triggering VL53L8CH ToF sensor data logging at each position.

    Steps:
      1. Moves the robot to the starting yaw (-edge_deg).
      2. Calculates the total number of positions (`num_locations`) based on `edge_deg`
         (works for positive or negative values).
      3. Initializes the VL53L8CH GUI for data collection (`num_frames` per position).
      4. For each yaw position:
         - Captures the current robot pose.
         - Logs a dataset via the GUI.
         - Waits for a new log folder to appear and records pose + data path in a master CSV.
         - Increments yaw by 1° (except after the last position).
      5. Prints progress and completion status.

    Args:
        robot: Connected UR5e robot control object.
        edge_deg (float): Max yaw angle (positive or negative) from the center position.
        movement_label (str): Label for the yaw movement column in the CSV log.
    """

    # Normalize input so negative values behave like positive magnitudes
    edge = abs(edge_deg)
    step_deg = abs(step_deg)
    if step_deg <= 0:
        raise ValueError("step_deg must be > 0")
    
    # Move to starting position
    start_angle = -edge
    robot.rotate_yaw_deg(start_angle)
    time.sleep(5)

    # Start the GUI (make sure it's open on the computer, do not drag it anywhere after opening)
    num_locations = int(round((edge * 2) / step_deg)) + 1
    num_frames = vl53l8ch_gui_startup(image_dir=IMAGE_DIR, num_locations=num_locations)

    current_yaw = start_angle
    for i in range(num_locations):
        pose_vector = robot.get_pose_vector()
        print(f"\nPose {i+1}/{num_locations} ({movement_label} = {current_yaw}° ): {pose_vector}")

        # Track folders before logging
        initial_folders = set(os.listdir(DATA_ROOT))

        # Trigger one logging run
        print(f"\nStarting data logging at location {i+1}...")
        success = data_logging_cycle(image_dir=IMAGE_DIR, num_frames=num_frames, image_timeout=IMAGE_TIMEOUT, logging_timeout=LOGGING_TIMEOUT)
        print(f"Finished data logging at location {i+1}. Success = {success}")

        # Wait for new folder to appear, then locate it
        time.sleep(2)
        new_folder = get_new_log_folder(DATA_ROOT, initial_folders)
        if new_folder:
            # Find the data .csv inside the new folder, then log to pose_log.csv
            data_csv = find_data_csv(new_folder)
            if data_csv:
                log_pose_to_csv(CSV_LOG_PATH, i+1, movement_label, current_yaw, pose_vector, data_csv)
            else:
                print(f"No data_*.csv found in {new_folder}")
        else:
            print("No new log folder detected.")
        
        # Step to next position (skip after last)
        if i < num_locations - 1:
            robot.rotate_yaw_deg(step_deg)
            current_yaw += step_deg
            print(f"Moved to position {i+2} ({current_yaw}°).")
            time.sleep(2)

    print(f"\nData logging at all {num_locations} locations complete!")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    robot = UR5eController(IP, TCP_M, PAYLOAD_KG, MAX_STARTUP_ATTEMPTS)

    if robot:
        try:
            robot.move_down_safe()
        finally:
            robot.close()