"""
ur5e_vl53l8ch_experiment.py
Experiment runner functions for UR5e + VL53L8CH ToF sensor.
Imports motion helpers from ur5e_control.
"""

import time
import os
from ur5e_control import UR5eController
from vl53l8ch_gui_auto import data_logging_cycle, vl53l8ch_gui_startup
from vl53l8ch_data import get_new_log_folder, find_data_csv, log_pose_to_csv


# -------------------------------------------------------------------
# UR5e CONFIGURATION
# -------------------------------------------------------------------

# for ur5e_control.py
IP = "192.168.0.1"              # UR5e IP address (computer's IP address needs to be 192.168.0.2)
TCP_M = (0, 0, 0.150, 0, 0, 0)  # Tool center point [m]
PAYLOAD_KG = 0.1
MAX_STARTUP_ATTEMPTS = 5


# -------------------------------------------------------------------
# VL53L8CH CONFIGURATION
# -------------------------------------------------------------------

# for vl53l8ch_gui_auto.py
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
    Rotates the UR5e robot's tool yaw in step_deg° increments between -edge_deg and +edge_deg,
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
    robot.yaw_deg(start_angle)
    time.sleep(5)

    # Start the GUI (make sure it's open on the computer, do not drag it anywhere after opening)
    num_locations = int(round((edge * 2) / step_deg)) + 1
    num_frames = vl53l8ch_gui_startup(num_locations=num_locations)

    current_yaw = start_angle
    for i in range(num_locations):
        pose_vector = robot.get_pose_vector()
        print(f"\nPose {i+1}/{num_locations} ({movement_label} = {current_yaw}° ): {pose_vector}")

        # Track folders before logging
        initial_folders = set(os.listdir(DATA_ROOT))

        # Trigger one logging run
        print(f"\nStarting data logging at location {i+1}...")
        success = data_logging_cycle(num_frames, image_timeout=IMAGE_TIMEOUT, logging_timeout=LOGGING_TIMEOUT)
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
            robot.yaw_deg(step_deg)
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
            robot.move_z_m(-0.5)
        finally:
            robot.close()