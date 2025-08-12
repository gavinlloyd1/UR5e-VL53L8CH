import time
import re
import math
import m3d
import os
from urx.urrobot import URRobot
from vl53l8ch_gui_auto import data_logging_cycle, vl53l8ch_gui_startup
from vl53l8ch_data import get_new_log_folder, find_data_csv, log_pose_to_csv


# -------------------------------------------------------------------
# UR5e CONFIGURATION
# -------------------------------------------------------------------

IP = "192.168.0.1"              # UR5e IP address (computer's IP address needs to be 192.168.0.2)
TCP_M = (0, 0, 0.150, 0, 0, 0)  # Tool center point [m]
PAYLOAD_KG = 0.1
MAX_STARTUP_ATTEMPS = 5


# -------------------------------------------------------------------
# VL53L8CH CONFIGURATION
# -------------------------------------------------------------------

IMAGE_TIMEOUT = 5
LOGGING_TIMEOUT = 300


# -------------------------------------------------------------------
# MOTION HELPERS
# -------------------------------------------------------------------

def move_relative(robot, dx=0, dy=0, dz=0, rx=0, ry=0, rz=0, acc=0.2, vel=0.1):
    """Move linearly in Cartesian space by given offsets."""
    pose = robot.getl()
    pose[0] += dx
    pose[1] += dy
    pose[2] += dz
    pose[3] += rx
    pose[4] += ry
    pose[5] += rz
    
    try:
        robot.movel(pose, acc=acc, vel=vel, wait=False)
    except Exception as e:
        print("Cartesian move failed:", e)

def move_x(robot, distance_m, acc=0.2, vel=0.1):
    move_relative(robot, dx=distance_m, acc=acc, vel=vel)

def move_y(robot, distance_m, acc=0.2, vel=0.1):
    move_relative(robot, dy=distance_m, acc=acc, vel=vel)

def move_z(robot, distance_m, acc=0.2, vel=0.1):
    move_relative(robot, dz=distance_m, acc=acc, vel=vel)

def move_rx_rad(robot, angle_rad, acc=0.2, vel=0.1):
    move_relative(robot, rx=angle_rad, acc=acc, vel=vel)

def move_ry_rad(robot, angle_rad, acc=0.2, vel=0.1):
    move_relative(robot, ry=angle_rad, acc=acc, vel=vel)

def move_rz_rad(robot, angle_rad, acc=0.2, vel=0.1):
    move_relative(robot, rz=angle_rad, acc=acc, vel=vel)

def move_rx_deg(robot, angle_deg, acc=0.2, vel=0.1):
    move_relative(robot, rx=math.radians(angle_deg), acc=acc, vel=vel)

def move_ry_deg(robot, angle_deg, acc=0.2, vel=0.1):
    move_relative(robot, ry=math.radians(angle_deg), acc=acc, vel=vel)

def move_rz_deg(robot, angle_deg, acc=0.2, vel=0.1):
    move_relative(robot, rz=math.radians(angle_deg), acc=acc, vel=vel)

def pitch_deg(robot, angle_deg, acc=0.2, vel=0.1):
    """Rotate around tool's X-axis (pitch)."""
    angle_rad = math.radians(angle_deg)
    pose = robot.getl()
    pos = m3d.Vector(pose[:3])
    orient = m3d.Orientation(pose[3:6])

    delta = m3d.Orientation()
    delta.rotate_xb(angle_rad)

    new_orient = orient * delta
    target = m3d.Transform(new_orient, pos)
    robot.movel([float(v) for v in target.pose_vector], acc=acc, vel=vel, wait=False)

def yaw_deg(robot, angle_deg, acc=0.2, vel=0.1):
    """Rotate around tool's Y-axis (yaw)."""
    angle_rad = math.radians(angle_deg)
    pose = robot.getl()
    pos = m3d.Vector(pose[:3])
    orient = m3d.Orientation(pose[3:6])

    delta = m3d.Orientation()
    delta.rotate_yb(angle_rad)

    new_orient = orient * delta
    target = m3d.Transform(new_orient, pos)
    robot.movel([float(v) for v in target.pose_vector], acc=acc, vel=vel, wait=False)

def roll_deg(robot, angle_deg, acc=0.2, vel=0.1):
    """Rotate around tool's Z-axis (roll)."""
    angle_rad = math.radians(angle_deg)
    pose = robot.getl()
    pos = m3d.Vector(pose[:3])
    orient = m3d.Orientation(pose[3:6])

    delta = m3d.Orientation()
    delta.rotate_zb(angle_rad)

    new_orient = orient * delta
    target = m3d.Transform(new_orient, pos)
    robot.movel([float(v) for v in target.pose_vector], acc=acc, vel=vel, wait=False)

def rotate_base_deg(robot, delta_deg, acc=0.2, vel=0.1):
    delta_rad = math.radians(delta_deg)
    joints = robot.getj()
    joints[0] += delta_rad  # rotate base joint
    robot.movej(joints, acc=acc, vel=vel, wait=False)

def go_home(robot, acc=0.5, vel=0.5):
    """Move robot to a known safe joint position."""
    home_joint_angles = (0, -1.57, 0, -1.57, 0, 0)
    robot.movej(home_joint_angles, acc=acc, vel=vel, wait=False)
    time.sleep(9)
    print("Moved to home position: ", get_pose_vector(robot))
    time.sleep(1)


# -------------------------------------------------------------------
# POSE HELPER
# -------------------------------------------------------------------

def get_pose_vector(robot):
    """Return current pose as a 6-element float list."""
    return robot.getl()


# -------------------------------------------------------------------
# EXPERIMENT HELPER
# -------------------------------------------------------------------

def yaw_stepper(robot, edge_deg, movement_label="yaw_deg", movement_value=1):
    num_locations = abs(edge_deg * 2) + 1
    num_frames = vl53l8ch_gui_startup(num_locations=num_locations)

    for i in range(num_locations):
        pose_vector = get_pose_vector(robot)

        print(f"\nPose {i+1}/{num_locations} ({movement_label} = {edge_deg + i} ): {pose_vector}")

        # Track folders before logging
        initial_folders = set(os.listdir(DATA_ROOT))

        # Trigger one logging run (blocking)
        print(f"\nStarting data logging at location {i+1}...")
        data_logging_cycle(num_frames, image_timeout=IMAGE_TIMEOUT, logging_timeout=LOGGING_TIMEOUT)
        print(f"Finished data logging at location {i+1}.")

        # Wait for folder to appear
        new_folder = get_new_log_folder(DATA_ROOT, initial_folders)
        if new_folder:
            data_csv = find_data_csv(new_folder)
            if data_csv:
                log_pose_to_csv(CSV_LOG_PATH, i+1, movement_label, movement_value, pose_vector, data_csv)
            else:
                print(f"No data_*.csv found in {new_folder}")
        else:
            print("No new log folder detected.")
        
        yaw_deg(robot, movement_value)

    print(f"\nData logging at all {num_locations} locations complete!")


# -------------------------------------------------------------------
# STARTUP HELPER
# -------------------------------------------------------------------

def startup(ip=IP, tcp_m=TCP_M, payload_kg=PAYLOAD_KG, max_startup_attempts=MAX_STARTUP_ATTEMPS, delay=2.0):
    for attempt in range(1, max_startup_attempts + 1):
        try:
            print(f"[UR5e] Attempt {attempt}/{max_startup_attempts} to connect...")
            robot = URRobot(ip)
            time.sleep(1)
            print("[UR5e] Successfully connected!")

            robot.set_tcp(tcp_m)
            robot.set_payload(payload_kg, (0, 0, 0.1))
            go_home(robot)

            # Monkey-patch broken getl()
            def safe_getl():
                # Get pose data from secondary monitor directly
                data = robot.secmon.get_cartesian_info()
                return [data["X"], data["Y"], data["Z"], data["Rx"], data["Ry"], data["Rz"]]
            robot.getl = safe_getl

            return robot
        except Exception as e:
                    print(f"[UR5e] Connection failed: {e}")
                    if attempt < max_startup_attempts:
                        print(f"[UR5e] Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        raise RuntimeError(f"[UR5e] Failed to connect after {max_startup_attempts} attempts.")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    robot = startup()

    if robot:
        try:
            move_z(robot, -0.5)
        finally:
            robot.close()
            print("Connection closed.")
