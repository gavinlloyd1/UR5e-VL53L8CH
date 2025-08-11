import time
import re
from urx.urrobot import URRobot

# CONFIGURATION
IP = "192.168.0.1"              # UR5e IP address (computer's IP address needs to be 192.168.0.2)
TCP_M = (0, 0, 0.150, 0, 0, 0)  # Tool center point [m]
PAYLOAD_KG = 0.1

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

def rotate_x(robot, angle_rad, acc=0.2, vel=0.1):
    move_relative(robot, rx=angle_rad, acc=acc, vel=vel)

def rotate_y(robot, angle_rad, acc=0.2, vel=0.1):
    move_relative(robot, ry=angle_rad, acc=acc, vel=vel)

def rotate_z(robot, angle_rad, acc=0.2, vel=0.1):
    move_relative(robot, rz=angle_rad, acc=acc, vel=vel)

def go_home(robot, acc=0.5, vel=0.3):
    """Move robot to a known safe joint position."""
    home_joint_angles = (0, -1.57, 0, -1.57, 0, 0)
    robot.movej(home_joint_angles, acc=acc, vel=vel, wait=False)
    time.sleep(10)
    print("Moved to home position")

# -------------------------------------------------------------------
# POSE HELPER
# -------------------------------------------------------------------

def get_pose_vector(robot):
    """Return current pose as a 6-element float list."""
    return robot.getl()

# -------------------------------------------------------------------
# STARTUP HELPER
# -------------------------------------------------------------------

def startup(ip=IP, tcp_m=TCP_M, payload_kg=PAYLOAD_KG):
    try:
        robot = URRobot(ip)
        time.sleep(1)
        print("Connected to UR5e")

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
        print("Exception during startup:", e)
        return None

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    robot = startup()

    if robot:
        try:
            pose_list = get_pose_vector(robot)
            print("Initial pose:", pose_list)

            move_z(robot, -0.4)  # Move down 5 cm
            #time.sleep(1)
            #move_y(robot, 0.10)   # Move forward 10 cm
            #time.sleep(4)
            #rotate_x(robot, 0.3)  # Rotate 0.3 rad

        finally:
            robot.close()
            print("Connection closed.")
