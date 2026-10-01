"""
ur5e_motion_demo.py
-------------------
Simple demonstration of UR5e control over Ethernet.

This script demonstrates several movement functions from ur5e_control.py
without using the VL53L8CH ToF sensor or any sensor-related functions.

10/1/2026
"""

import time

from ur5e_control import UR5eController


# -------------------------------------------------------------------
# UR5e CONFIGURATION
# -------------------------------------------------------------------

IP = "10.219.1.138"                     # UR5e IP address
TCP_M = (0, -0.025, 0.150, 0, 0, 0)   # Tool center point [m]
PAYLOAD_KG = 0.1
MAX_STARTUP_ATTEMPTS = 5


# -------------------------------------------------------------------
# DEMO SETTINGS
# -------------------------------------------------------------------

MOVE_DISTANCE_M = 0.03   # 3 cm translation
YAW_ANGLE_DEG = 10       # 10 degree rotation
MOVE_DELAY_S = 3         # Allow each movement to finish


# -------------------------------------------------------------------
# MOVEMENT DEMO
# -------------------------------------------------------------------

def run_motion_demo(robot: UR5eController):
    """Demonstrate basic UR5e movement functions."""

    print("\nStarting UR5e movement demonstration.")
    print("Initial pose:", robot.get_pose_vector())

    # Move to the known safe lowered position
    print("\nMoving to safe lowered position...")
    robot.move_down_safe()

    # Move in the X direction and return
    print("\nMoving +X...")
    robot.move_x_m(MOVE_DISTANCE_M)
    time.sleep(MOVE_DELAY_S)

    print("Returning -X...")
    robot.move_x_m(-MOVE_DISTANCE_M)
    time.sleep(MOVE_DELAY_S)

    # Move in the Y direction and return
    print("\nMoving +Y...")
    robot.move_y_m(MOVE_DISTANCE_M)
    time.sleep(MOVE_DELAY_S)

    print("Returning -Y...")
    robot.move_y_m(-MOVE_DISTANCE_M)
    time.sleep(MOVE_DELAY_S)

    # Move in the Z direction and return
    print("\nMoving +Z...")
    robot.move_z_m(MOVE_DISTANCE_M)
    time.sleep(MOVE_DELAY_S)

    print("Returning -Z...")
    robot.move_z_m(-MOVE_DISTANCE_M)
    time.sleep(MOVE_DELAY_S)

    # Rotate the tool around its Y-axis and return
    print("\nRotating yaw +10 degrees...")
    robot.rotate_yaw_deg(YAW_ANGLE_DEG)
    time.sleep(MOVE_DELAY_S)

    print("Returning yaw to starting orientation...")
    robot.rotate_yaw_deg(-YAW_ANGLE_DEG)
    time.sleep(MOVE_DELAY_S)

    print("\nFinal pose:", robot.get_pose_vector())
    print("\nMovement demonstration complete.")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():    
    print("Entering main()...")

    robot = UR5eController(
        IP,
        TCP_M,
        PAYLOAD_KG,
        MAX_STARTUP_ATTEMPTS
    )

    try:
        run_motion_demo(robot)

        print("\nReturning robot to home position...")
        robot.move_home_safe()

    finally:
        robot.close()


if __name__ == "__main__":
    main()