"""
ur5e_control.py
---------------
Encapsulates control of a Universal Robots UR5e robotic arm using the `urx` Python API.

This module defines the UR5eController class, which:
    • Handles TCP connection and retry logic to the UR5e.
    • Sets the tool center point (TCP) and payload parameters.
    • Provides high-level motion helpers for:
        - Cartesian translations (X/Y/Z)
        - Rotations in radians or degrees
        - Tool-axis roll, pitch, and yaw
        - Base joint rotation
        - Moving to a predefined home position
    • Poses can be read as 6-element [X, Y, Z, Rx, Ry, Rz] lists.
    • Connection is cleaned up via the `close()` method.

Key details:
    • Automatically retries connection a configurable number of times.
    • Monkey-patches `getl()` to work around issues in certain urx versions by
      pulling pose data from the secondary monitor.
    • All motion helpers default to non-blocking moves (`wait=False`), so explicit
      delays are often needed between sequential commands.
"""


import time
import math
import m3d
from urx.urrobot import URRobot


class UR5eController:
    """Encapsulates UR5e motion, pose, and startup helpers."""

    def __init__(self, ip, tcp_m, payload_kg, max_startup_attempts=3, delay=2.0):
        self.robot = None
        self._connect(ip, tcp_m, payload_kg, max_startup_attempts, delay)


    def _connect(self, ip, tcp_m, payload_kg, max_startup_attempts, delay):
        for attempt in range(1, max_startup_attempts + 1):
            try:
                print(f"[UR5e] Attempt {attempt}/{max_startup_attempts} to connect...")
                self.robot = URRobot(ip)
                time.sleep(1)
                print("[UR5e] Successfully connected!")

                self.robot.set_tcp(tcp_m)
                self.robot.set_payload(payload_kg, (0, 0, 0.1))
                self.go_home()

                # Monkey-patch broken getl()
                def safe_getl():
                    data = self.robot.secmon.get_cartesian_info()
                    return [data["X"], data["Y"], data["Z"], data["Rx"], data["Ry"], data["Rz"]]
                self.robot.getl = safe_getl
                return
            except Exception as e:
                print(f"[UR5e] Connection failed: {e}")
                if attempt < max_startup_attempts:
                    print(f"[UR5e] Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"[UR5e] Failed to connect after {max_startup_attempts} attempts.")


    # -------------------------------------------------------------------
    # Motion helpers
    # -------------------------------------------------------------------

    def move_relative(self, dx=0, dy=0, dz=0, rx=0, ry=0, rz=0, acc=0.2, vel=0.1):
        """Move linearly in Cartesian space by given offsets."""
        
        pose = self.robot.getl()

        pose[0] += dx
        pose[1] += dy
        pose[2] += dz
        pose[3] += rx
        pose[4] += ry
        pose[5] += rz

        try:
            self.robot.movel(pose, acc=acc, vel=vel, wait=False)
        except Exception as e:
            print("[UR5e] Cartesian move failed:", e)


    def move_x_m(self, distance_m, acc=0.2, vel=0.1):
        self.move_relative(dx=distance_m, acc=acc, vel=vel)


    def move_y_m(self, distance_m, acc=0.2, vel=0.1):
        self.move_relative(dy=distance_m, acc=acc, vel=vel)


    def move_z_m(self, distance_m, acc=0.2, vel=0.1):
        self.move_relative(dz=distance_m, acc=acc, vel=vel)


    def move_rx_rad(self, angle_rad, acc=0.2, vel=0.1):
        self.move_relative(rx=angle_rad, acc=acc, vel=vel)


    def move_ry_rad(self, angle_rad, acc=0.2, vel=0.1):
        self.move_relative(ry=angle_rad, acc=acc, vel=vel)


    def move_rz_rad(self, angle_rad, acc=0.2, vel=0.1):
        self.move_relative(rz=angle_rad, acc=acc, vel=vel)


    def move_rx_deg(self, angle_deg, acc=0.2, vel=0.1):
        self.move_relative(rx=math.radians(angle_deg), acc=acc, vel=vel)


    def move_ry_deg(self, angle_deg, acc=0.2, vel=0.1):
        self.move_relative(ry=math.radians(angle_deg), acc=acc, vel=vel)


    def move_rz_deg(self, angle_deg, acc=0.2, vel=0.1):
        self.move_relative(rz=math.radians(angle_deg), acc=acc, vel=vel)


    def pitch_deg(self, angle_deg, acc=0.2, vel=0.1):
        """Rotate around tool's X-axis (pitch)."""

        angle_rad = math.radians(angle_deg)
        pose = self.robot.getl()
        pos = m3d.Vector(pose[:3])
        orient = m3d.Orientation(pose[3:6])
        delta = m3d.Orientation()
        delta.rotate_xb(angle_rad)
        new_orient = orient * delta
        target = m3d.Transform(new_orient, pos)
        self.robot.movel([float(v) for v in target.pose_vector], acc=acc, vel=vel, wait=False)


    def yaw_deg(self, angle_deg, acc=0.2, vel=0.1):
        """Rotate around tool's Y-axis (yaw)."""

        angle_rad = math.radians(angle_deg)
        pose = self.robot.getl()
        pos = m3d.Vector(pose[:3])
        orient = m3d.Orientation(pose[3:6])
        delta = m3d.Orientation()
        delta.rotate_yb(angle_rad)
        new_orient = orient * delta
        target = m3d.Transform(new_orient, pos)
        self.robot.movel([float(v) for v in target.pose_vector], acc=acc, vel=vel, wait=False)


    def roll_deg(self, angle_deg, acc=0.2, vel=0.1):
        """Rotate around tool's Z-axis (roll)."""

        angle_rad = math.radians(angle_deg)
        pose = self.robot.getl()
        pos = m3d.Vector(pose[:3])
        orient = m3d.Orientation(pose[3:6])
        delta = m3d.Orientation()
        delta.rotate_zb(angle_rad)
        new_orient = orient * delta
        target = m3d.Transform(new_orient, pos)
        self.robot.movel([float(v) for v in target.pose_vector], acc=acc, vel=vel, wait=False)


    def rotate_base_deg(self, delta_deg, acc=0.2, vel=0.1):
        delta_rad = math.radians(delta_deg)
        joints = self.robot.getj()
        joints[0] += delta_rad
        self.robot.movej(joints, acc=acc, vel=vel, wait=False)


    def go_home(self, acc=0.5, vel=0.5):
        """Move robot to a known safe joint position."""

        home_joint_angles = (0, -1.57, 0, -1.57, 0, 0)
        self.robot.movej(home_joint_angles, acc=acc, vel=vel, wait=False)
        time.sleep(9)
        print("[UR5e] Moved to home position:", self.get_pose_vector())
        time.sleep(1)



    # -------------------------------------------------------------------
    # POSE HELPER
    # -------------------------------------------------------------------

    def get_pose_vector(self):
        """Return current pose as a 6-element float list."""

        return self.robot.getl()



    # -------------------------------------------------------------------
    # SHUTDOWN
    # -------------------------------------------------------------------

    def close(self):
        if self.robot:
            self.robot.close()
            print("[UR5e] Connection closed.")
