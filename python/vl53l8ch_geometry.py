import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compute_vl53l8ch_geometry(distance_m: float) -> pd.DataFrame:
    num_zones = 8
    fov_deg = 45.0
    zone_angle = fov_deg / num_zones
    half_zone = zone_angle / 2

    data = []

    for row in range(num_zones):         
        for col in range(num_zones):     
            theta_x_center = -fov_deg / 2 + (col + 0.5) * zone_angle
            theta_y_center = fov_deg / 2 - (row + 0.5) * zone_angle

            theta_x_left = np.deg2rad(theta_x_center - half_zone)
            theta_x_right = np.deg2rad(theta_x_center + half_zone)
            theta_y_bottom = np.deg2rad(theta_y_center - half_zone)
            theta_y_top = np.deg2rad(theta_y_center + half_zone)

            x_left = distance_m * np.tan(theta_x_left)
            x_right = distance_m * np.tan(theta_x_right)
            y_bottom = distance_m * np.tan(theta_y_bottom)
            y_top = distance_m * np.tan(theta_y_top)

            width = x_right - x_left
            height = y_top - y_bottom
            x = x_left
            y = y_bottom

            data.append({
                "Row": row,
                "Col": col,
                "X (m)": x,
                "Y (m)": y,
                "Width (m)": width,
                "Height (m)": height
            })

    return pd.DataFrame(data)

def vl53l8ch_zone_geometry_plot(df, title="VL53L8CH Zone Footprint", show_zone_id=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_facecolor('whitesmoke')

    for i, row in df.iterrows():
        zone_id = i
        label = f"{int(row['Row'])},{int(row['Col'])}"
        if show_zone_id:
            label += f"\n{zone_id}"
        
        rect = patches.Rectangle(
            (row["X (m)"], row["Y (m)"]),
            row["Width (m)"],
            row["Height (m)"],
            edgecolor='black',
            facecolor='skyblue',
            linewidth=1,
            alpha=0.8
        )
        ax.add_patch(rect)
        ax.text(
            row["X (m)"] + row["Width (m)"]/2,
            row["Y (m)"] + row["Height (m)"]/2,
            label,
            fontsize=8,
            ha='center',
            va='center'
        )

    # Mark center
    ax.plot(0, 0, 'r+', markersize=12, label='Sensor Center')

    # Axis & Grid
    ax.set_xlabel("X position (m)", fontsize=12)
    ax.set_ylabel("Y position (m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.legend(loc='upper right')
    
    pad = 0.05
    ax.set_xlim(df["X (m)"].min() - pad, (df["X (m)"] + df["Width (m)"]).max() + pad)
    ax.set_ylim(df["Y (m)"].min() - pad, (df["Y (m)"] + df["Height (m)"]).max() + pad)

    plt.tight_layout()
    plt.show()

# Example usage:
distance = 1.0  # meters
df = compute_vl53l8ch_geometry(distance)
vl53l8ch_zone_geometry_plot(df, title=f"VL53L8CH Zone Footprint at {distance} m")
