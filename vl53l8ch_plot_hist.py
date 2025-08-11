import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV ===
#csv_path = "C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/GUIs/MZAI_EVK_v1.0.1/data/log__VL53L8CH__AIKit__CNH_8x8__20250603_203403/data_VL53L8CH__AIKit__CNH_8x8__20250603_203403.csv"
#csv_path = "C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/GUIs/MZAI_EVK_v1.0.1/data/log__VL53L8CH__AIKit__CNH_8x8__20250611_193627/data_VL53L8CH__AIKit__CNH_8x8__20250611_193627.csv"
#csv_path = "C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/GUIs/MZAI_EVK_v1.0.1/data/log__VL53L8CH__AIKit__CUSTOM_CNH_8x8__20250627_161620/data_VL53L8CH__AIKit__CUSTOM_CNH_8x8__20250627_161620.csv"
csv_path = "C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/GUIs/MZAI_EVK_v1.0.1/data/log__VL53L8CH__AIKit__CUSTOM_CNH_8x8__20250627_162132/data_VL53L8CH__AIKit__CUSTOM_CNH_8x8__20250627_162132.csv"

print("Looking for file at:", csv_path)
df = pd.read_csv(csv_path)

# === Prompt user for zone layout ===
try:
    num_rows = int(input("Please enter number of zone rows (1-8): "))
    num_cols = int(input("Please enter number of zone columns (1-8): "))
    total_zones = num_rows * num_cols
except ValueError:
    print("Invalid zone layout input.")
    exit()

# === Prompt user for frame and zone index ===
try:
    frame_index = int(input("Please enter frame index: "))
    zone_index = int(input(f"Please enter ROI zone index (0–{total_zones - 1}): "))
except ValueError:
    print("Invalid input.")
    exit()

# === Validate zone index ===
if zone_index < 0 or zone_index >= total_zones:
    print(f"Zone index {zone_index} is out of bounds for a {num_rows}x{num_cols} layout.")
    exit()

# === Detect number of histogram bins dynamically ===
bin_prefix = f"cnh__hist_bin_"
zone_suffix = f"_a{zone_index}"
bin_columns = []

for col in df.columns:
    # Example column format: 'cnh__hist_bin_0_a12'
    if col.startswith(bin_prefix) and col.endswith(zone_suffix):
        bin_columns.append(col)

bin_count = len(bin_columns)

if bin_count == 0:
    print("No histogram bins found for this zone.")
    exit()

# === Extract bin values ===
bin_values = []
for bin_idx in range(bin_count):
    col_name = f"{bin_prefix}{bin_idx}{zone_suffix}"
    if col_name in df.columns:
        bin_values.append(df.loc[frame_index, col_name])
    else:
        bin_values.append(0.0)  # Default to 0 if bin is missing

# === Calculate centroid ===
indices = list(range(bin_count))   # Bin indices: 0 through N-1
weights = bin_values               # Corresponding bin values (can be negative)

# Compute weighted sum of bin indices
weighted_sum = 0.0
for i in indices:
    weighted_sum += i * weights[i]

# Compute total weight of histogram
total_weight = sum(weights)

# Final centroid calculation
if total_weight != 0:
    centroid = weighted_sum / total_weight
else:
    centroid = float('nan')  # Handle division by zero gracefully

# === Output centroid info ===
print(f"Centroid of histogram (Zone {zone_index}, Frame {frame_index}): {centroid:.2f}")
#print(f"Estimated target depth according to centroid: {centroid * bin_width * 3.0e10 / 2}")
print(f"Total bins: {bin_count} | Total zones: {total_zones} ({num_rows}x{num_cols})")

# === Plot histogram ===
plt.figure(figsize=(10, 4))
plt.plot(indices, weights, marker='o', linestyle='-', linewidth=2.0, color='black', markerfacecolor='blue', markersize=6)
plt.xticks(indices)
plt.title(f"Histogram for Zone {zone_index} (Frame {frame_index})", fontsize=14)
plt.xlabel("Bin Index", fontsize=12)
plt.ylabel("Normalized Return", fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()