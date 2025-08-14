import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to one run's data_*.csv file
DATA_CSV = Path("c:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/shared/Koerner, Lucas J.'s files - lloyd_gavin/data/experiment_20250814_004115/yaw_step_20250814_004115__wide.csv")

# Load the CSV
df = pd.read_csv(DATA_CSV)

# --- Identify CNH bin columns ---
# Usually something like "Bin_0", "Bin_1", etc.
cnh_cols = [col for col in df.columns if col.lower().startswith("bin")]

if not cnh_cols:
    raise ValueError("No CNH bin columns found in the CSV.")

print(f"Found {len(cnh_cols)} CNH bins: {cnh_cols}")

# --- Aggregate histogram data ---
# Option 1: average across all frames & zones
avg_cnh = df[cnh_cols].mean()

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.bar(range(len(avg_cnh)), avg_cnh, width=0.8)
plt.xlabel("Bin index")
plt.ylabel("Normalized return rate (CNH)")
plt.title("Average CNH Histogram")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
