import numpy as np
import matplotlib.pyplot as plt

def plot_ideal_cnh(D, b_start, B, sigma_t = 0.85, delta_t = 0.25):
    """
    Plots an "ideal" CNH with the PDF equation from 
    Models of Direct Time-of-Flight Sensor Precision That Enable Optimal Design and Dynamic Configuration (Koerner, 2021)

    Parameters:
        D (float): Target distance in meters.
        b_start (int): Start bin index.
        B (int): Number of bins.
        sigma_t (float): Temporal standard deviation (ns).
        delta_t (float): Bin width (ns).
    """
    c = 3e8                # speed of light in m/s
    t0 = (2 * D / c) * 1e9 # round-trip time in ns

    # Bin center times and absolute bin indices
    b_absolute = np.arange(b_start, b_start + B)
    t_b = (b_absolute + 0.5) * delta_t

    # Gaussian photon reception
    H = np.exp(-((t_b - t0) ** 2) / (2 * sigma_t ** 2))
    
    CNH = H

    # Print CNH values with absolute bin numbers
    print("\nIdeal CNH Histogram:")
    print(f"{'Bin':>5} | {'Time (ns)':>10} | {'CNH':>10}")
    print("-" * 32)
    for b, tb, val in zip(b_absolute, t_b, CNH):
        print(f"{b:>5} | {tb:>10.3f} | {val:>10.6f}")

    # Plot CNHs
    plt.figure(figsize=(10, 5))
    plt.stem(b_absolute, CNH, basefmt=" ")
    plt.xticks(b_absolute)
    plt.title(f"Ideal CNH (Model): D = {D:.2f} m, Start Bin = {b_start}, Bins = {B}")
    plt.xlabel("Bin Number")
    plt.ylabel("Normalized Count")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# Prompt user for CNH variables
try:
    D = float(input("Enter target distance in meters (e.g., 1.0): "))
    b_start = int(input("Enter start bin index (e.g., 20): "))
    B = int(input("Enter number of bins (e.g., 18): "))

    plot_ideal_cnh(D, b_start, B)

except ValueError:
    print("Invalid input. Please enter numeric values.")
