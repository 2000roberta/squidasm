"""
Plot latency and throughput vs distance from e91_distance_sweep.csv
(produced by e91_distance_sweep.py).

Reads only -- never re-runs the simulation. Edit and re-run this script
as many times as you want while iterating on the plot.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("e91_distance_sweep.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Latency vs distance (log scale: linear-in-dB fiber loss shows up
# as a much clearer trend here than on a linear axis, where the 1-20km
# region gets compressed near zero) ---
ax = axes[0]
ax.plot(df["distance_km"], df["avg_entanglement_latency_ms"], marker="o", label="Entanglement latency")
ax.plot(df["distance_km"], df["avg_total_latency_ms"], marker="s", label="Total round latency")
ax.plot(df["distance_km"], df["classical_overhead_ms"], marker="^", linestyle="--", label="Classical exchange overhead")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Latency (ms)")
ax.set_yscale("log")
ax.set_title("Latency vs distance")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

# --- Throughput vs distance, with an exponential fit ---
# log(throughput) = log(throughput_0) - decay_rate * distance
# decay_rate is directly comparable to the fiber loss parameter
# (p_loss_length, default 0.2 dB/km in HeraldedLinkConfig) as a sanity
# check that the simulated falloff matches the configured physics.
ax = axes[1]
valid = df["throughput_bps"] > 0
x = df.loc[valid, "distance_km"].values
y = np.log(df.loc[valid, "throughput_bps"].values)
slope, intercept = np.polyfit(x, y, 1)
decay_rate_per_km = -slope
fit_throughput = np.exp(intercept + slope * x)

ax.plot(df["distance_km"], df["throughput_bps"], marker="o", color="tab:green", label="Simulated")
ax.plot(x, fit_throughput, linestyle="--", color="black", alpha=0.6,
        label=f"Exponential fit (decay = {decay_rate_per_km:.4f} /km)")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Raw key throughput (bit/s)")
ax.set_title("Throughput vs distance")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

print(f"Fitted throughput decay rate: {decay_rate_per_km:.4f} per km "
      f"({decay_rate_per_km * 10 / np.log(10):.4f} dB/km equivalent)")
print("Compare against the p_loss_length parameter used in HeraldedLinkConfig "
      "(default 0.2 dB/km) -- note the decay rate reflects TWO-photon "
      "heralding success probability, so a rough expectation is roughly "
      "2x the single-photon fiber loss, not a 1:1 match.")

plt.tight_layout()
plt.savefig("e91_distance_sweep.png", dpi=150)
print("Saved plot to e91_distance_sweep.png")
plt.show()