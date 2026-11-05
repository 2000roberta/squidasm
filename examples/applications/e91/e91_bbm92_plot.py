"""
Plot QBER and raw key throughput vs distance from e91_bbm92_sweep.csv
(produced by e91_bbm92_sweep_data.py). Reads only: never runs a simulation.

Four series: E91 / BBM92, each on the NV + heralded link (curve vs distance)
and on the noiseless network (horizontal reference line: that network has no
distance parameter).
"""
import pandas as pd
import matplotlib.pyplot as plt

CSV = "e91_bbm92_sweep.csv"
OUT_PNG = "e91_bbm92_qber_throughput.png"
STYLE = {"e91": ("tab:blue", "o"), "bbm92": ("tab:orange", "s")}

df = pd.read_csv(CSV)
nv = df[df["network"] == "nv"]
ref = df[df["network"] == "noiseless"]

x_min, x_max = (nv["distance_km"].min(), nv["distance_km"].max()) if len(nv) else (0, 1)

fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(13, 5))

for protocol, (color, marker) in STYLE.items():
    name = protocol.upper()
    d = nv[nv["protocol"] == protocol].sort_values("distance_km")
    r = ref[ref["protocol"] == protocol]

    if len(d):
        q = d["qber_sample"] * 100
        yerr = [(q - d["qber_sample_ci_low"] * 100).clip(lower=0), (d["qber_sample_ci_high"] * 100 - q).clip(lower=0)]
        ax_q.errorbar(d["distance_km"], q, yerr=yerr, color=color, marker=marker, capsize=3,
                      label=f"{name} - NV + heralded")
        ax_t.plot(d["distance_km"], d["throughput_bps"], color=color, marker=marker,
                  label=f"{name} - NV + heralded")

    if len(r):
        r = r.iloc[0]
        ax_q.hlines(r["qber_sample"] * 100, x_min, x_max, colors=color, linestyles="--", label=f"{name} - noiseless")
        ax_q.fill_between([x_min, x_max], r["qber_sample_ci_low"] * 100, r["qber_sample_ci_high"] * 100,
                          color=color, alpha=0.15)
        ax_t.hlines(r["throughput_bps"], x_min, x_max, colors=color, linestyles="--",
                    label=f"{name} - noiseless")

# QBER panel
ax_q.axhline(11, color="gray", linestyle=":", linewidth=1, label="11% asymptotic threshold")
ax_q.set_xlabel("Distance (km)")
ax_q.set_ylabel("QBER (%)")
ax_q.set_title("QBER vs distance")
ax_q.set_ylim(bottom=-0.5)
ax_q.grid(True, alpha=0.3)

# Throughput panel
ax_t.set_xlabel("Distance (km)")
ax_t.set_ylabel("Raw key throughput (bit/s)")
ax_t.set_title("Throughput vs distance")
ax_t.set_yscale("log")
ax_t.grid(True, alpha=0.3, which="both")

fig.suptitle(f"{int(df['n_rounds'].max())} rounds per point; error bars / bands = 95% Wilson CI", fontsize=10)
# One shared legend (both panels have the same series), below the figure
handles, labels = ax_q.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
plt.tight_layout(rect=[0, 0.1, 1, 0.94])
plt.savefig(OUT_PNG, dpi=150)
print(f"Saved plot to {OUT_PNG}")
plt.show()