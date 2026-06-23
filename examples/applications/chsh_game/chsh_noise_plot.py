import pandas
import matplotlib.pyplot as plt

# Read CSV
df = pandas.read_csv("risultati_nv_heralded.csv")

print(df)
print("\nStatistics:")
print(df.describe())

# Three panels sharing x axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# ── TOP PANEL: Win rate ───────────────────────────────────────────────────────
ax1.errorbar(
    df["distance_km"], df["win_rate"],
    yerr=df["margin"],
    marker="o", color="steelblue", capsize=5,
    label="Win rate (±95% CI)"
)
ax1.axhline(y=85.36, color="green",  linestyle="--", label="Quantum limit (85.4%)")
ax1.axhline(y=75.0,  color="orange", linestyle="--", label="Classical limit (75%)")
ax1.set_ylabel("Win rate (%)")
ax1.set_title("CHSH game: NV center + heralded link")
ax1.set_ylim(60, 90)
ax1.legend(fontsize=8)
ax1.grid(True)

# ── MIDDLE PANEL: Fidelity ────────────────────────────────────────────────────
ax2.plot(
    df["distance_km"], df["fidelity"],
    marker="o", color="darkorange",
    label="EPR pair fidelity"
)
#ax2.axhline(y=1.0,  color="green", linestyle="--", label="Perfect fidelity (1.0)")
ax2.set_ylabel("EPR pair fidelity")
ax2.set_ylim(0.9015, 0.9035)
ax2.legend(fontsize=8)
ax2.grid(True)

# ── BOTTOM PANEL: Latency ─────────────────────────────────────────────────────
ax3.plot(
    df["distance_km"], df["latency_ms"],
    marker="o", color="purple",
    label="Latency"
)
ax3.set_xlabel("Distance (km)")
ax3.set_ylabel("Latency (ms)")
ax3.set_yscale("log")  # scala logaritmica perché cresce esponenzialmente
ax3.legend(fontsize=8)
ax3.grid(True, which="both")  # griglia anche per le linee minori in scala log

plt.tight_layout()
plt.savefig("chsh_nv_heralded.png", dpi=150)
print("Plot saved as chsh_nv_heralded.png")