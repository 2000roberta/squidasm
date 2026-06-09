import pandas
import matplotlib.pyplot as plt

# Read CSV with pandas
df = pandas.read_csv("risultati_fidelity.csv")

# Print table and statistics
print(df)
print("\nStatistics:")
print(df.describe())

# Create figure with two stacked panels sharing the same x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

# ── TOP PANEL: Win rate ───────────────────────────────────────────────────────
ax1.errorbar(
    df["noise"], df["win_rate_link"],
    yerr=df["margin_link"],
    marker="o", color="steelblue", capsize=5,
    label="Link noise (±95% CI)"
)
ax1.errorbar(
    df["noise"], df["win_rate_qdevice"],
    yerr=df["margin_qdevice"],
    marker="s", color="darkorange", capsize=5,
    label="QDevice noise (±95% CI)"
)
ax1.axhline(y=85.36, color="green",  linestyle="--", label="Quantum limit (85.4%)")
ax1.axhline(y=75.0,  color="gray",   linestyle="--", label="Classical limit (75%)")
ax1.axhline(y=50.0,  color="red",    linestyle="--", label="Random (50%)")
ax1.set_ylabel("Win rate (%)")
ax1.set_title("CHSH game: link noise vs qdevice noise")
ax1.set_ylim(40, 100)
ax1.legend(fontsize=8)
ax1.grid(True)

# ── BOTTOM PANEL: Fidelity ────────────────────────────────────────────────────
ax2.plot(
    df["noise"], df["fidelity_link"],
    marker="o", color="steelblue",
    label="Link noise"
)
ax2.plot(
    df["noise"], df["fidelity_qdevice"],
    marker="s", color="darkorange",
    label="QDevice noise"
)
ax2.axhline(y=1.0,  color="green", linestyle="--", label="Perfect fidelity (1.0)")
ax2.axhline(y=0.25, color="red",   linestyle="--", label="Minimum fidelity (0.25)")
ax2.set_xlabel("Noise level (0 = perfect, 1 = maximum)")
ax2.set_ylabel("EPR pair fidelity")
ax2.set_ylim(0.0, 1.1)
ax2.legend(fontsize=8)
ax2.grid(True)

plt.tight_layout()
plt.savefig("chsh_fidelity.png", dpi=150)
plt.show()
print("Plot saved as chsh_fidelity.png")