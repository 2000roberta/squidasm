"""
Plot QBER and raw key throughput vs delivered fidelity from
e91_bbm92_fidelity_sweep.csv (produced by e91_bbm92_fidelity_sweep_data.py).
Reads only: never runs a simulation, and works on a partial CSV too.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "e91_bbm92_fidelity_sweep.csv"
OUT_PNG = "e91_bbm92_fidelity_qber_throughput.png"

# True : QBER the protocol would estimate (sacrificed sample), wider bars
# False: QBER on the whole sifted key, about half as wide
USE_SAMPLE_QBER = True

# True : test sulle CI al 95% (come security_check: S_low > 2, QBER_high < 11%)
# False: test sui valori puntuali (S > 2, QBER < 11%)
USE_CI = False
THRESHOLD_QBER = 0.11

STYLE = {"e91": ("tab:blue", "o"), "bbm92": ("tab:orange", "s")}

df_all = pd.read_csv(CSV)
DIST_LINE_PLOTS = 10.0
df = df_all[df_all["distance_km"] == DIST_LINE_PLOTS]
col = "qber_sample" if USE_SAMPLE_QBER else "qber"

fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(13, 5))

for protocol, (color, marker) in STYLE.items():
    d = df[df["protocol"] == protocol].sort_values("fidelity")
    if d.empty:
        continue
    name = protocol.upper()

    q = d[col] * 100
    yerr = [(q - d[col + "_ci_low"] * 100).clip(lower=0), (d[col + "_ci_high"] * 100 - q).clip(lower=0)]
    ax_q.errorbar(d["fidelity"], q, yerr=yerr, color=color, marker=marker, capsize=3, label=name)
    if protocol == "e91":
        ok = (d["s_ci_low"] > 2) if USE_CI else (d["s_abs"] > 2)
        crit = r"$S_{\mathrm{low}}>2$" if USE_CI else r"$|S|>2$"
    else:
        ok = (d[col + "_ci_high"] < THRESHOLD_QBER) if USE_CI else (d[col] < THRESHOLD_QBER)
        crit = ("QBER$_{\\mathrm{high}}$" if USE_CI else "QBER") + f" < {THRESHOLD_QBER * 100:.0f}%"
    # NaN dove il test fallisce: nessun marker e la linea si interrompe
    ax_t.plot(d["fidelity"], d["throughput_bps"].where(ok), color=color, marker=marker,
              label=f"{name}: {crit}")

    # fidelity minima (tra i punti dello sweep) per cui il test è superato
    if ok.any():
        f_min = d.loc[ok, "fidelity"].min()
        ax_t.axvline(f_min, color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax_t.annotate(f"$F_{{\\mathrm{{min}}}}$ = {f_min:.3f}", xy=(f_min, 1),
                      xycoords=("data", "axes fraction"),
                      xytext=(4, -14 if protocol == "e91" else -28),
                      textcoords="offset points", color=color, fontsize=8)
"""        
# Reference: Werner state with delivered fidelity F, ideal measurements -> QBER = 2(1-F)/3
f_grid = np.linspace(df["fidelity"].min(), 1.0, 100)
ax_q.plot(f_grid, 2 * (1 - f_grid) / 3 * 100, color="gray", linestyle="--",
          label="Werner state, link only: 2(1-F)/3")
ax_q.axhline(11, color="gray", linestyle=":", linewidth=1, label="11% asymptotic threshold")
"""
ax_q.set_xlabel("Fidelity")
ax_q.set_ylabel("QBER (%)")
ax_q.set_title("QBER vs fidelity")
ax_q.set_ylim(bottom=0)
ax_q.grid(True, alpha=0.3)

ax_t.set_xlabel("Fidelity")
ax_t.set_ylabel("Raw key throughput (bit/s)")
ax_t.set_title("Throughput vs fidelity (only points passing the security test)")
ax_t.legend(fontsize=8, loc="lower right")
ax_t.set_ylim(bottom=0)
ax_t.grid(True, alpha=0.3)

qber_src = "sacrificed sample" if USE_SAMPLE_QBER else "whole sifted key"
fig.suptitle(f"{int(df['n_rounds'].max())} rounds per point, {df['distance_km'].iloc[0]:g} km; "
             f"QBER from {qber_src}; error bars = 95% Wilson CI", fontsize=10)

# One shared legend below the figure
handles, labels = ax_q.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
plt.tight_layout(rect=[0, 0.08, 1, 0.94])
plt.savefig(OUT_PNG, dpi=150)
print(f"Saved plot to {OUT_PNG}")

# ---- S vs fidelity (solo E91) ----
OUT_PNG_S = "e91_SvsFidelity.png"
d = df[df["protocol"] == "e91"].sort_values("fidelity")

fig_s, ax_s = plt.subplots(figsize=(6.5, 5))
ax_s.errorbar(d["fidelity"], d["s_abs"], yerr=d["s_margin"], color=STYLE["e91"][0],
              marker=STYLE["e91"][1], capsize=3, label="E91: |S| misurato")

# Riferimento: stato Werner con fidelity F e misure ideali, S = 2*sqrt(2)*(4F-1)/3
f_grid = np.linspace(d["fidelity"].min(), 1.0, 100)
#ax_s.plot(f_grid, 2 * np.sqrt(2) * (4 * f_grid - 1) / 3, color="gray", linestyle="--",
#          label="Werner, misure ideali")
ax_s.axhline(2, color="red", linestyle=":", label="limite classico S = 2")
ax_s.axhline(2 * np.sqrt(2), color="gray", linestyle=":", linewidth=1,
             label=r"limite di Tsirelson $2\sqrt{2}$")

ax_s.set_xlabel("Fidelity")
ax_s.set_ylabel("|S| (CHSH)")
ax_s.set_title("CHSH S vs fidelity")
ax_s.grid(True, alpha=0.3)
ax_s.legend(fontsize=8, loc="upper left")
fig_s.suptitle("E91, barre di errore = intervallo di confidenza al 95%", fontsize=9)
fig_s.tight_layout()
fig_s.savefig(OUT_PNG_S, dpi=150)
print(f"Saved plot to {OUT_PNG_S}")

# ---- distanza x fidelity consegnata: throughput della chiave valida ----
OUT_PNG_MAP = "e91_bbm92_distancevsfidelity.png"
FID_MIN = 0.8
fig_m, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
#vmax = df_all.loc[df_all["fidelity"] >= FID_MIN, "throughput_bps"].max()
from matplotlib.colors import LogNorm
tp_pos = df_all.loc[(df_all["fidelity"] >= FID_MIN) & (df_all["throughput_bps"] > 0), "throughput_bps"]
norm = LogNorm(vmin=tp_pos.min(), vmax=tp_pos.max())
sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")

for ax, protocol in zip(axes, ["e91", "bbm92"]):
    d = df_all[(df_all["protocol"] == protocol) & (df_all["fidelity"] >= FID_MIN)].copy()
    if protocol == "e91":
        d["ok"] = (d["s_ci_low"] > 2) if USE_CI else (d["s_abs"] > 2)
        crit = r"$S_{\mathrm{low}}>2$" if USE_CI else r"$|S|>2$"
    else:
        d["ok"] = (d[col + "_ci_high"] < THRESHOLD_QBER) if USE_CI else (d[col] < THRESHOLD_QBER)
        crit = f"QBER < {THRESHOLD_QBER * 100:.0f}%"

    bad, good = d[~d["ok"]], d[d["ok"]]
    ax.scatter(bad["distance_km"], bad["fidelity"], marker="x", color="gray", s=40,
               label="test non superato")
    ax.scatter(good["distance_km"], good["fidelity"], c=good["throughput_bps"],
               cmap="viridis", norm=norm, s=90, edgecolor="k")
    ax.set_xlabel("Distance (km)")
    ax.set_title(f"{protocol.upper()}: {crit}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(FID_MIN - 0.01, 1.005)
    ax.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel("Delivered fidelity")
fig_m.colorbar(sm, ax=axes, label="Throughput della chiave valida (bit/s)")
fig_m.savefig(OUT_PNG_MAP, dpi=150, bbox_inches="tight")
print(f"Saved plot to {OUT_PNG_MAP}")

plt.show()