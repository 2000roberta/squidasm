import csv
import matplotlib.pyplot as plt

noise_levels = []
win_rates = []

with open("risultati.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        noise_levels.append(float(row["noise"]))
        win_rates.append(float(row["win_rate"]))

plt.figure(figsize=(8, 5))
plt.plot(noise_levels, win_rates, marker="o", color="steelblue", label="Win rate simulato")
plt.axhline(y=85.36, color="green",  linestyle="--", label="Limite quantistico (85.4%)")
plt.axhline(y=75.0,  color="orange", linestyle="--", label="Limite classico (75%)")
plt.axhline(y=50.0,  color="red",    linestyle="--", label="Casuale (50%)")
plt.xlabel("Livello di rumore (0 = perfetto, 1 = massimo)")
plt.ylabel("Percentuale di vittorie (%)")
plt.title("CHSH game: prestazioni vs rumore")
plt.ylim(40, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("chsh_noise_sweep.png", dpi=150)
plt.show()
print("Grafico salvato come chsh_noise_sweep.png")