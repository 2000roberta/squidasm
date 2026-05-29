import pandas
import matplotlib.pyplot as plt

# Leggi il CSV con pandas — una riga invece di un loop!
df = pandas.read_csv("risultati.csv")

# Stampa la tabella nel terminale (utile per verificare i dati)
print(df)

# Statistiche di base — pandas le calcola automaticamente
print("\nStatistiche:")
print(df.describe())

# Grafico
plt.figure(figsize=(8, 5))

#curva principale
#plt.plot(df["noise"], df["win_rate"], marker="o", color="steelblue", label="Win rate simulato")

# barre di errore al 95%
plt.errorbar(
    df["noise"],          # asse X
    df["win_rate"],       # asse Y
    yerr=df["margin"],    # dimensione delle barre di errore
    marker="o",           # pallino su ogni punto
    color="steelblue",
    capsize=5,            # la lineetta orizzontale in cima e in fondo alla barra
    label="Win rate simulato (±95% CI)"
)

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