"""Figura: dados em bruto contra dados suavizados, para uma medição (ΓX, a = 48 mm, N = 7).
Usa as mesmas funções da análise (cristal_lego_g/analise 2.0/analise.py), por isso as curvas
suavizadas são exatamente as que entram nos resultados.
  (a) espectros da referência e do cristal: Welch sem suavização (bruto) e com 1/6 de oitava
  (b) T(f) bruto (divisão ponto a ponto) e T(f) suavizado, com T0, o limiar e os bordos do hiato
  python fig_raw_vs_suavizado.py   ->   figuras/fig_raw_vs_smoothed.pdf"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(RAIZ / "cristal_lego_g" / "analise 2.0"))
import analise as A  # noqa: E402

DADOS = RAIZ / "cristal_simples" / "cristal_simples" / "projeto" / "dados"
MED, REF = "a48_N7", "ref_a48"

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.3,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "lines.linewidth": 0.9,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL = 3.37  # largura de uma coluna (in)

med, ref = A.ler_pasta(DADOS / MED), A.ler_pasta(DADOS / REF)
f = med["f"]
Pc, Pr = med["P"].mean(axis=0), ref["P"].mean(axis=0)          # média das repetições (Welch)
Pc_s, Pr_s = A.suavizar_potencia(Pc, f), A.suavizar_potencia(Pr, f)
T_bruto = 10 * np.log10(Pc / Pr)
T_suav = 10 * np.log10(Pc_s / Pr_s)

temp = float(med["meta"]["temp_C"])
fb = A.f_bragg(48, temp, "ΓX")
n = A.numeros(f, T_suav, fb, A.f_controlo(48, temp))
T0 = n["T0"]
k = f / 1000

fig, axs = plt.subplots(2, 1, figsize=(COL, 3.3), sharex=True,
                        gridspec_kw=dict(height_ratios=[1, 1.15]))
# (a) espectros
ax = axs[0]
ref_db = 10 * np.log10(Pr_s.max())
for P, Ps, c, nome in ((Pr, Pr_s, "#08306b", "reference"), (Pc, Pc_s, "#d95f02", f"crystal, $N=7$")):
    ax.plot(k, 10 * np.log10(P) - ref_db, color=c, lw=0.45, alpha=0.35)
    ax.plot(k, 10 * np.log10(Ps) - ref_db, color=c, lw=1.0, label=nome)
ax.set_ylabel("$P$ (dB, rel.)")
ax.set_ylim(-45, 8)
ax.legend(loc="lower left", ncol=2, handlelength=1.4)
ax.text(0.985, 0.94, "(a)  thin: Welch only;  thick: 1/6-octave smoothing", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.3)

# (b) transmissão
ax = axs[1]
ax.plot(k, T_bruto - T0, color="0.6", lw=0.45, label="unsmoothed")
ax.plot(k, T_suav - T0, color="#08306b", lw=1.1, label="smoothed")
ax.axhline(0, color="0.4", lw=0.5)
ax.axhline(-A.LIMIAR_DB, color="0.3", lw=0.6, ls=":", label=r"$T_0-6$ dB")
ax.axvline(fb / 1000, color="k", lw=0.7, ls="--", label=r"$f_B=c/2a$")
if n["hiato"]:
    for v in n["hiato"]:
        ax.axvline(v / 1000, color="#de2d26", lw=0.8)
ax.set_xlim(1.0, 8.0)
ax.set_ylim(-36, 22)
ax.set_xlabel("$f$ (kHz)")
ax.set_ylabel(r"$T-T_0$ (dB)")
ax.legend(loc="lower left", ncol=1, handlelength=1.4, labelspacing=0.25, borderaxespad=0.3)
ax.text(0.985, 0.955, r"(b)  $\Gamma X$, $a=48$ mm, $N=7$;  red: gap edges", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.3, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
fig.tight_layout(h_pad=0.3)
(AQUI / "figuras").mkdir(exist_ok=True)
fig.savefig(AQUI / "figuras" / "fig_raw_vs_smoothed.pdf")
fig.savefig(AQUI / "figuras" / "fig_raw_vs_smoothed.png", dpi=300)

# números para a legenda
nb = 10 * np.log10(Pc / Pr) - T_suav
print(f"f_inf = {n['f_inf']:.0f} Hz, f_sup = {n['f_sup']:.0f} Hz, T0 = {T0:.1f} dB, "
      f"desvio bruto-suavizado (RMS, 1-8 kHz) = {np.sqrt(np.mean(nb[(f > 1000) & (f < 8000)] ** 2)):.1f} dB")
