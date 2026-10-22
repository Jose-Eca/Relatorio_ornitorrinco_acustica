#!/usr/bin/env python3
"""
teoria.py — o que a teoria prevê para o cristal, e a comparação com as medições.

Modelo: cristal 2D infinito, rede quadrada de hastes quadradas (lado s) de ABS em ar.
  ∇·(σ∇p) + ω²ηp = 0,   σ = 1/ρ,   η = 1/(ρc²)        (expansão em ondas planas)
  1. Bandas ω(k) ao longo de ΓX e de ΓM → bordos do hiato.
     Só contam os modos SIMÉTRICOS em relação ao plano que contém a direção do som: os
     antissimétricos ("bandas surdas") não são excitados por uma onda que chega de frente.
  2. Dentro do hiato, k é complexo: p ∝ exp(−κx). Resolve-se o problema ao contrário
     (dá-se ω, sai k) e a atenuação por fila é 20·log10(e)·κ·d dB.
  3. Ordens de difração: uma fila de tijolos com período L (L = a em ΓX, √2·a em ΓM) desvia
     som para os lados a partir de f = c/L. Em ΓX é 2·f_B (longe do hiato); em ΓM é
     exatamente f_B = c/(√2·a) — o microfone no eixo deixa de receber tudo o que passa.

  python teoria.py      (usa as mesmas pastas e o mesmo modo de referência da app)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eig, eigh

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import analise as A  # noqa: E402

LADO_MM, S_LADO_MM = 15.8, 0.1          # lado do tijolo 2×2
RHO_AR = 1.204
RHO_S, C_S = 1050.0, 2200.0             # ABS (tratado como fluido: com este contraste é rígido)
N_ONDAS = 7                             # ondas planas: (2·7+1)² = 225
# erro de truncatura da expansão: os bordos com 225 ondas ficam 0.1–0.4 % acima dos de 729 ondas
# (N_ONDAS = 13), e a convergência é lenta (hastes rígidas). Entra como incerteza relativa.
S_TRUNC = 0.005


def _coef(G, a, s, v_ar, v_s):
    """Coeficiente de Fourier de uma grandeza que vale v_s na haste quadrada e v_ar fora."""
    ff = (s / a) ** 2
    F = (v_s - v_ar) * ff * np.sinc(G[..., 0] * s / (2 * np.pi)) * np.sinc(G[..., 1] * s / (2 * np.pi))
    return np.where((G ** 2).sum(-1) < 1e-9, v_ar + (v_s - v_ar) * ff, F)


def _matrizes(a, s, direcao, c_ar, n=N_ONDAS):
    """G, σ(G−G'), η(G−G') e a base dos modos simétricos (espelho ΓX: y→−y; ΓM: x↔y)."""
    idx = np.array([(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1)])
    G = idx * 2 * np.pi / a
    dG = G[:, None, :] - G[None, :, :]
    sig = _coef(dG, a, s, 1 / RHO_AR, 1 / RHO_S)
    eta = _coef(dG, a, s, 1 / (RHO_AR * c_ar ** 2), 1 / (RHO_S * C_S ** 2))
    onde = {tuple(v): k for k, v in enumerate(idx)}
    esp = [onde[(i, -j)] if direcao == "ΓX" else onde[(j, i)] for i, j in idx]
    cols = []
    for k, m in enumerate(esp):
        if m >= k:
            v = np.zeros(len(idx))
            v[k] = v[m] = 1 if m == k else 1 / np.sqrt(2)
            cols.append(v)
    return G, sig, eta, np.array(cols).T


def versor(direcao):
    return np.array([1.0, 0.0]) if direcao == "ΓX" else np.array([1.0, 1.0]) / np.sqrt(2)


def bandas(a, s, direcao, c_ar, nk=41, nb=6):
    """Frequências (Hz) das nb primeiras bandas simétricas, de Γ ao bordo da zona. [nk, nb]"""
    G, sig, eta, S = _matrizes(a, s, direcao, c_ar)
    k_bordo = np.pi / A.dist_planos(1000 * a, direcao)
    B = S.T @ eta @ S
    fs = []
    for t in np.linspace(1e-4, 1, nk):
        kG = t * k_bordo * versor(direcao) + G
        w2 = eigh(S.T @ (sig * (kG @ kG.T)) @ S, B, eigvals_only=True, subset_by_index=[0, nb - 1])
        fs.append(np.sqrt(np.abs(w2)) / (2 * np.pi))
    return np.array(fs)


def hiato_teorico(a, s, direcao, c_ar):
    """1.º hiato para uma onda que chega de frente: (topo da 1.ª banda, fundo da 2.ª)."""
    fs = bandas(a, s, direcao, c_ar)
    return float(fs[:, 0].max()), float(fs[:, 1:].min())


def atenuacao(a, s, direcao, c_ar, f):
    """Atenuação teórica (dB por fila) em cada frequência f: 0 fora do hiato; dentro,
    20·log10(e)·κ·d com κ = menor |Im k| das soluções de k(ω)."""
    G, sig, eta, S = _matrizes(a, s, direcao, c_ar)
    n = versor(direcao)
    Gn = G @ n
    M2 = S.T @ sig @ S                                   # k²
    M1 = S.T @ (sig * (Gn[:, None] + Gn[None, :])) @ S   # k
    GG, B = S.T @ (sig * (G @ G.T)) @ S, S.T @ eta @ S
    m = len(M2)
    Z, I = np.zeros((m, m)), np.eye(m)
    d = A.dist_planos(1000 * a, direcao)
    out = []
    for fi in np.atleast_1d(f):
        M0 = GG - (2 * np.pi * fi) ** 2 * B
        # (k²·M2 + k·M1 + M0)·p = 0  →  problema linear de tamanho 2m
        k = eig(np.block([[Z, I], [-M0, -M1]]), np.block([[I, Z], [Z, M2]]), right=False)
        # só a 1.ª zona de Brillouin: fora dela a base truncada já não descreve bem os modos
        k = k[np.isfinite(k) & (np.abs(k.real) <= 1.001 * np.pi / d)]
        out.append(20 * np.log10(np.e) * np.abs(k.imag).min() * d if k.size else np.nan)
    return np.array(out)


def comparar(R):
    """Teoria e medição para cada bloco (direção, a). Devolve uma lista de dicts."""
    s = LADO_MM / 1000
    out = []
    for (d, a), p in R["por_a"].items():
        c = R[d]["c"]
        lo, hi = hiato_teorico(a / 1000, s, d, c)
        # sensibilidade ao lado do tijolo (± 0.1 mm)
        lo2, hi2 = hiato_teorico(a / 1000, s + S_LADO_MM / 1000, d, c)
        ff = np.linspace(A.JAN_PROF[0], A.JAN_PROF[1], 21) * p["fb"]
        at = atenuacao(a / 1000, s, d, c, ff)
        L = A.dist_planos(a, "ΓX") * (np.sqrt(2) if d == "ΓM" else 1)     # período ao longo da fila
        # parte da janela da profundidade que fica dentro do hiato teórico
        dentro = float(np.mean((ff >= lo) & (ff <= hi)))
        out.append(dict(dir=d, a=a, fb=p["fb"], f_inf_teo=lo, s_inf_teo=np.hypot(lo2 - lo, S_TRUNC * lo),
                        f_sup_teo=hi, s_sup_teo=np.hypot(hi2 - hi, S_TRUNC * hi), dentro=dentro,
                        f_dif=c / L, alfa_teo=float(at.mean()),
                        alfa_max=float(at.max()), f_inf=p["f_inf"], f_sup=p["f_sup"],
                        fecha=p["fecha"], n_fecha=p["n_fecha"], n_hiato=p["n_hiato"],
                        alfa=R["prof"].get((d, a))))
    return out


def texto(C):
    t = ["TEORIA (cristal 2D infinito, hastes rígidas quadradas de "
         f"{LADO_MM} ± {S_LADO_MM} mm; ondas planas) vs MEDIÇÃO", ""]
    for c in C:
        t.append(f"{c['dir']} a = {c['a']:g} mm   (f_B = {c['fb']:.0f} Hz; difração lateral a partir de "
                 f"{c['f_dif']:.0f} Hz = {c['f_dif'] / c['fb']:.2f} f_B)")
        for g, nome in (("f_inf", "bordo de baixo"), ("f_sup", "bordo de cima ")):
            teo, s_teo, med = c[g + "_teo"], c["s_" + g[2:] + "_teo"], c[g]
            lin = f"    {nome}: teoria {teo:.0f} ± {s_teo:.0f} Hz"
            if med:
                z = abs(med["m"] - teo) / np.hypot(med["s"], s_teo)
                lin += (f";  medido {A.pm(med['m'], med['s'])} Hz (n = {med['n']});  "
                        f"medido/teoria = {med['m'] / teo:.3f},  z = {z:.1f}")
            elif g == "f_sup" and c["n_hiato"]:
                lin += f";  medido: não fecha (fecha em {c['n_fecha']} de {c['n_hiato']} medições)"
            else:
                lin += ";  medido: —"
            t.append(lin)
        lin = (f"    atenuação em f_B ± 10 %: teoria {c['alfa_teo']:.2f} dB/fila "
               f"(máx. {c['alfa_max']:.2f})")
        if c["alfa"]:
            aj = c["alfa"]
            # a razão só faz sentido se a janela da profundidade estiver (quase) toda no hiato teórico
            lin += f";  medido {A.pm(aj['m'], aj['sm_f'], 2)} dB/fila" + (
                f"  ({aj['m'] / c['alfa_teo']:.2f} × teoria)" if c["dentro"] >= 0.8 else
                f"  (só {100 * c['dentro']:.0f} % da janela f_B ± 10 % está no hiato teórico: "
                f"não se comparam)")
        t.append(lin)
    # as duas direções para o mesmo a: a razão esperada NÃO é √2 (o hiato desce abaixo de f_B
    # tanto mais quanto mais cheia a rede, e de maneira diferente em cada direção)
    por = {(c["dir"], c["a"]): c for c in C}
    comuns = sorted({c["a"] for c in C if ("ΓX", c["a"]) in por and ("ΓM", c["a"]) in por})
    if comuns:
        t += ["", "ΓX ↔ ΓM, mesmo a (teoria):"]
        for a in comuns:
            x, m = por["ΓX", a], por["ΓM", a]
            q = m["f_inf_teo"] / x["f_inf_teo"]
            lin = f"    a = {a:g} mm: f_inf(ΓM)/f_inf(ΓX) = {q:.3f} (Bragg puro: 1.414)"
            if x["f_inf"] and m["f_inf"]:
                qm = m["f_inf"]["m"] / x["f_inf"]["m"]
                sq = qm * np.hypot(m["f_inf"]["s"] / m["f_inf"]["m"], x["f_inf"]["s"] / x["f_inf"]["m"])
                lin += f";  medido {qm:.3f} ± {sq:.3f} (z = {abs(qm - q) / sq:.1f})"
            sob = x["f_sup_teo"] - m["f_inf_teo"]
            lin += (f";  f_sup(ΓX) − f_inf(ΓM) = {sob:+.0f} Hz → "
                    + ("hiato completo" if sob > 0 else "SEM hiato completo no cristal infinito"))
            t.append(lin)
    return "\n".join(t)


def tabela(C):
    """A comparação em tabela de largura fixa: bordos do hiato e atenuação, teoria e medição."""
    def med(c, g):
        teo, s_teo, m = c[g + "_teo"], c["s_" + g[2:] + "_teo"], c[g]
        if m:
            return f"{teo:>7.0f}  {A.pm(m['m'], m['s']):<13}{abs(m['m'] - teo) / np.hypot(m['s'], s_teo):>4.1f}"
        return f"{teo:>7.0f}  {'não fecha' if g == 'f_sup' and c['n_hiato'] else '—':<13}{'—':>4}"
    t = [f"{'':<10}{'bordo de baixo (Hz)':<30}{'bordo de cima (Hz)':<30}{'atenuação (dB/fila)'}",
         f"{'dir   a':<10}{' teoria  medido          z':<30}{' teoria  medido          z':<30}"
         f"{'teoria (máx.)  medido'}",
         "-" * 92]
    for c in C:
        aj = c["alfa"]
        t.append(f"{c['dir']:<4}{c['a']:>3g}   {med(c, 'f_inf'):<30}{med(c, 'f_sup'):<30}"
                 f"{c['alfa_max']:>8.2f}       {A.pm(aj['m'], aj['sm_f'], 2) if aj else '—'}")
    return "\n".join(t)


def figura(fig, R, C):
    """T − T0 medido (N ≥ 4) com o hiato teórico por cima. Uma linha por direção."""
    import graficos as G
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.clear()
    As = sorted({r["a_mm"] for r in R["linhas"]})
    axs = fig.subplots(len(R["dirs"]), len(As), sharey=True, squeeze=False)
    teo = {(c["dir"], c["a"]): c for c in C}
    Ns = sorted({r["N"] for r in R["linhas"] if r["N"] >= A.N_MIN_F})
    fig.legend(handles=[Patch(color="#4a3aa7", alpha=0.16, lw=0, label="hiato teórico"),
                        Line2D([], [], color="#e34948", ls="--", lw=1, label="início da difração lateral (c/L)"),
                        Line2D([], [], color=G.TINTA2, ls=":", lw=0.9, label=f"limiar (T0 − {A.LIMIAR_DB} dB)")]
               + [Line2D([], [], color=G.cor_N(N), lw=1.6, label=f"N = {N}") for N in Ns],
               loc="outside lower center", ncol=len(Ns) + 3, fontsize=8, frameon=False,
               labelcolor=G.TINTA2, handlelength=1.6, columnspacing=1.2)
    s = LADO_MM / 1000
    for i, d in enumerate(R["dirs"]):
        for j, a in enumerate(As):
            ax = axs[i, j]
            L = [r for r in R["linhas"] if r["dir"] == d and r["a_mm"] == a and r["N"] >= A.N_MIN_F]
            if not L:       # (sem mexer nos eixos: são partilhados com os outros painéis)
                ax.text(0.5, 0.5, f"{d}: sem medições com a = {a:g} mm e N ≥ {A.N_MIN_F}",
                        ha="center", va="center", transform=ax.transAxes, color=G.TINTA2, fontsize=9)
                ax.set_axis_off()
                continue
            c = teo.get((d, a))
            if c is None:       # bloco sem referência do mesmo dia: a teoria desenha-se na mesma
                cs = A.velocidade_som(L[0]["temp_C"])
                lo, hi = hiato_teorico(a / 1000, s, d, cs)
                c = dict(f_inf_teo=lo, f_sup_teo=hi, fb=L[0]["fb"],
                         f_dif=cs / (a / 1000 * (np.sqrt(2) if d == "ΓM" else 1)))
            ax.axvspan(c["f_inf_teo"], c["f_sup_teo"], color="#4a3aa7", alpha=0.16, lw=0)
            ax.axvline(c["f_dif"], color="#e34948", ls="--", lw=1)
            ax.axhline(-A.LIMIAR_DB, color=G.TINTA2, ls=":", lw=0.9)
            ax.axhline(0, color=G.TINTA2, lw=0.6)
            for r in L:
                ax.plot(r["f"], r["Ts"] - r["T0"], color=G.cor_N(r["N"]), lw=1.1,
                        ls="-" if r["valida"] else ":")
            ax.set_xlim(0.4 * c["fb"], min(A.FMAX, 2.3 * c["fb"]))
            ax.set_ylim(-26, 12)
            prov = "" if all(r["valida"] for r in L) else "  (provisório: sem ref. do dia)"
            G._eixo(ax, f"{d}  a = {a:g} mm{prov}", "f (Hz)", "T − T0 (dB)" if j == 0 else None)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import app
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = app.calcular(app.ler_config())
    C = res["teoria"]
    txt = texto(C)
    print(txt)
    app.SAIDA.mkdir(exist_ok=True)
    (app.SAIDA / "teoria.txt").write_text(txt, encoding="utf-8")
    fig = Figure(figsize=(16, 8), layout="constrained")
    figura(fig, res["R"], C)
    fig.savefig(app.SAIDA / "teoria_vs_medido.png", dpi=160)
    print(f"\nGuardado em {app.SAIDA}: teoria.txt, teoria_vs_medido.png")
