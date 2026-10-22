"""
graficos.py — os gráficos da análise conjunta. Cada função desenha numa Figure do matplotlib
(serve tanto para a janela como para guardar em PNG). As duas direções aparecem sempre com os
mesmos eixos e as mesmas janelas, lado a lado ou sobrepostas.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import analise as A

# cor = identidade (sempre a mesma para a mesma coisa), nunca a ordem em que aparece
COR_DIR = {"ΓX": "#2a78d6", "ΓM": "#eb6834"}
COR_A = {32: "#2a78d6", 40: "#eb6834", 48: "#1baf7a", 56: "#eda100"}
MARCA_A = {32: "o", 40: "s", 48: "^", 56: "D"}       # a forma repete a cor (daltonismo, impressão)
MARCA_DIR = {"ΓX": "o", "ΓM": "s"}
OUTRAS = ["#4a3aa7", "#e87ba4", "#008300", "#e34948"]
TINTA, TINTA2, GRELHA = "#0b0b0b", "#52514e", "#d9d8d2"
AZUIS = LinearSegmentedColormap.from_list("azuis", ["#bcd6f5", "#2a78d6", "#0b2a55"])
N_MAX_COR = 11      # a cor de cada N é fixa: N = 1 claro … N = 11 escuro


def cor_a(a):
    return COR_A.get(int(round(a)), OUTRAS[int(round(a)) % len(OUTRAS)])


def marca_a(a):
    return MARCA_A.get(int(round(a)), "v")


def cor_N(N):
    return AZUIS(min(max((N - 1) / (N_MAX_COR - 1), 0), 1))


def _eixo(ax, titulo=None, x=None, y=None):
    ax.grid(color=GRELHA, lw=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA2)
    ax.tick_params(colors=TINTA2, labelsize=8)
    if titulo:
        ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
    if x:
        ax.set_xlabel(x, fontsize=9, color=TINTA2)
    if y:
        ax.set_ylabel(y, fontsize=9, color=TINTA2)


def _vazio(ax, texto):
    ax.text(0.5, 0.5, texto, ha="center", va="center", transform=ax.transAxes, color=TINTA2,
            fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_color(GRELHA)


def _janelas(ax, fb=1.0, ctrl=1.0):
    """Janela da profundidade e f_B (em Hz se fb for dado, senão em f/f_B).
    ctrl = c/(2a) nas mesmas unidades (a banda de controlo é 0.35–0.70 × c/(2a))."""
    ax.axvspan(A.JAN_PROF[0] * fb, A.JAN_PROF[1] * fb, color="#4a3aa7", alpha=0.10, lw=0,
               label="janela da profundidade")
    ax.axvline(fb, color=TINTA2, ls="--", lw=0.9, label="f_B = c/(2d)")


# ------------------------------------------------------------ T(f) de um a, as duas direções
def fig_T(fig, R, a):
    """T − T0 em função de f/f_B para todos os N de um a: uma coluna por direção, mesmos eixos."""
    fig.clear()
    dirs = R["dirs"]
    axs = fig.subplots(1, len(dirs), sharey=True, squeeze=False)[0]
    for ax, d in zip(axs, dirs):
        L = [r for r in R["linhas"] if r["dir"] == d and abs(r["a_mm"] - a) < 0.01]
        if not L:
            _vazio(ax, f"{d}: não há medições com a = {a:g} mm")
            ax.set_title(f"{d}", fontsize=10, color=TINTA, loc="left")
            continue
        _janelas(ax, 1.0, L[0]["f_ctrl"] / L[0]["fb"])
        ax.axhline(0, color=TINTA2, lw=0.6)
        ax.axhline(-A.LIMIAR_DB, color=TINTA2, ls=":", lw=0.9, label=f"limiar (T0 − {A.LIMIAR_DB} dB)")
        for r in L:
            ax.plot(r["f"] / r["fb"], r["Ts"] - r["T0"], color=cor_N(r["N"]), lw=1.3,
                    ls="-" if r["valida"] else ":", label=f"N = {r['N']}")
        ax.set_xlim(0.3, min(2.0, A.FMAX / L[0]["fb"]))
        ax.set_ylim(-28, 12)
        _eixo(ax, f"{d}   a = {a:g} mm   (d = {A.nome_d(d)} = {1000 * L[0]['d_m']:.1f} mm, "
                  f"f_B = {L[0]['fb']:.0f} Hz)", "f / f_B", "T − T0 (dB)" if ax is axs[0] else None)
        ax.legend(fontsize=7, ncol=3, loc="lower left", frameon=False, labelcolor=TINTA2)


# ------------------------------------------------------------ profundidade vs N
def fig_prof(fig, R):
    """D(N) com as retas ajustadas (faixa = ±1σ do ajuste) e, por baixo, os resíduos
    normalizados (D − ajuste)/σ. Uma coluna por direção."""
    fig.clear()
    dirs = R["dirs"]
    axs = fig.subplots(2, len(dirs), sharex=True, sharey="row", squeeze=False,
                       gridspec_kw=dict(height_ratios=[3, 1]))
    n_max = max([r["N"] for r in R["linhas"]] + [1])
    for j, d in enumerate(dirs):
        ax, ar = axs[0, j], axs[1, j]
        for a in R[d]["As"]:
            L = [r for r in R["linhas"] if r["dir"] == d and r["a_mm"] == a and r["valida"]]
            c, mk = cor_a(a), marca_a(a)
            N = np.array([r["N"] for r in L])
            aj = R["prof"].get((d, a))
            rot = f"a = {a:g} mm"
            if aj:
                rot += f":  α = {A.pm(aj['m'], aj['sm_f'], 2)} dB/fila,  χ²/ν = {aj['chi2nu']:.1f}"
                xx = np.linspace(0, n_max + 0.5, 50)
                yy, ss = aj["m"] * xx + aj["b"], A.faixa(aj, xx)
                ax.fill_between(xx, yy - ss, yy + ss, color=c, alpha=0.15, lw=0)
                ax.plot(xx, yy, color=c, lw=1.4)
                ar.plot(N, aj["res"], mk, color=c, ms=5, mec="white", mew=0.6)
            ax.errorbar(N, [r["profundidade"] for r in L], [r["s_profundidade"] for r in L], fmt=mk,
                        color=c, ms=6, mec="white", mew=0.8, capsize=2.5, elinewidth=1, label=rot)
        ax.axhline(0, color=TINTA2, lw=0.6)
        _eixo(ax, f"{d}:  profundidade do hiato vs número de filas", None,
              "D (dB)" if j == 0 else None)
        if R[d]["As"]:
            ax.legend(fontsize=7.5, loc="upper left", frameon=False, labelcolor=TINTA2)
        else:
            _vazio(ax, f"{d}: sem medições válidas")
        ar.axhline(0, color=TINTA2, lw=0.6)
        for v in (-1, 1):
            ar.axhline(v, color=TINTA2, lw=0.6, ls=":")
        ar.set_ylim(-4, 4)
        _eixo(ar, None, "N (filas)", "resíduo / σ" if j == 0 else None)
        ar.set_xlim(0, n_max + 0.5)


# ------------------------------------------------------------ f vs 1/(2d)
def fig_freq(fig, R):
    """Frequências do hiato de cada bloco em função de 1/(2d), as duas direções no mesmo eixo.
    Se f = f_B os pontos caem na reta f = c·x. Esquerda: f_inf; direita: f_centro."""
    fig.clear()
    axs = fig.subplots(2, 2, sharex=True, sharey="row", gridspec_kw=dict(height_ratios=[3, 1]))
    x_max = max([fr["x"].max() for fr in R["freq"].values() if fr["As"]] + [1.0]) * 1.12
    xx = np.linspace(0, x_max, 50)
    for j, (g, nome) in enumerate((("f_inf", "f_inf (bordo de baixo)"), ("fc", "f_centro"))):
        ax, ar = axs[0, j], axs[1, j]
        cs = [R[d]["c"] for d in R["dirs"] if np.isfinite(R[d]["c"])]
        if cs:
            ax.plot(xx, np.mean(cs) * xx, color=TINTA2, ls="--", lw=1,
                    label=f"f = c·x  (c = {np.mean(cs):.0f} m/s)")
        for d in R["dirs"]:
            fr = R["freq"][d, g]
            if not fr["As"]:
                continue
            c, mk = COR_DIR.get(d, OUTRAS[0]), MARCA_DIR.get(d, "v")
            ax.errorbar(fr["x"], fr["y"], fr["sy"], fmt=mk, color=c, ms=6.5, mec="white", mew=0.8,
                        capsize=2.5, elinewidth=1, label=f"{d}: média dos N ≥ {A.N_MIN_F} de cada a")
            for a, x, y in zip(fr["As"], fr["x"], fr["y"]):
                ax.annotate(f"{a:g}", (x, y), xytext=(6, -11), textcoords="offset points",
                            fontsize=7, color=TINTA2)
            for aj, ls, rot in ((fr["origem"], "-", "pela origem"), (fr["livre"], "-.", "reta livre")):
                if not aj:
                    continue
                yy, ss = aj["m"] * xx + aj["b"], A.faixa(aj, xx)
                if aj["origem"]:
                    ax.fill_between(xx, yy - ss, yy + ss, color=c, alpha=0.13, lw=0)
                ax.plot(xx, yy, color=c, lw=1.3, ls=ls,
                        label=f"{d} {rot}: m = {A.pm(aj['m'], aj['sm_f'], 0)} m/s"
                              + (f", χ²/ν = {aj['chi2nu']:.1f}" if aj["nu"] > 0 else ""))
            aj = fr["livre"] or fr["origem"]
            if aj:
                ar.plot(aj["x"], aj["res"], mk, color=c, ms=5.5, mec="white", mew=0.6)
        _eixo(ax, f"{nome} vs 1/(2d)", None, "f (Hz)" if j == 0 else None)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.25 * np.mean(cs) * x_max if cs else None)
        ax.legend(fontsize=7, loc="upper left", frameon=False, labelcolor=TINTA2)
        ar.axhline(0, color=TINTA2, lw=0.6)
        for v in (-1, 1):
            ar.axhline(v, color=TINTA2, lw=0.6, ls=":")
        ar.set_ylim(-4, 4)
        _eixo(ar, None, "x = 1/(2d)  (m⁻¹);  d = a em ΓX, a/√2 em ΓM",
              "resíduo / σ" if j == 0 else None)
    fig.suptitle("Os números junto aos pontos são a (mm). Resíduos: da reta livre, ou da reta pela "
                 "origem se não houver pontos para a livre.", fontsize=8, color=TINTA2)


# ------------------------------------------------------------ comparação ΓX ↔ ΓM
def fig_comp(fig, R):
    """Em cima: T − T0 das duas direções para o mesmo a (N maior de cada), em Hz, com os hiatos
    do bloco — vê-se se se sobrepõem. Em baixo: f/f_B de cada bloco (devia ser igual nas duas)."""
    fig.clear()
    comuns = [c["a"] for c in R["comparar"]]
    n = max(len(comuns), 1)
    gs = fig.add_gridspec(2, n, height_ratios=[3, 2])
    for k, a in enumerate(comuns):
        ax = fig.add_subplot(gs[0, k])
        for d in R["dirs"]:
            L = [r for r in R["linhas"] if r["dir"] == d and r["a_mm"] == a and r["valida"]]
            r = max(L, key=lambda r: r["N"])
            c = COR_DIR.get(d, OUTRAS[0])
            ax.plot(r["f"], r["Ts"] - r["T0"], color=c, lw=1.5, label=f"{d}, N = {r['N']}")
            ax.axvline(r["fb"], color=c, ls="--", lw=0.9)
            p = R["por_a"][d, a]
            if p["f_inf"]:
                fim = p["f_sup"]["m"] if p["f_sup"] else A.FMAX
                ax.axvspan(p["f_inf"]["m"], fim, color=c, alpha=0.13, lw=0)
                ax.annotate(f"{d}" + ("" if p["f_sup"] else " (aberto) →"), (p["f_inf"]["m"], 11),
                            xytext=(3, 0), textcoords="offset points", fontsize=7.5, color=TINTA2,
                            va="top")
        ax.axhline(-A.LIMIAR_DB, color=TINTA2, ls=":", lw=0.9)
        ax.axhline(0, color=TINTA2, lw=0.6)
        ax.set_xlim(A.FMIN, min(A.FMAX, 2.2 * max(r["fb"] for r in R["linhas"] if r["a_mm"] == a)))
        ax.set_ylim(-28, 12)
        _eixo(ax, f"a = {a:g} mm" + (":  hiatos (faixas), f_B (tracejado)" if k == 0 else ""), "f (Hz)",
              "T − T0 (dB)" if k == 0 else None)
        ax.legend(fontsize=7.5, loc="lower left", frameon=False, labelcolor=TINTA2)
    if not comuns:
        _vazio(fig.add_subplot(gs[0, 0]), "Não há nenhum a medido nas duas direções.")

    ax = fig.add_subplot(gs[1, :])
    As = sorted({a for d in R["dirs"] for a in R[d]["As"]})
    for d in R["dirs"]:
        c = COR_DIR.get(d, OUTRAS[0])
        desl = -0.12 if d == R["dirs"][0] else 0.12
        for g, mk, cheio, nome in (("f_inf", "v", True, "f_inf"), ("fc", "o", False, "f_centro")):
            pts = [(As.index(a) + desl + (0.04 if g == "fc" else -0.04), R["por_a"][d, a])
                   for a in R[d]["As"] if R["por_a"][d, a][g]]
            if not pts:
                continue
            ax.errorbar([x for x, _ in pts], [p[g]["m"] / p["fb"] for _, p in pts],
                        [p[g]["s"] / p["fb"] for _, p in pts], fmt=mk, color=c, ms=6.5,
                        mfc=c if cheio else "white", mec=c, mew=1.2, capsize=2.5, elinewidth=1,
                        label=f"{d}: {nome} / f_B")
    ax.axhline(1, color=TINTA2, ls="--", lw=0.9)
    ax.set_xticks(range(len(As)))
    ax.set_xticklabels([f"a = {a:g} mm" for a in As])
    ax.set_xlim(-0.5, len(As) - 0.5)
    ax.set_ylim(0.5, 1.5)
    _eixo(ax, "Frequências do hiato em unidades de f_B = c/(2d): se só a distância entre planos "
              "contasse, as duas direções davam o mesmo", None, "f / f_B")
    ax.legend(fontsize=7.5, ncol=4, loc="upper left", frameon=False, labelcolor=TINTA2)


# ------------------------------------------------------------ uma medição
def fig_uma(fig, r):
    """T(f) de uma medição: faixa = ±σ entre repetições, linha = T suavizado; janelas, nível
    base, limiar e hiato encontrado."""
    fig.clear()
    ax = fig.add_subplot(111)
    c = COR_DIR.get(r["dir"], OUTRAS[0])
    _janelas(ax, r["fb"], r["f_ctrl"])
    ax.fill_between(r["f"], r["T"] - r["sT"], r["T"] + r["sT"], color=c, alpha=0.25, lw=0)
    ax.plot(r["f"], r["Ts"], color=c, lw=1.6, label="T(f)  (faixa: ±σ entre repetições)")
    ax.axhline(r["T0"], color="#1baf7a", lw=1, label=f"T0 = {r['T0']:.1f} dB")
    ax.axhline(r["T0"] - A.LIMIAR_DB, color=TINTA2, ls=":", lw=1, label=f"limiar (T0 − {A.LIMIAR_DB} dB)")
    if r["hiato"]:
        ax.axvspan(*r["hiato"], color=c, alpha=0.12, lw=0,
                   label="hiato" + ("" if r["fecha"] else " (não fecha)"))
    ax.set_xlim(A.FMIN, A.FMAX)
    _eixo(ax, f"{r['dir']}  {r['id']}   (a = {r['a_mm']:g} mm, N = {r['N']}, referência {r['ref']})",
          "f (Hz)", "T (dB)")
    ax.legend(fontsize=8, loc="lower left", frameon=False, labelcolor=TINTA2)


def texto_uma(r):
    """Os números de uma medição com o orçamento de incerteza por extenso."""
    def linha(g, nome, un, casas):
        if not np.isfinite(r[g]):
            return f"{nome}: —"
        partes = [f"estat. {A._n(r['s_' + g + '_est'], casas)}", f"ondulação {A._n(r['s_' + g + '_ond'], casas)}",
                  f"ref. {A._n(r['s_' + g + '_ref'], casas)}"]
        if g in A.FREQS:
            partes.append(f"resolução {r['s_' + g + '_res']:.0f}")
        return f"{nome} = {r[g]:.{casas}f} ± {r['s_' + g]:.{casas}f} {un}   ({', '.join(partes)})"
    t = [linha("profundidade", "Profundidade D", "dB", 1), linha("f_inf", "f_inf", "Hz", 0)]
    if r["hiato"] is None:
        t.append("Não há hiato abaixo do limiar perto de f_B (normal com poucas filas).")
    elif r["fecha"]:
        t += [linha("f_sup", "f_sup", "Hz", 0), linha("fc", "f_centro", "Hz", 0)
              + "      " + linha("largura", "largura", "Hz", 0)]
    else:
        t.append(f"O hiato não fecha: T não volta a subir acima do limiar antes de {A.FECHO_MAX} × f_B "
                 f"(sem f_sup, f_centro nem largura).")
    t.append(f"f_B = {r['fb']:.0f} Hz.  Ondulação na banda de controlo: {r['ondulacao']:.1f} dB.  "
             f"Candidatas (ondulação): " + ", ".join(f"{i} ({o:.1f})" for i, o in r["cand"])
             + f".  Repetições: {r['n']}/{r['n_reps']}.  Sinal/fundo ≥ {r['snr_min']:.0f} dB.")
    if not r["valida"]:
        t.append("⚠ Sem referência do mesmo dia: esta medição não entra nas regressões.")
    return "\n".join(t)
