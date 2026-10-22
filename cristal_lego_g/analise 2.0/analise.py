"""
analise.py — análise CONJUNTA das duas experiências do cristal sónico de LEGO:
  ΓX (som perpendicular às filas, pasta cristal_simples) e ΓM (som na diagonal, cristal_lego_g).

As duas direções passam exatamente pelas mesmas contas. A única coisa que muda é a distância
entre planos de tijolos, d, que entra na frequência de Bragg:
      f_B = c / (2d),      d = a (ΓX),   d = a/√2 (ΓM).
Todas as janelas da análise são definidas em unidades de f_B, por isso são as mesmas nas duas.

  1. Espectro de cada gravação (Welch, Hann, 4096, 50 %) no troço com ruído (0.8–10.3 s).
     Repetições com saturação ou cortes do Bluetooth são postas de parte.
  2. T(f) = 10·log10(P_cristal / P_ref), com os espectros suavizados em potência (1/6 oitava).
  3. Referência: as candidatas são TODAS as referências da mesma sessão (mesma pasta de dados
     e mesmo dia) — a referência é a montagem sem tijolos, não depende de a. Para cada BLOCO
     (mesmo a, mesmo dia) usa-se a que deixa T(f) mais plano na BANDA DE CONTROLO
     (0.35–0.70 × c/(2a), por baixo do hiato, onde o cristal é transparente): a de menor
     mediana da ondulação nas medições do bloco. As outras servem para a incerteza σ_ref.
     Sem referência do mesmo dia a medição fica na tabela mas não entra nas regressões.
  4. T0 = mediana de T na banda de controlo (absorve mudanças de volume entre medições).
     A banda de controlo é 0.35–0.70 × c/(2a) nas DUAS direções (ver f_controlo): tem de ficar
     abaixo do 1.º hiato da rede em qualquer direção, que é o de ΓX.
     Hiato = zona com T < T0 − 6 dB que toca em 0.75–1.25 × f_B.
       f_inf = bordo de baixo (existe sempre que há hiato; grandeza principal nas duas direções)
       f_sup = bordo de cima, só se T voltar a subir antes de 1.5 × f_B ("fecha");
               f_centro e largura só existem quando o hiato fecha. Um bloco (um a) só tem
               f_sup, f_centro e largura se o hiato fechar em mais de metade das medições.
     Profundidade D = T0 − média de T em f_B ± 10 %.
  5. Incerteza de cada número:  σ² = σ_estat² + σ_ond² + σ_ref² (+ σ_res² nas frequências)
       σ_estat = desvio-padrão entre repetições / √n
       σ_ond   = ondulação de T propagada para o número. Ondulação = desvio robusto de T na
                 banda de controlo, onde T devia ser plano: é o "ruído" de T(f) que as
                 repetições não veem (reflexões, referência que não acerta).
                   profundidade: ondulação × √(π/2·1/n_base + 1/n_jan), n = largura da janela
                                 em unidades da largura de suavização (pontos independentes)
                   bordos:       ondulação / |dT/df| no cruzamento do limiar
       σ_ref   = dispersão do número calculado com cada referência candidata, em torno do
                 valor central, com pesos 1/ondulação² (uma referência que não serve pesa pouco)
       σ_res   = Δf/√12 (resolução do espectro)
  6. Regressões: mínimos quadrados ponderados (1/σ²), com incerteza em x por variância
     efetiva. De cada ajuste saem os parâmetros, a matriz de covariância, χ², ν, χ²/ν e o
     valor-p. Se χ²/ν > 1 as incertezas dos parâmetros são multiplicadas por √(χ²/ν).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import optimize, stats
from scipy.signal import butter, sosfiltfilt, welch

# ------------------------------------------------------------ parâmetros fixos
FS = 48000          # amostragem (Hz)
NPERSEG = 4096      # tamanho de cada segmento de Welch → Δf = 11.7 Hz
FMIN, FMAX = 500, 12000   # banda analisada (Hz)
T_INI, T_FIM = 0.8, 10.3  # troço usado (s): o ruído chega ~0.55 s depois (Bluetooth)
SUAVIZAR_OIT = 1 / 6      # largura da média em potência (fração de oitava)
LIMIAR_DB = 6       # hiato = T abaixo de T0 − 6 dB
# janelas em unidades de f_B — as mesmas para ΓX e ΓM
BASE = (0.35, 0.70)       # banda de controlo: nível base T0 e ondulação — em unidades de
#                           c/(2a), a 1.ª frequência de Bragg da rede (ver f_controlo)
JAN_PROF = (0.90, 1.10)   # profundidade = atenuação média em f_B ± 10 %
PROCURA = (0.75, 1.25)    # o hiato tem de tocar nesta zona
FECHO_MAX = 1.5     # bordo de cima acima de 1.5 × f_B → o hiato não fecha
JUNTAR = 0.10       # zonas abaixo do limiar separadas por < 10 % de f_B contam como uma
N_MIN_F = 4         # frequências do hiato: só N ≥ 4 (com menos filas o hiato mal existe)
SIGMA_A_MM = 0.1    # incerteza de a (posição dos tijolos nos pinos)
SIGMA_TEMP = 1.0    # incerteza da temperatura da sala (°C)
IGUAL_REF_DB = 0.5  # "cristal" a menos disto de uma referência → gravado sem tijolos
GRANDEZAS = ("profundidade", "f_inf", "f_sup", "fc", "largura")
FREQS = ("f_inf", "f_sup", "fc", "largura")
# controlo de qualidade de cada repetição
CORTE_DB = -6.0     # bloco de 0.1 s 6 dB abaixo do nível típico → corte do Bluetooth
CLIP = 0.98         # |amostra| acima disto conta como saturação
VERSAO_CACHE = 1

_SOS_BANDA = butter(6, [FMIN, FMAX], btype="bandpass", fs=FS, output="sos")


def velocidade_som(temp_C):
    return 331.3 * np.sqrt(1 + temp_C / 273.15)


def sigma_velocidade_som(temp_C, s_temp=SIGMA_TEMP):
    return velocidade_som(temp_C) / (2 * (273.15 + temp_C)) * s_temp


def dist_planos(a_mm, direcao):
    """Distância entre planos de tijolos perpendiculares ao som (m)."""
    return a_mm / 1000 / (np.sqrt(2) if direcao == "ΓM" else 1)


def f_bragg(a_mm, temp_C, direcao):
    return velocidade_som(temp_C) / (2 * dist_planos(a_mm, direcao))


def f_controlo(a_mm, temp_C):
    """Frequência de referência da banda de controlo: c/(2a), nas DUAS direções.
    A banda de controlo tem de ficar onde o cristal é transparente para TODAS as direções.
    O primeiro hiato da rede, em qualquer direção, é o de ΓX, perto de c/(2a). Em ΓM, uma banda
    0.35–0.70 × c/(√2·a) iria até 0.99 × c/(2a) e apanharia o hiato de ΓX, que o som vindo
    de lado (fonte próxima, cristal estreito) também vê: T já está a descer aí e T0 sairia
    puxado para baixo. Em ΓX não muda nada (f_B = c/(2a))."""
    return f_bragg(a_mm, temp_C, "ΓX")


def nome_d(direcao):
    return "a/√2" if direcao == "ΓM" else "a"


# ------------------------------------------------------------ espectros
def troco(y, dur):
    """Troço da gravação com ruído da coluna (fim limitado pela duração real)."""
    fim = min(T_FIM, dur + 0.3)
    return y[int(T_INI * FS):int(fim * FS)]


def espectro(y, dur):
    """PSD de Welch (janela de Hann, 50 % de sobreposição) do troço com ruído."""
    f, P = welch(troco(y, dur), fs=FS, window="hann", nperseg=NPERSEG)
    banda = (f >= FMIN) & (f <= FMAX)
    return f[banda], P[banda]


def espectro_fundo(y, dur=10.0):
    """PSD do silêncio final (depois de a coluna parar): ruído de fundo da sala."""
    f, P = welch(y[int((dur + 0.9) * FS):], fs=FS, window="hann", nperseg=NPERSEG)
    banda = (f >= FMIN) & (f <= FMAX)
    return f[banda], P[banda]


def qualidade(y, dur=10.0):
    """Controlo de qualidade de UMA repetição (nível na banda, cortes, saturação)."""
    y = np.asarray(y, dtype="float64")
    t = troco(sosfiltfilt(_SOS_BANDA, y), dur)
    nb = len(t) // 4800
    blocos = np.sqrt(np.mean(t[:nb * 4800].reshape(nb, 4800) ** 2, axis=1)) + 1e-12
    return dict(nivel_db=float(20 * np.log10(np.sqrt(np.mean(t ** 2)) + 1e-12)),
                corte_db=float(20 * np.log10(blocos.min() / np.percentile(blocos, 90))),
                pico=float(np.abs(y).max()), clip=int(np.sum(np.abs(y) > CLIP)))


def rep_ma(q):
    return q["clip"] > 0 or q["corte_db"] < CORTE_DB


def suavizar_potencia(P, f, frac=SUAVIZAR_OIT):
    """Média de P (linear, não dB) numa janela de `frac` oitava centrada em cada f.
    Funciona com P 1-D ou 2-D (uma linha por repetição)."""
    zeros = np.zeros(P.shape[:-1] + (1,))
    c = np.concatenate([zeros, np.cumsum(P, axis=-1)], axis=-1)
    lo = np.searchsorted(f, f * 2 ** (-frac / 2), side="left")
    hi = np.searchsorted(f, f * 2 ** (frac / 2), side="right")
    return (c[..., hi] - c[..., lo]) / (hi - lo)


def ler_pasta(pasta, cache=None):
    """Espectros de todas as repetições de uma pasta de medição + qualidade de cada uma.
    Com `cache` (pasta), os espectros ficam guardados e só se voltam a calcular se os .wav
    mudarem."""
    pasta = Path(pasta)
    fichs = sorted(pasta.glob("*rep*.wav"))
    meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
    marca = "|".join(f"{x.name}:{x.stat().st_size}:{x.stat().st_mtime_ns}" for x in fichs)
    chave = hashlib.sha1(f"{pasta.resolve()}|{marca}|{VERSAO_CACHE}".encode()).hexdigest()[:20]
    guardado = Path(cache) / f"{chave}.npz" if cache else None
    if guardado is not None and guardado.exists():
        z = np.load(guardado, allow_pickle=False)
        f, Ps, Bs, qcs = z["f"], z["P"], z["B"], json.loads(str(z["qc"]))
    else:
        Ps, Bs, qcs = [], [], []
        for fich in fichs:
            y, _ = sf.read(fich, dtype="float64")
            if y.ndim > 1:
                y = y[:, 0]
            f, P = espectro(y, meta["dur"])
            Ps.append(P)
            Bs.append(espectro_fundo(y, meta["dur"])[1])
            qcs.append(qualidade(y, meta["dur"]))
        Ps, Bs = np.array(Ps), np.array(Bs)
        if guardado is not None:
            guardado.parent.mkdir(parents=True, exist_ok=True)
            np.savez(guardado, f=f, P=Ps, B=Bs, qc=json.dumps(qcs))
    boas = [i for i, q in enumerate(qcs) if not rep_ma(q)]
    if len(boas) < 2:
        boas = list(range(len(qcs)))
    return dict(pasta=pasta, id=meta["id"], meta=meta, f=f, P=Ps[boas], fundo=Bs[boas],
                qc=qcs, usadas=boas, todas_mas=all(rep_ma(q) for q in qcs))


def carregar(conjuntos, cache=None, progresso=None):
    """conjuntos: lista de dicts {nome, pasta, dir} (dir só é usada se o meta.json não a tiver).
    Lê todas as medições (cristal e referência) de cada pasta 'dados'; _antigos não conta."""
    itens = []
    for c in conjuntos:
        raiz = Path(c["pasta"])
        if not raiz.is_dir():
            raise FileNotFoundError(f"Não encontro a pasta de dados de «{c['nome']}»:\n{raiz}")
        itens += [(c, p) for p in sorted(raiz.iterdir())
                  if p.is_dir() and not p.name.startswith("_") and (p / "meta.json").exists()]
    meds = []
    for k, (c, p) in enumerate(itens):
        if progresso:
            progresso(k, len(itens), f"{c['nome']}: {p.name}")
        m = ler_pasta(p, cache)
        if m["meta"]["tipo"] not in ("cristal", "ref") or not len(m["qc"]):
            continue
        m.update(conj=c["nome"], dir=m["meta"].get("dir") or c.get("dir", "ΓX"),
                 tipo=m["meta"]["tipo"], a_mm=float(m["meta"]["a_mm"]), N=m["meta"]["N"],
                 dia=m["meta"]["hora"][:10], temp_C=float(m["meta"]["temp_C"]))
        meds.append(m)
    if progresso:
        progresso(len(itens), len(itens), "")
    return meds


# ------------------------------------------------------------ hiato
def robusto(v):
    """Desvio-padrão robusto (1.4826 × desvio absoluto mediano)."""
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _zonas(f, Ts, T0, fb):
    abaixo = Ts < T0 - LIMIAR_DB
    zonas, ini = [], None
    for i, b in enumerate(abaixo):
        if b and ini is None:
            ini = i
        if not b and ini is not None:
            zonas.append((ini, i - 1))
            ini = None
    if ini is not None:
        zonas.append((ini, len(f) - 1))
    juntas = []
    for z in zonas:
        if juntas and f[z[0]] - f[juntas[-1][1]] < JUNTAR * fb:
            juntas[-1] = (juntas[-1][0], z[1])
        else:
            juntas.append(z)
    return juntas


def _cruzamento(f, Ts, i, j, nivel):
    """Frequência onde T cruza `nivel` entre os pontos i e j (interpolação linear)."""
    if not (0 <= i < len(f) and 0 <= j < len(f)) or Ts[i] == Ts[j]:
        return f[min(max(j, 0), len(f) - 1)]
    return f[i] + (nivel - Ts[i]) / (Ts[j] - Ts[i]) * (f[j] - f[i])


def _declive(f, Ts, f0):
    """|dT/df| (dB/Hz) à volta de f0: reta ajustada a T numa janela com a largura da
    suavização (pontos mais próximos do que isso não são independentes)."""
    m = (f > f0 * 2 ** (-SUAVIZAR_OIT / 2)) & (f < f0 * 2 ** (SUAVIZAR_OIT / 2))
    return abs(np.polyfit(f[m], Ts[m], 1)[0]) if m.sum() >= 3 else np.nan


def encontrar_hiato(f, Ts, fb, T0):
    """Zona contínua abaixo de T0 − limiar com mais sobreposição com 0.75–1.25 × f_B.
    Devolve (f_inf, f_sup, fecha) ou None; os bordos são interpolados no cruzamento."""
    lo, hi = PROCURA[0] * fb, PROCURA[1] * fb
    melhor, sobre = None, 0.0
    for z in _zonas(f, Ts, T0, fb):
        s = min(f[z[1]], hi) - max(f[z[0]], lo)
        if s > sobre:
            melhor, sobre = z, s
    if melhor is None:
        return None
    i, j = melhor
    nivel = T0 - LIMIAR_DB
    f_inf = _cruzamento(f, Ts, i - 1, i, nivel) if i > 0 else f[0]
    fim = j >= len(f) - 1
    f_sup = f[-1] if fim else _cruzamento(f, Ts, j, j + 1, nivel)
    return f_inf, f_sup, bool(not fim and f_sup <= FECHO_MAX * fb)


def numeros(f, Ts, fb, f_ctrl=None):
    """Grandezas de UMA curva T(f) (em dB, já suavizada em potência). f_ctrl = frequência de
    referência da banda de controlo (c/(2a)); por omissão, fb."""
    u = f / fb
    uc = f / (f_ctrl or fb)
    base = (uc > BASE[0]) & (uc < BASE[1])
    T0 = float(np.median(Ts[base]))
    jan = (u > JAN_PROF[0]) & (u < JAN_PROF[1])
    out = dict(T0=T0, ondulacao=robusto(Ts[base]), profundidade=T0 - Ts[jan].mean(),
               f_inf=np.nan, f_sup=np.nan, fc=np.nan, largura=np.nan, hiato=None, fecha=False,
               decl_inf=np.nan, decl_sup=np.nan)
    h = encontrar_hiato(f, Ts, fb, T0)
    if h:
        out.update(f_inf=h[0], hiato=(h[0], h[1]), fecha=h[2], decl_inf=_declive(f, Ts, h[0]))
        if h[2]:
            out.update(f_sup=h[1], fc=(h[0] + h[1]) / 2, largura=h[1] - h[0],
                       decl_sup=_declive(f, Ts, h[1]))
    return out


def numeros_por_ref(med, refs):
    """Os números de uma medição calculados com cada referência candidata."""
    f = med["f"]
    fb = f_bragg(med["a_mm"], med["temp_C"], med["dir"])
    fc = f_controlo(med["a_mm"], med["temp_C"])
    Pc_s = suavizar_potencia(med["P"].mean(axis=0), f)
    return [numeros(f, 10 * np.log10(Pc_s / suavizar_potencia(r["P"].mean(axis=0), f)), fb, fc)
            for r in refs]


# nº de pontos independentes de cada janela (largura em unidades da largura de suavização)
_N_BASE = np.log2(BASE[1] / BASE[0]) / SUAVIZAR_OIT
_N_JAN = np.log2(JAN_PROF[1] / JAN_PROF[0]) / SUAVIZAR_OIT
K_OND_PROF = float(np.sqrt(np.pi / 2 / _N_BASE + 1 / _N_JAN))   # σ_ond(D) = K × ondulação


def _erro_media(valores):
    v = np.array([x for x in valores if np.isfinite(x)])
    return v.std(ddof=1) / np.sqrt(v.size) if v.size >= 2 else np.nan


def analisar(med, refs, k, por_ref=None):
    """T(f) e os números do hiato de uma medição com cristal. `refs` = referências candidatas
    (as da mesma sessão), `k` = índice da que se usa. Devolve um dict com as curvas, os números
    e o orçamento de incerteza."""
    f, Pc = med["f"], med["P"]
    fb = f_bragg(med["a_mm"], med["temp_C"], med["dir"])
    por_ref = por_ref or numeros_por_ref(med, refs)
    central, ref = por_ref[k], refs[k]
    Pr = ref["P"].mean(axis=0)
    Pr_s = suavizar_potencia(Pr, f)
    Ts = 10 * np.log10(suavizar_potencia(Pc.mean(axis=0), f) / Pr_s)
    Ti = 10 * np.log10(Pc / Pr)                          # bruto (faixa ±σ do gráfico)
    sT = Ti.std(axis=0, ddof=1) / np.sqrt(len(Ti)) if len(Ti) > 1 else np.zeros(len(f))
    f_ctrl = f_controlo(med["a_mm"], med["temp_C"])
    por_rep = [numeros(f, t, fb, f_ctrl) for t in 10 * np.log10(suavizar_potencia(Pc, f) / Pr_s)]
    pesos = np.array([1 / max(n["ondulacao"], 1e-3) ** 2 for n in por_ref])

    u = f / fb
    jan = (u > JAN_PROF[0]) & (u < JAN_PROF[1])
    snr = 10 * np.log10(suavizar_potencia(Pc.mean(axis=0), f)[jan]
                        / suavizar_potencia(med["fundo"].mean(axis=0), f)[jan])
    ond = central["ondulacao"]
    res = dict(id=med["id"], conj=med["conj"], dir=med["dir"], a_mm=med["a_mm"], N=int(med["N"]),
               dia=med["dia"], hora=med["meta"]["hora"], temp_C=med["temp_C"], fb=fb, f_ctrl=f_ctrl,
               d_m=dist_planos(med["a_mm"], med["dir"]), f=f, T=Ti.mean(axis=0), sT=sT, Ts=Ts,
               T0=central["T0"], ondulacao=ond, hiato=central["hiato"],
               fecha=central["fecha"], ref=ref["id"], n_cand=len(refs),
               cand=[(r["id"], n["ondulacao"]) for r, n in zip(refs, por_ref)],
               ref_propria=abs(ref["a_mm"] - med["a_mm"]) < 0.01,
               ref_outro_dia=ref["dia"] != med["dia"], n=len(Pc), n_reps=len(med["qc"]),
               snr_min=float(snr.min()), nivel_db=float(np.median([q["nivel_db"] for q in med["qc"]])))
    res["valida"] = not res["ref_outro_dia"]
    # ondulação de T propagada para cada número
    s_inf, s_sup = ond / central["decl_inf"], ond / central["decl_sup"]
    s_ond = dict(profundidade=K_OND_PROF * ond, f_inf=s_inf, f_sup=s_sup,
                 fc=np.hypot(s_inf, s_sup) / 2, largura=np.hypot(s_inf, s_sup))
    for g in GRANDEZAS:
        x = central[g]
        s_est = _erro_media([r[g] for r in por_rep])
        v = np.array([n[g] for n in por_ref])
        ok = np.isfinite(v)
        s_ref = float(np.sqrt(np.sum(pesos[ok] * (v[ok] - x) ** 2) / pesos[ok].sum())) \
            if np.isfinite(x) and ok.sum() >= 2 else np.nan
        s_res = (f[1] - f[0]) / np.sqrt(12) if g in FREQS and np.isfinite(x) else 0.0
        res[g] = x
        res["s_" + g + "_est"], res["s_" + g + "_ref"], res["s_" + g + "_res"] = s_est, s_ref, s_res
        res["s_" + g + "_ond"] = float(s_ond[g]) if np.isfinite(x) else np.nan
        res["s_" + g] = float(np.sqrt(np.nansum([s_est ** 2, s_ref ** 2, s_res ** 2, s_ond[g] ** 2]))) \
            if np.isfinite(x) else np.nan
    return res


MODOS_REF = {"sessao": "a melhor referência da sessão (mesmo dia), escolhida por bloco",
             "bloco": "só as referências do próprio bloco (mesmo a); sem elas, as da sessão"}


def analisar_tudo(meds, modo="sessao"):
    """Analisa todas as medições com cristal. Devolve (linhas, postas_de_parte, referências).
    Uma medição "com cristal" igual a uma referência da sessão (< 0.5 dB em toda a banda) foi
    gravada sem tijolos (clique em MEDIR no passo errado) e é posta de parte.
    A referência é escolhida por bloco (mesma pasta, direção, a e dia): a candidata com menor
    mediana da ondulação na banda de controlo nas medições do bloco."""
    refs = [m for m in meds if m["tipo"] == "ref" and not m["todas_mas"]]
    fora, blocos = [], {}
    for m in meds:
        if m["tipo"] != "cristal":
            continue
        cand = [r for r in refs if r["conj"] == m["conj"] and r["dia"] == m["dia"]] \
            or [r for r in refs if r["conj"] == m["conj"]]
        if not cand:
            fora.append((m, "não há nenhuma referência válida nesta pasta de dados"))
            continue
        Pm = suavizar_potencia(m["P"].mean(axis=0), m["f"])
        dif = [float(np.std(10 * np.log10(Pm / suavizar_potencia(r["P"].mean(axis=0), r["f"]))))
               for r in cand]
        if min(dif) < IGUAL_REF_DB:
            r = cand[int(np.argmin(dif))]
            fora.append((m, f"espectro igual ao da referência {r['id']} (desvio {min(dif):.2f} dB, "
                            f"gravada às {r['meta']['hora'][11:16]}; esta às "
                            f"{m['meta']['hora'][11:16]}): gravado SEM tijolos"))
            continue
        blocos.setdefault((m["conj"], m["dir"], m["a_mm"], m["dia"]), []).append(
            (m, cand, numeros_por_ref(m, cand)))
    linhas = []
    for (_, _, a, _), itens in blocos.items():
        cand = itens[0][1]
        ond = np.median([[n["ondulacao"] for n in pr] for _, _, pr in itens], axis=0)
        proprias = [i for i, r in enumerate(cand) if abs(r["a_mm"] - a) < 0.01]
        elegiveis = proprias if modo == "bloco" and proprias else range(len(cand))
        k = min(elegiveis, key=lambda i: ond[i])
        linhas += [analisar(m, cand, k, pr) for m, _, pr in itens]
    linhas.sort(key=lambda r: (r["dir"] != "ΓX", r["a_mm"], r["N"]))
    return linhas, fora, refs


# ------------------------------------------------------------ regressões
def ajuste(x, y, sy, sx=None, origem=False):
    """Mínimos quadrados ponderados de y = m·x + b (ou y = m·x se origem=True).
    Pesos 1/σ²; com sx, σ² = σy² + (m·σx)² (variância efetiva) e minimiza-se χ²(m, b) com esses pesos.
    Devolve m, b, incertezas formais (sm, sb), covariância, χ², ν, χ²/ν, valor-p,
    fator S = √(χ²/ν) se > 1 (senão 1) e incertezas finais sm_f = S·sm, sb_f = S·sb."""
    x, y, sy = (np.asarray(v, dtype=float) for v in (x, y, sy))
    sx = np.zeros_like(x) if sx is None else np.asarray(sx, dtype=float)
    bom = np.isfinite(sy) & (sy > 0)
    sem_erro = int((~bom).sum())
    if sem_erro:       # ponto sem incerteza: fica com a maior das outras (não pode dominar)
        sy = np.where(bom, sy, np.max(sy[bom]) if bom.any() else 1.0)
    npar = 1 if origem else 2

    def linear(s2):
        """Solução fechada dos mínimos quadrados com variâncias s2 fixas."""
        w = 1 / s2
        S, Sx, Sy, Sxx, Sxy = w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum()
        if origem:
            return Sxy / Sxx, 0.0, 1 / np.sqrt(Sxx), 0.0, 0.0
        D = S * Sxx - Sx ** 2
        return (S * Sxy - Sx * Sy) / D, (Sxx * Sy - Sx * Sxy) / D, np.sqrt(S / D), np.sqrt(Sxx / D), -Sx / D

    def chi2_de(m):
        """χ² mínimo em b para um declive m, com a variância efetiva desse m."""
        w = 1 / (sy ** 2 + (m * sx) ** 2)
        b = 0.0 if origem else (w * (y - m * x)).sum() / w.sum()
        return (w * (y - m * x - b) ** 2).sum()

    m, b, sm, sb, cov = linear(sy ** 2)
    if sx.any():       # os pesos dependem de m: procura-se o mínimo de χ²(m) à volta da 1.ª solução
        m = float(optimize.minimize_scalar(chi2_de, bracket=(m - sm, m + sm), tol=1e-12).x)
        s2 = sy ** 2 + (m * sx) ** 2
        _, b, sm, sb, cov = linear(s2)
        if not origem:
            b = float(((y - m * x) / s2).sum() / (1 / s2).sum())
    s = np.sqrt(sy ** 2 + (m * sx) ** 2)
    res = (y - m * x - b) / s
    chi2, nu = float((res ** 2).sum()), len(x) - npar
    chi2nu = chi2 / nu if nu > 0 else np.nan
    S_f = float(np.sqrt(chi2nu)) if nu > 0 and chi2nu > 1 else 1.0
    return dict(m=m, b=b, sm=sm, sb=sb, cov=cov, chi2=chi2, nu=nu, chi2nu=chi2nu,
                p=float(stats.chi2.sf(chi2, nu)) if nu > 0 else np.nan, S=S_f,
                sm_f=S_f * sm, sb_f=S_f * sb, n=len(x), origem=origem, x=x, y=y, s=s, res=res,
                sem_erro=sem_erro)


def faixa(aj, xx):
    """Incerteza (1σ, já com o fator S) da reta ajustada nos pontos xx."""
    xx = np.asarray(xx, dtype=float)
    return aj["S"] * np.sqrt(np.maximum(aj["sb"] ** 2 + xx ** 2 * aj["sm"] ** 2
                                        + 2 * xx * aj["cov"], 0))


def media_bloco(v, s, s_sis):
    """Valor de um bloco (um a, vários N) a partir de valores com incerteza total s, da qual
    s_sis é a parte comum a todo o bloco (ondulação e referência: a referência e a montagem
    são as mesmas para todos os N, por isso NÃO diminui ao fazer a média).
      valor  = média ponderada (pesos 1/s²)
      σ_disp = erro da média tirado da dispersão observada entre N (ponderado)
      σ_sis  = média ponderada das partes sistemáticas
      σ      = √(σ_disp² + σ_sis²)          (com um só N: a incerteza desse ponto)"""
    v, s, s_sis = (np.asarray(x, dtype=float) for x in (v, s, s_sis))
    ok = np.isfinite(v)
    v, s, s_sis = v[ok], s[ok], np.nan_to_num(s_sis[ok])
    if v.size == 0:
        return None
    bom = np.isfinite(s) & (s > 0)
    s = np.where(bom, s, np.max(s[bom]) if bom.any() else 1.0)
    w = 1 / s ** 2
    m = float((w * v).sum() / w.sum())
    sis = float((w * s_sis).sum() / w.sum())
    if v.size == 1:
        return dict(m=m, s=float(s[0]), s_disp=np.nan, s_sis=sis, n=1)
    disp = float(np.sqrt((w * (v - m) ** 2).sum() / ((v.size - 1) * w.sum())))
    return dict(m=m, s=float(np.hypot(disp, sis)), s_disp=disp, s_sis=sis, n=int(v.size))


def z_de(v, s, v0, s0=0.0):
    """Nº de desvios-padrão entre v ± s e o valor esperado v0 ± s0."""
    return abs(v - v0) / np.sqrt(s ** 2 + s0 ** 2)


def resultados(linhas):
    """Todas as regressões e comparações, com as mesmas contas para cada direção."""
    dirs = sorted({r["dir"] for r in linhas}, key=lambda d: d != "ΓX")
    R = dict(dirs=dirs, linhas=linhas, prof={}, por_a={}, freq={}, comparar=[])
    for d in dirs:
        Ld = [r for r in linhas if r["dir"] == d and r["valida"]]
        if not Ld:
            R[d] = dict(temp=np.nan, c=np.nan, s_c=np.nan, As=[])
            for g in ("f_inf", "fc"):
                R["freq"][d, g] = dict(As=[], livre=None, origem=None)
            continue
        temp = float(np.mean([r["temp_C"] for r in Ld]))
        R[d] = dict(temp=temp, c=velocidade_som(temp), s_c=sigma_velocidade_som(temp),
                    As=sorted({r["a_mm"] for r in Ld}))
        for a in R[d]["As"]:
            La = [r for r in Ld if r["a_mm"] == a]
            # profundidade vs N:  D = α·N + b
            if len(La) >= 3:
                aj = ajuste([r["N"] for r in La], [r["profundidade"] for r in La],
                            [r["s_profundidade"] for r in La])
                aj["por_m"] = aj["m"] / (100 * La[0]["d_m"])       # dB/cm
                aj["s_por_m"] = aj["sm_f"] / (100 * La[0]["d_m"])
                R["prof"][d, a] = aj
            # frequências do hiato: média ponderada dos N ≥ N_MIN_F
            Lf = [r for r in La if r["N"] >= N_MIN_F]
            x = 1 / (2 * La[0]["d_m"])                              # 1/(2d)  (m⁻¹)
            pa = dict(x=x, sx=x * SIGMA_A_MM / a, fb=R[d]["c"] * x, n_N=len(Lf),
                      n_fecha=sum(r["fecha"] for r in Lf), n_hiato=sum(r["hiato"] is not None for r in Lf))
            # o hiato do bloco só "fecha" se fechar em mais de metade das medições com hiato
            pa["fecha"] = 2 * pa["n_fecha"] > pa["n_hiato"]
            for g in FREQS:
                pa[g] = media_bloco([r[g] for r in Lf], [r["s_" + g] for r in Lf],
                                    [np.hypot(r["s_" + g + "_ond"], np.nan_to_num(r["s_" + g + "_ref"]))
                                     for r in Lf]) if g == "f_inf" or pa["fecha"] else None
            R["por_a"][d, a] = pa
        # f vs 1/(2d):  f = m·x + b  → o declive é uma velocidade (c se f = f_B)
        for g in ("f_inf", "fc"):
            pts = [(a, R["por_a"][d, a]) for a in R[d]["As"] if R["por_a"][d, a][g]]
            fr = dict(As=[a for a, _ in pts], livre=None, origem=None)
            if pts:
                x, sx = [p["x"] for _, p in pts], [p["sx"] for _, p in pts]
                y, sy = [p[g]["m"] for _, p in pts], [p[g]["s"] for _, p in pts]
                fr.update(x=np.array(x), y=np.array(y), sy=np.array(sy), sx=np.array(sx))
                if len(pts) >= 3:
                    fr["livre"] = ajuste(x, y, sy, sx)
                if len(pts) >= 2:
                    fr["origem"] = ajuste(x, y, sy, sx, origem=True)
            R["freq"][d, g] = fr
    # comparação entre direções, para os a medidos nas duas
    if len(dirs) == 2:
        dx, dm = dirs
        for a in sorted(set(R[dx]["As"]) & set(R[dm]["As"])):
            px, pm = R["por_a"][dx, a], R["por_a"][dm, a]
            c = dict(a=a, razao=None, sobre=None)
            if px["f_inf"] and pm["f_inf"]:
                q = pm["f_inf"]["m"] / px["f_inf"]["m"]
                sq = q * np.hypot(pm["f_inf"]["s"] / pm["f_inf"]["m"], px["f_inf"]["s"] / px["f_inf"]["m"])
                c["razao"] = (q, sq, z_de(q, sq, np.sqrt(2)))
            if px["f_sup"] and pm["f_inf"]:
                # hiato completo se o de ΓM começar antes de o de ΓX acabar
                s = px["f_sup"]["m"] - pm["f_inf"]["m"]
                c["sobre"] = (s, float(np.hypot(px["f_sup"]["s"], pm["f_inf"]["s"])))
            R["comparar"].append(c)
    return R


# ------------------------------------------------------------ texto
def pm(v, s, casas=0, un=""):
    if v is None or not np.isfinite(v):
        return "—"
    s_txt = f"{s:.{casas}f}" if s is not None and np.isfinite(s) else "?"
    return f"{v:.{casas}f} ± {s_txt}{un}"


def txt_ajuste(aj):
    """χ², ν, valor-p e fator de escala de um ajuste."""
    if aj["nu"] <= 0:
        return "ν = 0 (tantos pontos como parâmetros: não dá para testar o ajuste)"
    t = f"χ² = {aj['chi2']:.1f}, ν = {aj['nu']}, χ²/ν = {aj['chi2nu']:.2f}, p = {aj['p']:.2g}"
    if aj["S"] > 1:
        t += f"  → incertezas × S = {aj['S']:.2f}"
    if aj["sem_erro"]:
        t += f"  ({aj['sem_erro']} ponto(s) sem incerteza própria)"
    return t


def relatorio(R, fora=(), modo="sessao"):
    L = R["linhas"]
    t = ["ANÁLISE CONJUNTA DO CRISTAL SÓNICO — ΓX (perpendicular) e ΓM (diagonal)",
         "As duas direções passam pelas mesmas contas; só muda d em f_B = c/(2d): d = a (ΓX), a/√2 (ΓM).",
         f"Janelas: nível base {BASE[0]}–{BASE[1]} × c/(2a) (nas duas direções); × f_B: profundidade "
         f"{JAN_PROF[0]}–{JAN_PROF[1]}, procura do hiato {PROCURA[0]}–{PROCURA[1]}, "
         f"limiar T0 − {LIMIAR_DB} dB, hiato fecha se f_sup < {FECHO_MAX} f_B.",
         f"Referência: {MODOS_REF[modo]}.", ""]
    for d in R["dirs"]:
        Ld = [r for r in L if r["dir"] == d and r["valida"]]
        if not Ld:
            t.append(f"{d}: nenhuma medição com referência do mesmo dia.")
            continue
        t.append(f"{d}: {len(Ld)} medições, a = {', '.join(f'{a:g}' for a in R[d]['As'])} mm, "
                 f"T = {R[d]['temp']:.1f} ± {SIGMA_TEMP:g} °C → c = {R[d]['c']:.1f} ± {R[d]['s_c']:.1f} m/s")
    sem_ref = [r for r in L if not r["valida"]]
    if fora or sem_ref:
        t += ["", "FORA DAS REGRESSÕES"]
        t += [f"  {m['conj']} / {m['id']}: {porque}" for m, porque in fora]
        t += [f"  {r['conj']} / {r['id']} ({r['dia']}): não há referência do mesmo dia — aparece na "
              f"tabela (com {r['ref']}) mas não entra em nenhum ajuste" for r in sem_ref]

    t += ["", "1. PROFUNDIDADE vs N     D = α·N + b   (mínimos quadrados ponderados)"]
    for (d, a), aj in R["prof"].items():
        t.append(f"  {d} a = {a:g} mm (n = {aj['n']}): α = {pm(aj['m'], aj['sm_f'], 2)} dB/fila "
                 f"= {pm(aj['por_m'], aj['s_por_m'], 2)} dB/cm,  b = {pm(aj['b'], aj['sb_f'], 1)} dB "
                 f"(z = {z_de(aj['b'], aj['sb_f'], 0):.1f} de 0)")
        t.append(f"      {txt_ajuste(aj)}")
    faltam = [(d, a) for d in R["dirs"] for a in R[d]["As"] if (d, a) not in R["prof"]]
    for d, a in faltam:
        n = sum(1 for r in L if r["dir"] == d and r["a_mm"] == a and r["valida"])
        t.append(f"  {d} a = {a:g} mm: só {n} valor(es) de N — não dá para ajustar uma reta.")

    t += ["", f"2. FREQUÊNCIAS DO HIATO por a   (média ponderada dos N ≥ {N_MIN_F};  σ² = σ_disp² + "
              f"σ_sis²:", "   σ_disp = dispersão entre N, σ_sis = ondulação e referência, comuns ao bloco "
              "— não diminuem com a média)"]
    for (d, a), p in R["por_a"].items():
        def um(g, nome):
            v = p[g]
            if not v:
                return f"{nome} —"
            return f"{nome} = {pm(v['m'], v['s'])} Hz (n = {v['n']}; disp. {v['s_disp']:.0f}, sis. " \
                   f"{v['s_sis']:.0f})" if v["n"] > 1 else f"{nome} = {pm(v['m'], v['s'])} Hz (n = 1)"
        t.append(f"  {d} a = {a:g} mm  [f_B = {p['fb']:.0f} Hz; hiato em {p['n_hiato']}/{p['n_N']} "
                 f"medições, fecha em {p['n_fecha']}"
                 + ("" if p["fecha"] or not p["n_hiato"] else " → hiato ABERTO por cima: sem f_sup") + "]")
        t.append(f"      {um('f_inf', 'f_inf')};  {um('f_sup', 'f_sup')}")
        t.append(f"      {um('fc', 'f_centro')};  {um('largura', 'largura')}")
        if p["f_inf"]:
            t.append(f"      f_inf / f_B = {pm(p['f_inf']['m'] / p['fb'], p['f_inf']['s'] / p['fb'], 3)}"
                     + (f";  f_centro / f_B = {pm(p['fc']['m'] / p['fb'], p['fc']['s'] / p['fb'], 3)}"
                        if p["fc"] else ""))

    t += ["", "3. FREQUÊNCIA vs 1/(2d)     f = m·x + b,  x = 1/(2d)   (incerteza de a incluída por "
              "variância efetiva)", "   O declive é uma velocidade: se f fosse exatamente f_B dava c."]
    for (d, g), fr in R["freq"].items():
        nome = "f_inf" if g == "f_inf" else "f_centro"
        c, sc = R[d]["c"], R[d]["s_c"]
        if not fr["As"]:
            t.append(f"  {d} {nome}: sem valores.")
            continue
        t.append(f"  {d} {nome}  ({len(fr['As'])} valores de a: {', '.join(f'{a:g}' for a in fr['As'])} mm)")
        aj = fr["livre"]
        if aj:
            t.append(f"      reta livre:  m = {pm(aj['m'], aj['sm_f'], 1)} m/s (z = "
                     f"{z_de(aj['m'], aj['sm_f'], c, sc):.1f} de c),  b = {pm(aj['b'], aj['sb_f'])} Hz "
                     f"(z = {z_de(aj['b'], aj['sb_f'], 0):.1f} de 0)")
            t.append(f"          {txt_ajuste(aj)}")
        else:
            t.append("      reta livre:  precisa de pelo menos 3 valores de a.")
        aj = fr["origem"]
        if aj:
            t.append(f"      pela origem: m = {pm(aj['m'], aj['sm_f'], 1)} m/s (z = "
                     f"{z_de(aj['m'], aj['sm_f'], c, sc):.1f} de c = {c:.1f} ± {sc:.1f})")
            t.append(f"          {txt_ajuste(aj)}")
        else:
            t.append("      pela origem: precisa de pelo menos 2 valores de a.")

    if R["comparar"]:
        dx, dm = R["dirs"]
        t += ["", f"4. COMPARAÇÃO {dx} ↔ {dm}  (mesmo a nas duas direções)"]
        for c in R["comparar"]:
            t.append(f"  a = {c['a']:g} mm:")
            if c["razao"]:
                q, sq, z = c["razao"]
                t.append(f"      f_inf({dm}) / f_inf({dx}) = {pm(q, sq, 3)}   (Bragg: √2 = 1.414; z = {z:.1f})")
            else:
                t.append("      razão das frequências: falta o hiato numa das direções.")
            if c["razao"] and not c["sobre"]:
                t.append(f"      sobreposição: o hiato de {dx} não fecha, não há f_sup para comparar.")
            if c["sobre"]:
                s, ss = c["sobre"]
                veredicto = "os hiatos sobrepõem-se" if s > 2 * ss else \
                    "os hiatos NÃO se sobrepõem" if s < -2 * ss else "inconclusivo (menos de 2σ de zero)"
                t.append(f"      f_sup({dx}) − f_inf({dm}) = {pm(s, ss)} Hz  → {veredicto}")

    t += ["", "5. TABELA  (incerteza total; entre parênteses: estatística, ondulação, referência)"]
    t.append(f"  {'dir':<4}{'passo':<9}{'ref. usada':<13}{'ond.':>5}{'T0':>7}{'profundidade (dB)':>29}"
             f"{'f_inf (Hz)':>27}{'f_sup (Hz)':>14}{'largura (Hz)':>15}")

    def par(r, g, casas):
        if not np.isfinite(r[g]):
            return "—"
        return (f"{r[g]:.{casas}f} ± {r['s_' + g]:.{casas}f} ({_n(r['s_' + g + '_est'], casas)}, "
                f"{_n(r['s_' + g + '_ond'], casas)}, {_n(r['s_' + g + '_ref'], casas)})")
    for r in L:
        notas = []
        if r["hiato"] and not r["fecha"]:
            notas.append("não fecha")
        if not r["ref_propria"]:
            notas.append("ref. de outro bloco")
        if r["ref_outro_dia"]:
            notas.append("REF. DE OUTRO DIA: fora das regressões")
        if r["n"] < r["n_reps"]:
            notas.append(f"{r['n']}/{r['n_reps']} rep.")
        if r["snr_min"] < 10:
            notas.append(f"sinal/fundo {r['snr_min']:.0f} dB")
        t.append(f"  {r['dir']:<4}{r['id']:<9}{r['ref']:<13}{r['ondulacao']:>5.1f}{r['T0']:>7.1f}"
                 f"{par(r, 'profundidade', 1):>29}{par(r, 'f_inf', 0):>27}"
                 f"{pm(r['f_sup'], r['s_f_sup']):>14}{pm(r['largura'], r['s_largura']):>15}"
                 + ("   ⚠ " + ", ".join(notas) if notas else ""))
    t += ["", "ond. = ondulação de T na banda de controlo (dB): quanto menor, melhor a referência.",
          "σ² = σ_estat² + σ_ond² + σ_ref² (+ σ_res² nas frequências). Detalhes em LEIA-ME.txt."]
    return "\n".join(t)


def _n(v, casas):
    return f"{v:.{casas}f}" if np.isfinite(v) else "?"


CAMPOS_CSV = ("dir", "conj", "id", "a_mm", "N", "hora", "temp_C", "fb", "ref", "n_cand", "ondulacao",
              "T0", "n", "n_reps", "snr_min", "fecha", "valida")


def tabela_csv(linhas):
    cab = list(CAMPOS_CSV)
    for g in GRANDEZAS:
        cab += [g, "s_" + g, "s_" + g + "_est", "s_" + g + "_ond", "s_" + g + "_ref", "s_" + g + "_res"]
    out = [",".join(cab)]
    for r in linhas:
        v = []
        for c in cab:
            x = r[c]
            v.append(f"{x:.3f}" if isinstance(x, float) else str(int(x)) if isinstance(x, (bool, np.bool_))
                     else f'"{x}"' if isinstance(x, str) else str(x))
        out.append(",".join(v))
    return "\n".join(out) + "\n"


def ajustes_csv(R):
    """Todos os ajustes numa tabela: parâmetros, incertezas formais e finais, χ², ν, p."""
    out = ["ajuste,dir,a_mm,grandeza,modelo,n,m,sm_formal,sm_final,b,sb_formal,sb_final,cov_mb,"
           "chi2,nu,chi2nu,p,S"]

    def linha(nome, d, a, g, aj):
        out.append(f"{nome},{d},{a},{g},{'y=m*x' if aj['origem'] else 'y=m*x+b'},{aj['n']},"
                   f"{aj['m']:.5g},{aj['sm']:.5g},{aj['sm_f']:.5g},{aj['b']:.5g},{aj['sb']:.5g},"
                   f"{aj['sb_f']:.5g},{aj['cov']:.5g},{aj['chi2']:.4g},{aj['nu']},{aj['chi2nu']:.4g},"
                   f"{aj['p']:.4g},{aj['S']:.4g}")
    for (d, a), aj in R["prof"].items():
        linha("D_vs_N", d, f"{a:g}", "profundidade", aj)
    for (d, g), fr in R["freq"].items():
        for aj in (fr["livre"], fr["origem"]):
            if aj:
                linha("f_vs_1/(2d)", d, "", g, aj)
    return "\n".join(out) + "\n"
