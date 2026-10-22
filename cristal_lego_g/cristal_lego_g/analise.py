"""
analise.py — as contas da experiência (sem interface).

Ideia (detalhes e fórmulas no documento «Cristal sónico — métodos e contas da análise»):
  1. Gravar ruído branco SEM cristal (referência) e COM cristal, 5 vezes cada.
  2. Espectro de cada gravação pelo método de Welch, só no troço onde há ruído
     (a coluna Bluetooth atrasa ~0.55 s: usa-se 0.8–10.3 s).
     Controlo de qualidade por repetição: repetições com saturação ou com cortes do
     Bluetooth (blocos de 0.1 s mais de 6 dB abaixo do nível típico) são postas de parte.
  3. Referência = média, em potência, das referências válidas do MESMO BLOCO (mesmo dia e
     mesmo a: ref_aXX no início e ref_aXX_fim no fim). Entre blocos a montagem pode mudar,
     por isso não se juntam referências de blocos diferentes. De cada passo só conta a
     ÚLTIMA medição: as repetidas que a app move para _antigos são ignoradas.
  4. Os espectros são suavizados EM POTÊNCIA (média em 1/6 de oitava) antes de dividir,
     para um "buraco" estreito da referência não virar um pico falso em T(f).
  5. T(f) = 10·log10(P_cristal / P_ref)  [dB].
  6. Nível normal T0 = mediana de T numa zona onde o cristal é transparente:
       ΓX: dos dois lados do hiato (0.4–0.8 e 1.25–1.8 × f_B);
       ΓM: SÓ por baixo (0.15–0.6 × f_B). Na diagonal, a partir de f_B = c/(√2·a) as filas
           (tijolos a √2·a uns dos outros) começam também a difratar o som para os lados,
           por isso T não volta a subir acima de f_B: o "lado de cima" não é nível normal.
     Hiato = zona onde T < T0 − 6 dB perto de f_B.
       ΓX: f_centro = meio do hiato (é a grandeza principal).
       ΓM: f_inf = bordo de baixo do hiato (grandeza principal; o bordo de cima pode nem
           existir dentro da banda — "aberto").
     Profundidade = T0 − média de T numa janela: ΓX f_B ± 10 %; ΓM 1.05–1.35 × f_B.
  7. Incerteza de cada número = √(σ_estat² + σ_ref² [+ σ_res² nas frequências, Δf/√12]):
       σ_estat = desvio-padrão das repetições / √n
       σ_ref   = desvio-padrão do mesmo número calculado com cada referência candidata
                 (as do mesmo bloco; com uma só referência no bloco, σ_ref não se pode
                 estimar — por isso cada bloco acaba com ref_aXX_fim).
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, welch

# ------------------------------------------------------------ parâmetros fixos
FS = 48000          # amostragem (Hz)
NPERSEG = 4096      # tamanho de cada segmento de Welch → Δf = 11.7 Hz
FMIN, FMAX = 500, 12000   # banda analisada (Hz)
T_INI, T_FIM = 0.8, 10.3  # troço usado (s): o ruído chega ~0.55 s depois (Bluetooth)
SUAVIZAR_OIT = 1 / 6      # largura da média em potência (fração de oitava)
LIMIAR_DB = 6       # hiato = T abaixo de T0 − 6 dB
JANELA_PROF = 0.10  # ΓX: profundidade = atenuação média em f_B ± 10 %
JAN_PROF_GM = (1.05, 1.35)   # ΓM: profundidade = atenuação média em 1.05–1.35 × f_B
BASE_GX = ((0.4, 0.8), (1.25, 1.8))
BASE_GM = ((0.15, 0.6),)
JUNTAR = 0.10       # zonas abaixo do limiar separadas por < 10 % de f_B contam como uma
GRANDEZAS = ("profundidade", "fc", "largura", "f_inf", "f_sup")
# controlo de qualidade de cada repetição
CORTE_DB = -6.0     # bloco de 0.1 s 6 dB abaixo do nível típico → corte do Bluetooth
CLIP = 0.98         # |amostra| acima disto conta como saturação

_SOS_BANDA = butter(6, [FMIN, FMAX], btype="bandpass", fs=FS, output="sos")


def velocidade_som(temp_C):
    return 331.3 * np.sqrt(1 + temp_C / 273.15)


def eh_gm(direcao):
    return direcao == "ΓM"


def f_bragg(a_mm, temp_C, direcao="ΓX"):
    """1.º hiato: ΓX → f = c/(2a);  ΓM (cristal rodado 45°) → f = c/(√2·a)."""
    f = velocidade_som(temp_C) / (2 * a_mm / 1000)
    return f * np.sqrt(2) if eh_gm(direcao) else f


def nome_bragg(direcao):
    return "c/(√2·a)" if eh_gm(direcao) else "c/2a"


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
    """Controlo de qualidade de UMA repetição.
    nivel_db  : nível do ruído na banda 0.5–12 kHz (dBFS) — o que interessa, sem o zumbido
    corte_db  : bloco de 0.1 s mais fraco do troço, relativo ao percentil 90 dos blocos (dB)
                (< −6 dB → corte do Bluetooth, ou o ruído começou/acabou fora do troço)
    pico, clip: pico absoluto e nº de amostras saturadas
    zumbido   : fração da potência total abaixo de 100 Hz (zumbido da rede, 50 Hz)"""
    y = np.asarray(y, dtype="float64")
    yb = sosfiltfilt(_SOS_BANDA, y)
    t = troco(yb, dur)
    nb = len(t) // 4800
    blocos = np.sqrt(np.mean(t[:nb * 4800].reshape(nb, 4800) ** 2, axis=1)) + 1e-12
    f, P = welch(y - y.mean(), fs=FS, nperseg=8192)
    return dict(nivel_db=float(20 * np.log10(np.sqrt(np.mean(t ** 2)) + 1e-12)),
                corte_db=float(20 * np.log10(blocos.min() / np.percentile(blocos, 90))),
                pico=float(np.abs(y).max()), clip=int(np.sum(np.abs(y) > CLIP)),
                zumbido=float(P[f < 100].sum() / (P.sum() + 1e-30)))


def rep_ma(q):
    return q["clip"] > 0 or q["corte_db"] < CORTE_DB


def avisos_qc(qcs):
    """Texto com os problemas de um conjunto de repetições (lista de dicts de qualidade)."""
    if not qcs:
        return []
    av = []
    clip = [i + 1 for i, q in enumerate(qcs) if q["clip"] > 0]
    corte = [i + 1 for i, q in enumerate(qcs) if q["corte_db"] < CORTE_DB]
    niveis = np.array([q["nivel_db"] for q in qcs])
    if clip:
        av.append(f"SATURAÇÃO nas repetições {clip} — baixa o volume e repete")
    elif max(q["pico"] for q in qcs) > 0.9:
        av.append(f"pico {max(q['pico'] for q in qcs):.2f}: quase a saturar — baixa um pouco o volume")
    mas = [i + 1 for i, q in enumerate(qcs) if rep_ma(q)]
    if len(qcs) - len(mas) < 2:
        av.append(f"repetições {mas} estragadas (cortes/saturação) — REPETE este passo")
    elif corte:
        av.append(f"cortes do Bluetooth nas repetições {corte} (postas de parte)")
    if niveis.size >= 2 and niveis.max() - niveis.min() > 1.5:
        av.append(f"nível varia {niveis.max() - niveis.min():.1f} dB entre repetições — algo mexeu?")
    if np.median(niveis) < -45:
        av.append(f"sinal fraco na banda ({np.median(niveis):.0f} dBFS) — sobe o volume")
    if np.median([q["zumbido"] for q in qcs]) > 0.5:
        av.append("zumbido de 50 Hz forte (desliga o carregador do PC / afasta o cabo do micro)")
    return av


_cache = {}


def ler_espectros(pasta, com_fundo=False):
    """Espectros das repetições BOAS de uma pasta. meta['_qc'] tem a qualidade de todas e
    meta['_usadas'] os índices das usadas (se sobrarem < 2 boas, usam-se todas)."""
    pasta = Path(pasta)
    fichs = sorted(pasta.glob("*rep*.wav"))
    chave = (str(pasta), tuple(f.stat().st_mtime for f in fichs))
    if chave not in _cache:
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        Ps, Bs, qcs = [], [], []
        for fich in fichs:
            y, _ = sf.read(fich, dtype="float64")
            if y.ndim > 1:
                y = y[:, 0]
            f, P = espectro(y, meta["dur"])
            Ps.append(P)
            Bs.append(espectro_fundo(y, meta["dur"])[1])
            qcs.append(qualidade(y, meta["dur"]))
        boas = [i for i, q in enumerate(qcs) if not rep_ma(q)]
        if len(boas) < 2:
            boas = list(range(len(qcs)))
        meta = {**meta, "_qc": qcs, "_usadas": boas}
        _cache[chave] = (meta, f, np.array(Ps)[boas], np.array(Bs)[boas])
    meta, f, Ps, Bs = _cache[chave]
    return (meta, f, Ps, Bs) if com_fundo else (meta, f, Ps)


def suavizar_potencia(P, f, frac=SUAVIZAR_OIT):
    """Média de P (linear, não dB) numa janela de `frac` oitava centrada em cada f.
    Funciona com P 1-D ou 2-D (uma linha por repetição)."""
    zeros = np.zeros(P.shape[:-1] + (1,))
    c = np.concatenate([zeros, np.cumsum(P, axis=-1)], axis=-1)
    lo = np.searchsorted(f, f * 2 ** (-frac / 2), side="left")
    hi = np.searchsorted(f, f * 2 ** (frac / 2), side="right")
    return (c[..., hi] - c[..., lo]) / (hi - lo)


# ------------------------------------------------------------ referências
def dia(meta):
    return meta["hora"][:10]


def referencias(pasta_dados):
    """Referências válidas: só a última medição de cada passo de referência (as repetidas,
    movidas para _antigos, não contam) e só se não estiverem estragadas."""
    pasta_dados = Path(pasta_dados)
    refs = []
    for p in sorted(pasta_dados.glob("ref_*")):
        if not (p / "meta.json").exists():
            continue
        meta, f, P = ler_espectros(p)
        if all(rep_ma(q) for q in meta["_qc"]):
            continue
        refs.append(dict(pasta=p, meta=meta, P=P.mean(axis=0)))
    return refs


def escolher_refs(pasta_ref, meta_cristal):
    """Referências candidatas para uma medição com cristal.
    Só se juntam referências do MESMO BLOCO (mesmo dia e mesmo a: ref_aXX e ref_aXX_fim):
    entre blocos a montagem pode ter mudado (placa, posição do micro…), e juntar referências
    de montagens diferentes estraga T(f).
    Devolve (refs_usadas, candidatas_para_σ, modo)."""
    todas = referencias(Path(pasta_ref).parent)
    mesmo_dia = [r for r in todas if dia(r["meta"]) == dia(meta_cristal)]
    bloco = [r for r in mesmo_dia if abs(r["meta"]["a_mm"] - meta_cristal["a_mm"]) < 0.01]
    if len(bloco) >= 2:
        return bloco, bloco, f"bloco a = {meta_cristal['a_mm']:g} ({len(bloco)} ref.)"
    if len(bloco) == 1:
        return bloco, bloco, "bloco (1 ref.; σ_ref indisponível — mede a ref. de fim)"
    if mesmo_dia:
        return mesmo_dia, mesmo_dia, "⚠ SEM referência deste a hoje — usa as de outros a"
    return todas, todas, "OUTRO DIA (sem referência desta sessão)"


# ------------------------------------------------------------ hiato
def janela_base(f, fb, direcao="ΓX"):
    m = np.zeros_like(f, dtype=bool)
    for lo, hi in (BASE_GM if eh_gm(direcao) else BASE_GX):
        m |= (f > lo * fb) & (f < hi * fb)
    return m


def janela_prof(f, fb, direcao="ΓX"):
    if eh_gm(direcao):
        lo, hi = JAN_PROF_GM
    else:
        lo, hi = 1 - JANELA_PROF, 1 + JANELA_PROF
    return (f > lo * fb) & (f < hi * fb)


def nivel_base(f, Ts, fb, direcao="ΓX"):
    """Nível 'normal' de T: mediana numa zona onde o cristal é transparente."""
    lados = janela_base(f, fb, direcao)
    return float(np.median(Ts[lados])) if lados.sum() > 10 else float(np.median(Ts))


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
    # junta zonas separadas por menos de 10 % de f_B (o hiato de um cristal pequeno
    # vem muitas vezes partido em 2–3 bocados por ondulações de ±3 dB)
    juntas = []
    for z in zonas:
        if juntas and f[z[0]] - f[juntas[-1][1]] < JUNTAR * fb:
            juntas[-1] = (juntas[-1][0], z[1])
        else:
            juntas.append(z)
    return juntas


def encontrar_hiato(f, Ts, fb, T0=None, direcao="ΓX"):
    """Zona contínua abaixo de T0 − limiar ligada ao 1.º hiato. Devolve (f_inf, f_sup) ou None.
    ΓX: a zona cujo centro está mais perto de f_B (até ±30 %).
    ΓM: a zona cujo bordo de baixo está mais perto de f_B (0.8–1.25 × f_B); o bordo de
        cima pode ser o fim da banda (hiato "aberto")."""
    if T0 is None:
        T0 = nivel_base(f, Ts, fb, direcao)
    juntas = _zonas(f, Ts, T0, fb)
    if eh_gm(direcao):
        boas = [z for z in juntas if 0.8 * fb < f[z[0]] < 1.25 * fb]
        if not boas:
            return None
        z = min(boas, key=lambda z: abs(f[z[0]] - fb))
    else:
        boas = [z for z in juntas if abs((f[z[0]] + f[z[1]]) / 2 - fb) < 0.3 * fb]
        if not boas:
            return None
        z = min(boas, key=lambda z: abs((f[z[0]] + f[z[1]]) / 2 - fb))
    return f[z[0]], f[z[1]]


def numeros(f, Ts, fb, direcao="ΓX"):
    """Grandezas de UMA curva T(f) (em dB, já suavizada em potência)."""
    T0 = nivel_base(f, Ts, fb, direcao)
    jan = janela_prof(f, fb, direcao)
    out = dict(profundidade=T0 - Ts[jan].mean(), fc=np.nan, largura=np.nan,
               f_inf=np.nan, f_sup=np.nan)
    h = encontrar_hiato(f, Ts, fb, T0, direcao)
    if h:
        out.update(f_inf=h[0], f_sup=h[1], fc=(h[0] + h[1]) / 2, largura=h[1] - h[0])
    return out


def media_erro(valores):
    v = np.array([x for x in valores if np.isfinite(x)])
    if v.size == 0:
        return np.nan, np.nan
    if v.size == 1:
        return v[0], np.nan
    return v.mean(), v.std(ddof=1) / np.sqrt(v.size)


def desvio(valores):
    v = np.array([x for x in valores if np.isfinite(x)])
    return v.std(ddof=1) if v.size >= 2 else np.nan


def analisar(pasta_cristal, pasta_ref):
    """T(f) e os números do hiato com incerteza estatística + da referência.
    `pasta_ref` só serve para saber onde estão os dados: a referência usada é a média
    das referências da mesma sessão (ver escolher_refs)."""
    meta, f, Pc, Bc = ler_espectros(pasta_cristal, com_fundo=True)
    refs, cand, modo = escolher_refs(pasta_ref, meta)
    if not refs:
        raise ValueError("Não há nenhuma referência válida.")
    Pr_m = np.mean([r["P"] for r in refs], axis=0)       # média em potência
    direcao = meta.get("dir", "ΓX")
    fb = f_bragg(meta["a_mm"], meta["temp_C"], direcao)

    # curvas
    Ti = 10 * np.log10(Pc / Pr_m)                        # bruto (faixa ±σ do gráfico)
    T = Ti.mean(axis=0)
    sT = Ti.std(axis=0, ddof=1) / np.sqrt(len(Ti)) if len(Ti) > 1 else np.zeros_like(T)
    Pr_s = suavizar_potencia(Pr_m, f)
    Pc_s = suavizar_potencia(Pc.mean(axis=0), f)
    Ts = 10 * np.log10(Pc_s / Pr_s)
    T0 = nivel_base(f, Ts, fb, direcao)

    # estatística: uma curva por repetição, mesma referência
    Tsi = 10 * np.log10(suavizar_potencia(Pc, f) / Pr_s)
    por_rep = [numeros(f, t, fb, direcao) for t in Tsi]
    # referência: a média das repetições dividida por cada referência candidata
    por_ref = [numeros(f, 10 * np.log10(Pc_s / suavizar_potencia(r["P"], f)), fb, direcao)
               for r in cand]
    central = numeros(f, Ts, fb, direcao)

    # relação sinal/ruído de fundo na janela do hiato (o cristal tem de ficar acima do fundo)
    jan = janela_prof(f, fb, direcao)
    snr = 10 * np.log10(suavizar_potencia(Pc.mean(0), f)[jan] / suavizar_potencia(Bc.mean(0), f)[jan])

    hiato = encontrar_hiato(f, Ts, fb, T0, direcao)
    res = dict(meta=meta, f=f, T=T, sT=sT, Ts=Ts, fb=fb, n=len(Ti), T0=T0, dir=direcao,
               hiato=hiato, aberto=bool(hiato and hiato[1] >= f[-1] - 1),
               base=janela_base(f, fb, direcao), jan_prof=jan,
               modo_ref=modo, n_refs=len(refs), n_cand=len(cand), snr_min=float(snr.min()),
               avisos=avisos_qc(meta["_qc"]), n_reps=len(meta["_qc"]))
    for k in GRANDEZAS:
        _, s_est = media_erro([r[k] for r in por_rep])
        s_ref = desvio([r[k] for r in por_ref])
        res[k] = central[k]
        res["s_" + k + "_est"] = s_est
        res["s_" + k + "_ref"] = s_ref
        # frequências: junta a resolução do espectro (Δf/√12; Δf = 11.7 Hz), senão com
        # repetições iguais ao bin a incerteza saía 0 Hz
        s_res = (f[1] - f[0]) / np.sqrt(12) if k != "profundidade" and np.isfinite(central[k]) else 0.0
        res["s_" + k + "_res"] = s_res
        res["s_" + k] = float(np.sqrt(np.nansum([s_est ** 2, s_ref ** 2, s_res ** 2]))) \
            if np.isfinite(central[k]) else np.nan
    # grandeza de frequência principal: ΓX → centro do hiato; ΓM → bordo de baixo
    res["f_hiato"], res["s_f_hiato"] = (res["f_inf"], res["s_f_inf"]) if eh_gm(direcao) \
        else (res["fc"], res["s_fc"])
    return res


# ------------------------------------------------------------ regressões
def reta(x, y, sy):
    """Regressão linear ponderada y = m x + b. Devolve m, σm, b, σb, χ²/ν."""
    x, y, sy = map(np.asarray, (x, y, sy))
    sy = np.where(np.isfinite(sy) & (sy > 0), sy, np.nanmax(sy) if np.isfinite(sy).any() else 1)
    w = 1 / sy ** 2
    S, Sx, Sy, Sxx, Sxy = w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx ** 2
    m, b = (S * Sxy - Sx * Sy) / D, (Sxx * Sy - Sx * Sxy) / D
    chi2 = (w * (y - m * x - b) ** 2).sum()
    nu = len(x) - 2
    return dict(m=m, sm=np.sqrt(S / D), b=b, sb=np.sqrt(Sxx / D),
                chi2nu=chi2 / nu if nu > 0 else np.nan)


def reta_origem(x, y, sy):
    """Regressão ponderada y = m x (reta pela origem). Devolve m, σm, χ²/ν."""
    x, y, sy = map(np.asarray, (x, y, sy))
    sy = np.where(np.isfinite(sy) & (sy > 0), sy, np.nanmax(sy) if np.isfinite(sy).any() else 1)
    w = 1 / sy ** 2
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    m = Sxy / Sxx
    nu = len(x) - 1
    chi2 = (w * (y - m * x) ** 2).sum()
    return dict(m=m, sm=1 / np.sqrt(Sxx), chi2nu=chi2 / nu if nu > 0 else np.nan)


# ------------------------------------------------------------ simulação (treino)
def simular(n, passo, temp_C=20, lado_mm=15.8, rng=np.random.default_rng()):
    """Gravação falsa: coluna + (cristal com hiato de Bragg) + ruído.
    Em ΓM, acima de f_B junta-se a perda por difração lateral (T não volta a subir)."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    ganho = 1 / (1 + (200 / (f + 1)) ** 4) / (1 + (f / 14000) ** 4)   # coluna
    if passo["tipo"] in ("cristal", "extra"):
        a, N = passo["a_mm"], passo["N"]
        gm = eh_gm(passo.get("dir", "ΓX"))
        ff = (lado_mm / a) ** 2
        fb0 = f_bragg(a, temp_C, passo.get("dir", "ΓX"))
        # pequenas diferenças de montagem de repetição para repetição
        if gm:   # em ΓM o hiato de Bragg é estreito (ff pequeno) e fica junto a f_B
            fb = fb0 * (1 + ff / 3) * (1 + 0.003 * rng.standard_normal())   # bordo de baixo ≈ f_B
            meia = fb * ff / 3
        else:
            fb = fb0 * (1 - 0.1 * ff) * (1 + 0.003 * rng.standard_normal())
            meia = fb * ff
        N = N * (1 + 0.04 * rng.standard_normal())
        dentro = 1 / (1 + np.exp(-(f - fb + meia) / 60)) / (1 + np.exp((f - fb - meia) / 60))
        TdB = -N * 50 * ff * dentro
        if gm:
            TdB -= 10 * (1 - np.exp(-N / 4)) / (1 + np.exp(-(f - fb0) / 80))
        TdB = 10 * np.log10(10 ** (TdB / 10) + 10 ** (-2.4))       # satura a −24 dB
        TdB += 0.8 * np.cos(2 * np.pi * f * 2 * N * a / 1000 / velocidade_som(temp_C))
        ganho = ganho * 10 ** (TdB / 20)
    y = np.fft.irfft(X * ganho, n) * 0.05 * (1 + 0.03 * rng.standard_normal())
    y += 2e-4 * rng.standard_normal(n)
    return y.astype(np.float32)
