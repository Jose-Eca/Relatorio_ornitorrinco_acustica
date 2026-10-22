"""
analise.py — as contas da experiência (sem interface).

Ideia (detalhes e fórmulas no documento «Cristal sónico — métodos e contas da análise»):
  1. Gravar ruído branco SEM cristal (referência) e COM cristal, 5 vezes cada.
  2. Espectro de cada gravação pelo método de Welch, só no troço onde há ruído
     (a coluna Bluetooth atrasa ~0.6 s: usa-se 0.8–10.3 s).
  3. Referência = média, em potência, de TODAS as referências válidas da mesma sessão (dia).
     Sem cristal a referência não depende de a, por isso todas servem.
  4. Os espectros são suavizados EM POTÊNCIA (média em 1/6 de oitava) antes de dividir,
     para um "buraco" estreito da referência não virar um pico falso em T(f).
  5. T(f) = 10·log10(P_cristal / P_ref)  [dB].
  6. Nível normal T0 = mediana de T dos dois lados do hiato (0.4–0.8 e 1.25–1.8 × c/2a).
     Hiato = zona onde T < T0 − 6 dB, perto de c/2a.
  7. Incerteza de cada número = √(σ_estat² + σ_ref²):
       σ_estat = desvio-padrão das 5 repetições / √5
       σ_ref   = desvio-padrão do mesmo número calculado com cada referência candidata
                 (as da mesma sessão; se só houver uma, todas as do projeto — conservador).
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
T_INI, T_FIM = 0.8, 10.3  # troço usado (s): o ruído chega ~0.6 s depois (Bluetooth)
SUAVIZAR_OIT = 1 / 6      # largura da média em potência (fração de oitava)
LIMIAR_DB = 6       # hiato = T abaixo de T0 − 6 dB
JANELA_PROF = 0.10  # profundidade = atenuação média em c/2a ± 10 %
JUNTAR = 0.10       # zonas abaixo do limiar separadas por < 10 % de c/2a contam como uma
GRANDEZAS = ("profundidade", "fc", "largura", "f_inf", "f_sup")


def velocidade_som(temp_C):
    return 331.3 * np.sqrt(1 + temp_C / 273.15)


def f_bragg(a_mm, temp_C, direcao="ΓX"):
    """1.º hiato: ΓX → f = c/(2a);  ΓM (cristal rodado 45°) → f = c/(√2·a)."""
    f = velocidade_som(temp_C) / (2 * a_mm / 1000)
    return f * np.sqrt(2) if direcao == "ΓM" else f


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


def espectro_fundo(y):
    """PSD do silêncio final (depois de a coluna parar): ruído de fundo da sala."""
    f, P = welch(y[int(10.9 * FS):], fs=FS, window="hann", nperseg=NPERSEG)
    banda = (f >= FMIN) & (f <= FMAX)
    return f[banda], P[banda]


_cache = {}


def ler_espectros(pasta, com_fundo=False):
    pasta = Path(pasta)
    fichs = sorted(pasta.glob("*rep*.wav"))
    chave = (str(pasta), tuple(f.stat().st_mtime for f in fichs))
    if chave not in _cache:
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        Ps, Bs = [], []
        for fich in fichs:
            y, _ = sf.read(fich, dtype="float64")
            f, P = espectro(y, meta["dur"])
            Ps.append(P)
            Bs.append(espectro_fundo(y)[1])
        _cache[chave] = (meta, f, np.array(Ps), np.array(Bs))
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
    """Todas as referências válidas (sem saturação), incluindo as repetidas (_antigos)."""
    pasta_dados = Path(pasta_dados)
    pastas = list(pasta_dados.glob("ref_*")) + list((pasta_dados / "_antigos").glob("ref_*"))
    refs = []
    for p in sorted(pastas):
        if not (p / "meta.json").exists():
            continue
        meta, f, P = ler_espectros(p)
        if meta.get("pico", 0) > 0.99:
            continue
        refs.append(dict(pasta=p, meta=meta, P=P.mean(axis=0)))
    return refs


def escolher_refs(pasta_ref, meta_cristal):
    """Referências candidatas para uma medição com cristal.
    Devolve (refs_usadas, candidatas_para_σ, modo): modo = 'sessão' (≥2 do mesmo dia), 'sessão (1 ref.)' ou 'outro dia'."""
    todas = referencias(Path(pasta_ref).parent)
    mesmo_dia = [r for r in todas if dia(r["meta"]) == dia(meta_cristal)]
    if len(mesmo_dia) >= 2:
        return mesmo_dia, mesmo_dia, "sessão"
    if len(mesmo_dia) == 1:
        return mesmo_dia, todas, "sessão (1 ref.; σ_ref de todas as sessões)"
    return todas, todas, "OUTRO DIA (sem referência desta sessão)"


# ------------------------------------------------------------ hiato
def nivel_base(f, Ts, fb):
    """Nível 'normal' de T: mediana dos dois lados do hiato (evita o hiato e o 2.º hiato)."""
    lados = ((f > 0.4 * fb) & (f < 0.8 * fb)) | ((f > 1.25 * fb) & (f < 1.8 * fb))
    return float(np.median(Ts[lados])) if lados.sum() > 10 else float(np.median(Ts))


def encontrar_hiato(f, Ts, fb, T0=None):
    """Zona contínua abaixo de T0 − limiar mais perto de c/2a.
    Devolve (f_inf, f_sup) ou None."""
    if T0 is None:
        T0 = nivel_base(f, Ts, fb)
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
    # junta zonas separadas por menos de 10 % de c/2a (o hiato de um cristal pequeno
    # vem muitas vezes partido em 2–3 bocados por ondulações de ±3 dB)
    juntas = []
    for z in zonas:
        if juntas and f[z[0]] - f[juntas[-1][1]] < JUNTAR * fb:
            juntas[-1] = (juntas[-1][0], z[1])
        else:
            juntas.append(z)
    boas = [z for z in juntas if abs((f[z[0]] + f[z[1]]) / 2 - fb) < 0.3 * fb]
    if not boas:
        return None
    z = min(boas, key=lambda z: abs((f[z[0]] + f[z[1]]) / 2 - fb))
    return f[z[0]], f[z[1]]


def numeros(f, Ts, fb):
    """Grandezas de UMA curva T(f) (em dB, já suavizada em potência)."""
    T0 = nivel_base(f, Ts, fb)
    jan = (f > fb * (1 - JANELA_PROF)) & (f < fb * (1 + JANELA_PROF))
    out = dict(profundidade=T0 - Ts[jan].mean(), fc=np.nan, largura=np.nan,
               f_inf=np.nan, f_sup=np.nan)
    h = encontrar_hiato(f, Ts, fb, T0)
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
    Pr_m = np.mean([r["P"] for r in refs], axis=0)       # média em potência
    fb = f_bragg(meta["a_mm"], meta["temp_C"], meta.get("dir", "ΓX"))

    # curvas
    Ti = 10 * np.log10(Pc / Pr_m)                        # bruto (faixa ±σ do gráfico)
    T = Ti.mean(axis=0)
    sT = Ti.std(axis=0, ddof=1) / np.sqrt(len(Ti)) if len(Ti) > 1 else np.zeros_like(T)
    Pr_s = suavizar_potencia(Pr_m, f)
    Pc_s = suavizar_potencia(Pc.mean(axis=0), f)
    Ts = 10 * np.log10(Pc_s / Pr_s)
    T0 = nivel_base(f, Ts, fb)

    # estatística: uma curva por repetição, mesma referência
    Tsi = 10 * np.log10(suavizar_potencia(Pc, f) / Pr_s)
    por_rep = [numeros(f, t, fb) for t in Tsi]
    # referência: a média das repetições dividida por cada referência candidata
    por_ref = [numeros(f, 10 * np.log10(Pc_s / suavizar_potencia(r["P"], f)), fb) for r in cand]
    central = numeros(f, Ts, fb)

    # relação sinal/ruído de fundo na janela do hiato (o cristal tem de ficar acima do fundo)
    jan = (f > fb * (1 - JANELA_PROF)) & (f < fb * (1 + JANELA_PROF))
    snr = 10 * np.log10(suavizar_potencia(Pc.mean(0), f)[jan] / suavizar_potencia(Bc.mean(0), f)[jan])

    res = dict(meta=meta, f=f, T=T, sT=sT, Ts=Ts, fb=fb, n=len(Ti), T0=T0,
               hiato=encontrar_hiato(f, Ts, fb, T0), modo_ref=modo, n_refs=len(refs),
               n_cand=len(cand), snr_min=float(snr.min()))
    for k in GRANDEZAS:
        _, s_est = media_erro([r[k] for r in por_rep])
        s_ref = desvio([r[k] for r in por_ref])
        res[k] = central[k]
        res["s_" + k + "_est"] = s_est
        res["s_" + k + "_ref"] = s_ref
        res["s_" + k] = float(np.sqrt(np.nansum([s_est ** 2, s_ref ** 2]))) \
            if np.isfinite(s_est) or np.isfinite(s_ref) else np.nan
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
    """Gravação falsa: coluna + (cristal com hiato de Bragg) + ruído."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    ganho = 1 / (1 + (200 / (f + 1)) ** 4) / (1 + (f / 14000) ** 4)   # coluna
    if passo["tipo"] in ("cristal", "extra"):
        a, N = passo["a_mm"], passo["N"]
        ff = (lado_mm / a) ** 2
        # pequenas diferenças de montagem de repetição para repetição
        fb = f_bragg(a, temp_C, passo.get("dir", "ΓX")) * (1 - 0.1 * ff) * (1 + 0.003 * rng.standard_normal())
        meia = fb * ff
        N = N * (1 + 0.04 * rng.standard_normal())
        dentro = 1 / (1 + np.exp(-(f - fb + meia) / 60)) / (1 + np.exp((f - fb - meia) / 60))
        TdB = -N * 50 * ff * dentro
        TdB = 10 * np.log10(10 ** (TdB / 10) + 10 ** (-2.4))       # satura a −24 dB
        TdB += 0.8 * np.cos(2 * np.pi * f * 2 * N * a / 1000 / velocidade_som(temp_C))
        ganho = ganho * 10 ** (TdB / 20)
    y = np.fft.irfft(X * ganho, n) * 0.05 * (1 + 0.03 * rng.standard_normal())
    y += 2e-4 * rng.standard_normal(n)
    return y.astype(np.float32)
