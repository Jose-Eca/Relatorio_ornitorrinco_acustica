"""Curvas T(f) - T0 usadas na Fig. 2 do relatório (a = 48 mm, ΓX e ΓM).
Mesmo método de analise.py (Welch Hann 4096/50 %, troço 0.8–10.3 s, suavização 1/6 de oitava
em potência, T0 = mediana em 0.35–0.70 × c/(2a)), escrito só com numpy."""
import wave, json, numpy as np
from pathlib import Path
FS, NS = 48000, 4096
R = Path(__file__).resolve().parents[2]
X = R / "cristal_simples/cristal_simples/projeto/dados"
M = R / "cristal_lego_g/cristal_lego_g/projeto/dados"

def ler(p):
    w = wave.open(str(p)); y = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    return y

def welch(y):
    y = y[int(0.8 * FS):int(10.3 * FS)]
    win = np.hanning(NS + 1)[:-1]  # Hann periódica (scipy)
    step = NS // 2
    segs = [y[i:i + NS] for i in range(0, len(y) - NS + 1, step)]
    P = np.mean([np.abs(np.fft.rfft(win * (s - s.mean())))**2 for s in segs], axis=0)
    f = np.fft.rfftfreq(NS, 1 / FS)
    b = (f >= 500) & (f <= 12000)
    return f[b], P[b]

def suav(P, f, frac=1/6):
    c = np.concatenate([[0], np.cumsum(P)])
    lo = np.searchsorted(f, f * 2**(-frac/2), "left"); hi = np.searchsorted(f, f * 2**(frac/2), "right")
    return (c[hi] - c[lo]) / (hi - lo)

def espectro(pasta):
    Ps = []
    for w in sorted(pasta.glob("*rep*.wav")):
        f, P = welch(ler(w)); Ps.append(P)
    return f, np.mean(Ps, axis=0)

def c_som(T): return 331.3 * np.sqrt(1 + T / 273.15)

out = {}
for nome, base, ref, meds, T in [("GX", X, "ref_a48", ["a48_N4", "a48_N7", "a48_N10"], 20.5),
                                 ("GM", M, "ref_a48_fim", ["a48_N4", "a48_N7", "a48_N11"], 20.0)]:
    f, Pr = espectro(base / ref); Pr = suav(Pr, f)
    fc = c_som(T) / (2 * 0.048)
    ctrl = (f >= 0.35 * fc) & (f <= 0.70 * fc)
    out[nome] = {"f": f.tolist()}
    for m in meds:
        _, P = espectro(base / m)
        Tf = 10 * np.log10(suav(P, f) / Pr)
        out[nome][m] = (Tf - np.median(Tf[ctrl])).tolist()
Path(__file__).with_name("curvas_T_a48.json").write_text(json.dumps(out))
print("ok", {k: list(v) for k, v in out.items()})
