#!/usr/bin/env python3
"""
app.py — Cristal sónico de LEGO (LFE), rede quadrada com g pinos livres (a = (2+g)·8 mm),
  N de 11 a 1 filas, na diagonal (ΓM) ou ao longo do eixo (ΓX). Mesmos métodos da app anterior
  (analise.py igual). Um ecrã:
  esquerda: plano de medições · direita: instrução + botão MEDIR + gráfico.
  "Resultados": T(f) por a, profundidade vs N, f_centro vs 1/a e regressões.

  python app.py      (Windows: duplo clique em INICIAR.bat)
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa
from matplotlib.figure import Figure  # noqa

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import analise as A  # noqa: E402

try:
    import sounddevice as sd
except Exception:
    sd = None

PASTA = AQUI / "projeto"
DEF = {"entrada": "", "saida": "", "reps": 5, "dur": 10.0, "amp": 0.3, "temp_C": 20.0,
       "g": "2, 3, 4", "N": "11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1", "W": 4, "dir": "ΓM",
       "primeira": "B", "placa": 32, "simulacao": False}
PASSO_MM = 8.0   # passo dos pinos


def a_de_g(g):
    """g pinos livres entre tijolos → distância centro a centro (mm)."""
    return (2 + g) * PASSO_MM


def filas_max(g, d):
    """Máximo de filas que cabem na placa (diagonal: filas alternadas de 4 e 3 tijolos)."""
    lado, P = int(d["placa"]), int(round(2 + g))
    if d["dir"] != "ΓM":
        return (lado - 2) // P + 1
    melhor = 0
    for o in range(P):                       # encaixe da rede nos pinos
        for o2 in range(P):
            for k0 in range(-(lado // P) - 4, lado // P + 5):   # posição do eixo
                corrida, inicio = 0, None
                for m in range(-lado // P - 6, 2 * lado // P + 8):
                    tipo = "A" if (m + k0 + 3) % 2 == 0 else "B"
                    ok = True
                    for dk in ((-3, -1, 1, 3) if tipo == "A" else (-2, 0, 2)):
                        k = k0 + dk
                        x, y = (m + k) // 2 * P + o, (m - k) // 2 * P + o2
                        if (m + k) % 2 or not (0 <= x <= lado - 2 and 0 <= y <= lado - 2):
                            ok = False
                            break
                    if ok:
                        inicio = tipo if corrida == 0 else inicio
                        corrida += 1
                        melhor = max(melhor, corrida if inicio == d["primeira"] else corrida - 1)
                    else:
                        corrida = 0
    return melhor


def tipo_fila(n, primeira):
    """Fila n (1 = lado da coluna): A = 4 tijolos, B = 3 tijolos (alternam na diagonal)."""
    return primeira if n % 2 == 1 else ("B" if primeira == "A" else "A")
N_MIN_FC = 4   # f_centro vs 1/a usa a média dos N ≥ 4 (com N pequeno o hiato mal existe)


def escala(aj, sigma, nome):
    """χ²/ν > 1 → as barras de erro subestimam a dispersão real: incerteza × √(χ²/ν)."""
    k = aj.get("chi2nu", np.nan)
    if np.isfinite(k) and k > 1:
        return f"  →  com fator de escala √(χ²/ν): σ_{nome} = {sigma * np.sqrt(k):.2f}"
    return ""


CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]


def numeros_de(texto, tipo=float):
    return [tipo(v) for v in str(texto).replace(";", ",").split(",") if v.strip()]


def abrir(caminho):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(caminho))  # noqa
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(caminho)])
    except Exception:
        pass


def dispositivos():
    if sd is None:
        return [], []
    ent, sai = [], []
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        nome = f"{i}: {d['name']} [{apis[d['hostapi']]['name']}]"
        if d["max_input_channels"] > 0:
            ent.append(nome)
        if d["max_output_channels"] > 0:
            sai.append(nome)
    return ent, sai


def indice(txt):
    return int(txt.split(":")[0]) if txt and txt.split(":")[0].isdigit() else None


# ====================================================================== app
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cristal sónico — LFE")
        self.geometry("1280x800")
        PASTA.mkdir(exist_ok=True)
        (PASTA / "dados").mkdir(exist_ok=True)
        self.defs = {**DEF, **self._ler("definicoes.json", {})}
        self.plano = self._ler("plano.json", None) or self.gerar_plano(guardar=False)
        if not any(p["id"].endswith("_fim") for p in self.plano):
            self.gerar_plano(guardar=False)   # plano antigo: junta a 2.ª referência de cada bloco
            self.guardar()
        self.ocupado = False
        self._interface()
        self.atualizar_lista()

    # ---------------------------------------------------------------- ficheiros
    def _ler(self, nome, defeito):
        f = PASTA / nome
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else defeito

    def guardar(self):
        (PASTA / "definicoes.json").write_text(json.dumps(self.defs, indent=2), encoding="utf-8")
        (PASTA / "plano.json").write_text(json.dumps(self.plano, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    def pasta_dados(self):
        """Simulação grava à parte, para nunca se misturar com as medições reais."""
        return PASTA / ("dados_simulacao" if self.defs["simulacao"] else "dados")

    def pasta(self, pid):
        return self.pasta_dados() / pid

    def feito(self, pid):
        return (self.pasta(pid) / "meta.json").exists()

    # ---------------------------------------------------------------- plano
    def gerar_plano(self, guardar=True):
        d = self.defs
        W, dr, prim = int(d["W"]), d["dir"], d["primeira"]
        plano = []
        for g in numeros_de(d["g"]):
            a = a_de_g(g)
            nmx = filas_max(g, d)
            Ns = [N for N in sorted(numeros_de(d["N"], int), reverse=True) if N <= nmx]
            plano.append(dict(id=f"ref_a{a:g}", tipo="ref", a_mm=a, g=g, N=None, dir=dr,
                              instr=f"Tira TODOS os tijolos. Coluna de som e microfone no sítio "
                                    f"marcado e PRESOS COM FITA — não lhes toquem até à referência "
                                    f"de fim deste bloco. Referência para g = {g:g} (a = {a:g} mm)."))
            for k, N in enumerate(Ns):
                if k == 0:
                    txt = f"Monta a rede com {g:g} pino{'' if g == 1 else 's'} livre{'' if g == 1 else 's'} entre tijolos (a = {a:g} mm) e N = {N} filas"
                    if dr == "ΓM":
                        nA = (N + (prim == "A")) // 2
                        txt += (f" na diagonal: filas alternadas de 4 e 3 tijolos (A = 4, B = 3), "
                                f"1.ª fila do lado da coluna = {prim}; {nA} filas A + {N - nA} filas B "
                                f"= {4 * nA + 3 * (N - nA)} tijolos, simétrico em relação ao eixo.")
                    else:
                        txt += f" de {W} tijolos (centros separados por {a / 8:g} pinos)."
                    if Ns[0] < max(numeros_de(d["N"], int)):
                        txt += f"  (Com g = {g:g} só cabem {nmx} filas na placa {d['placa']}×{d['placa']}.)"
                else:
                    tiradas = range(Ns[k - 1], N, -1)
                    if dr == "ΓM":
                        tij = sum(4 if tipo_fila(n, prim) == "A" else 3 for n in tiradas)
                        txt = (f"Tira filas do lado do MICROFONE até ficarem N = {N} ({tij} tijolos); "
                               f"a última fila passa a ser {tipo_fila(N, prim)} "
                               f"({4 if tipo_fila(N, prim) == 'A' else 3} tijolos).")
                    else:
                        txt = f"Tira filas do lado do MICROFONE até ficarem N = {N}."
                plano.append(dict(id=f"a{a:g}_N{N}", tipo="cristal", a_mm=a, g=g, N=N, dir=dr,
                                  instr=txt))
            plano.append(dict(id=f"ref_a{a:g}_fim", tipo="ref", a_mm=a, g=g, N=None, dir=dr,
                              instr=f"Tira os tijolos que restam SEM mexer na coluna nem no "
                                    f"microfone. 2.ª referência de g = {g:g}: mede a deriva da "
                                    f"montagem e dá a incerteza σ_ref."))
        extras = [p for p in getattr(self, "plano", None) or [] if p["tipo"] == "extra"]
        self.plano = plano + extras
        if guardar:
            self.guardar()
            self.atualizar_lista()
        return self.plano

    def proximo(self):
        return next((p for p in self.plano if not self.feito(p["id"])), None)

    # ---------------------------------------------------------------- interface
    def _interface(self):
        st = ttk.Style(self)
        st.configure("Grande.TButton", font=("Segoe UI", 16, "bold"), padding=(20, 12))
        st.configure("Titulo.TLabel", font=("Segoe UI", 16, "bold"))
        st.configure("Passo.TLabel", font=("Segoe UI", 13, "bold"))

        topo = ttk.Frame(self, padding=8)
        topo.pack(fill="x")
        ttk.Label(topo, text="Cristal sónico de LEGO", style="Titulo.TLabel").pack(side="left")
        self.lbl_sim = ttk.Label(topo, text="  SIMULAÇÃO  ", background="#c05a00",
                                 foreground="white")
        for txt, cmd in (("📊 Resultados", self.resultados), ("📂 Abrir pasta", lambda: abrir(PASTA)),
                         ("⚙ Definições", self.definicoes)):
            ttk.Button(topo, text=txt, command=cmd).pack(side="right", padx=3)

        corpo = ttk.Frame(self, padding=8)
        corpo.pack(fill="both", expand=True)
        esq = ttk.Frame(corpo)
        esq.pack(side="left", fill="y")
        self.lista = ttk.Treeview(esq, columns=("g (a)", "N", "estado"), height=24, selectmode="browse")
        self.lista.heading("#0", text="passo")
        self.lista.column("#0", width=110)
        for c, w in (("g (a)", 70), ("N", 40), ("estado", 80)):
            self.lista.heading(c, text=c)
            self.lista.column(c, width=w, anchor="center")
        self.lista.tag_configure("feito", foreground="#0a7d0a")
        self.lista.pack(fill="y", expand=True)
        self.lista.bind("<<TreeviewSelect>>", lambda e: self.mostrar_passo())
        ttk.Button(esq, text="+ Passo extra (ΓM, defeito, remontagem…)",
                   command=self.extra).pack(fill="x", pady=(4, 0))
        ttk.Label(esq, text="Clica num passo para o ver ou repetir.\n"
                            "Sem seleção, MEDIR faz o próximo.", foreground="#666").pack(pady=4)

        dir_ = ttk.Frame(corpo, padding=(12, 0))
        dir_.pack(side="left", fill="both", expand=True)
        linha = ttk.Frame(dir_)
        linha.pack(fill="x")
        info = ttk.Frame(linha)
        info.pack(side="left", fill="x", expand=True)
        self.lbl_passo = ttk.Label(info, style="Passo.TLabel")
        self.lbl_passo.pack(anchor="w")
        self.lbl_instr = ttk.Label(info, font=("Segoe UI", 11), wraplength=700, justify="left")
        self.lbl_instr.pack(anchor="w", pady=4)
        self.bt = ttk.Button(linha, text="▶  MEDIR", style="Grande.TButton", command=self.medir)
        self.bt.pack(side="right")
        self.lbl_estado = ttk.Label(dir_, text="Pronto.", foreground="#555", wraplength=900,
                                    justify="left")
        self.lbl_estado.pack(anchor="w")

        self.lbl_res = ttk.Label(dir_, font=("Segoe UI", 11), justify="left")
        self.lbl_res.pack(side="bottom", anchor="w", pady=6)
        self.fig = Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=dir_)
        barra = NavigationToolbar2Tk(self.canvas, dir_, pack_toolbar=False)
        barra.update()
        barra.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def atualizar_lista(self):
        sel = self.lista.selection()
        self.lista.delete(*self.lista.get_children())
        for p in self.plano:
            ok = self.feito(p["id"])
            self.lista.insert("", "end", iid=p["id"], text=p["id"], tags=("feito",) if ok else (),
                              values=(f"{p['a_mm'] / 8 - 2:g} ({p['a_mm']:g})", p["N"] or "",
                                      "✔ feito" if ok else ""))
        if sel and self.lista.exists(sel[0]):
            self.lista.selection_set(sel[0])
        if self.defs["simulacao"]:
            self.lbl_sim.pack(side="left", padx=10)
        else:
            self.lbl_sim.pack_forget()
        self.mostrar_passo()

    def passo_alvo(self):
        sel = self.lista.selection()
        return next(p for p in self.plano if p["id"] == sel[0]) if sel else self.proximo()

    def mostrar_passo(self):
        p = self.passo_alvo()
        if p is None:
            self.lbl_passo.config(text="Plano concluído ✔")
            self.lbl_instr.config(text="Carrega em 📊 Resultados.")
            return
        self.lbl_passo.config(text=p["id"] + ("   (já medido — MEDIR repete)" if self.feito(p["id"])
                                               else ""))
        self.lbl_instr.config(text=p["instr"])
        if self.feito(p["id"]):
            self.desenhar(p)

    # ---------------------------------------------------------------- medir
    def medir(self):
        if self.ocupado:
            return
        p = self.passo_alvo()
        if p is None:
            return
        # verificar o áudio ANTES de mexer nos dados antigos
        if not self.defs["simulacao"] and (sd is None or not self.defs["entrada"]
                                           or not self.defs["saida"]):
            messagebox.showerror("Áudio", "Escolhe o microfone e a coluna em ⚙ Definições "
                                          "(ou liga o modo simulação).")
            return
        if self.feito(p["id"]):
            if not messagebox.askyesno("Repetir", f"{p['id']} já foi medido. Medir de novo?\n"
                                                  "(A medição atual vai para _antigos e deixa "
                                                  "de contar.)"):
                return
            antigo = self.pasta_dados() / "_antigos" / f"{p['id']}_{datetime.now():%H%M%S}"
            antigo.parent.mkdir(exist_ok=True)
            shutil.move(str(self.pasta(p["id"])), str(antigo))
        self.ocupado = True
        self.bt.state(["disabled"])
        threading.Thread(target=self._gravar, args=(p,), daemon=True).start()

    def _gravar(self, p):
        d = self.defs
        try:
            pasta = self.pasta(p["id"])
            pasta.mkdir(parents=True, exist_ok=True)
            n = int(d["dur"] * A.FS)
            qcs = []
            for r in range(int(d["reps"])):
                self.after(0, self.lbl_estado.config,
                           {"text": f"A gravar repetição {r + 1} de {d['reps']}…"})
                if d["simulacao"]:
                    # como a gravação real: atraso do Bluetooth, ruído, silêncio no fim
                    ruido = A.simular(n, p, d["temp_C"])
                    y = np.concatenate([np.zeros(int(0.55 * A.FS)), ruido,
                                        np.zeros(int(0.95 * A.FS))])
                    y = (y + 2e-4 * np.random.default_rng().standard_normal(y.size)).astype(np.float32)
                    time.sleep(0.3)
                else:
                    x = np.random.default_rng().standard_normal(n)
                    x = d["amp"] * x / np.abs(x).max()
                    x = np.concatenate([x, np.zeros(int(1.5 * A.FS))]).astype(np.float32)
                    y = sd.playrec(np.column_stack([x, x]), samplerate=A.FS, channels=1,
                                   dtype="float32",
                                   device=(indice(d["entrada"]), indice(d["saida"])))
                    sd.wait()
                    y = y[:, 0]
                sf.write(pasta / f"{p['id']}_rep{r + 1:02d}.wav", y, A.FS)
                qcs.append(A.qualidade(y, d["dur"]))
            pico = max(q["pico"] for q in qcs)          # TODAS as repetições, não só a última
            meta = dict(id=p["id"], tipo=p["tipo"], a_mm=p["a_mm"], N=p["N"], dir=p.get("dir", "ΓX"),
                        dur=d["dur"],
                        temp_C=d["temp_C"], hora=datetime.now().isoformat(timespec="seconds"),
                        pico=pico)
            (pasta / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self.after(0, self._fim, p, qcs, None)
        except Exception as e:
            self.after(0, self._fim, p, [], e)

    def _fim(self, p, qcs, erro):
        self.ocupado = False
        self.bt.state(["!disabled"])
        if erro:
            shutil.rmtree(self.pasta(p["id"]), ignore_errors=True)
            self.lbl_estado.config(text="Erro.")
            messagebox.showerror("Erro na gravação", str(erro))
            return
        av = A.avisos_qc(qcs)
        nivel = float(np.median([q["nivel_db"] for q in qcs]))
        self.lbl_estado.config(text=f"{p['id']} gravado (ruído na banda {nivel:.0f} dBFS, pico "
                                    f"{max(q['pico'] for q in qcs):.2f}).  "
                                    + ("⚠ " + " · ".join(av) if av else "✔ sem problemas"))
        if any("REPETE" in t or "SATURAÇÃO" in t or "mexeu" in t for t in av):
            messagebox.showwarning("Atenção", f"{p['id']}:\n\n" + "\n".join(av))
        self.lista.selection_remove(self.lista.selection())
        self.desenhar(p)
        self.atualizar_lista()
        self.lbl_passo.config(text=self.lbl_passo.cget("text"))

    # ---------------------------------------------------------------- gráfico de um passo
    def desenhar(self, p):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ref = f"ref_a{p['a_mm']:g}"
        if p["tipo"] == "ref":
            meta, f, P = A.ler_espectros(self.pasta(p["id"]))
            ax.plot(f, 10 * np.log10(P.mean(0)), color=CORES[0], lw=0.6, alpha=0.4)
            ax.plot(f, 10 * np.log10(A.suavizar_potencia(P.mean(0), f)), color=CORES[0], lw=1.6)
            ax.set(ylabel="espectro (dB)", title=f"Referência {p['id']} (sem cristal)")
            av = A.avisos_qc(meta["_qc"])
            self.lbl_res.config(text="Referência: deve ser uma curva suave, sem buracos fundos.\n"
                                     + ("⚠ " + " · ".join(av) if av else "Qualidade: ✔"))
        elif not self.feito(ref):
            ax.text(0.5, 0.5, f"Falta medir a referência {ref}", ha="center",
                    transform=ax.transAxes)
            self.lbl_res.config(text="")
        else:
            try:
                r = A.analisar(self.pasta(p["id"]), self.pasta(ref))
            except Exception as e:
                ax.text(0.5, 0.5, f"Não consegui analisar: {e}", ha="center", transform=ax.transAxes)
                self.lbl_res.config(text="")
                self.canvas.draw_idle()
                return
            gm = A.eh_gm(r["dir"])
            nome_b = A.nome_bragg(r["dir"])
            for k, (lo, hi) in enumerate(A.BASE_GM if gm else A.BASE_GX):
                ax.axvspan(lo * r["fb"], hi * r["fb"], color=CORES[2], alpha=0.08, lw=0,
                           label="zona do nível base T0" if k == 0 else None)
            ax.fill_between(r["f"], r["T"] - r["sT"], r["T"] + r["sT"], color=CORES[0],
                            alpha=0.25, lw=0)
            ax.plot(r["f"], r["Ts"], color=CORES[0], lw=1.6, label="T(f)")
            ax.axhline(r["T0"], color=CORES[2], lw=0.8)
            ax.axhline(r["T0"] - A.LIMIAR_DB, color=CORES[1], ls=":", lw=1,
                       label=f"limiar (T0 − {A.LIMIAR_DB} dB)")
            ax.axvline(r["fb"], color=CORES[4], ls="--", lw=1, label=f"{nome_b} = {r['fb']:.0f} Hz")
            if r["hiato"]:
                ax.axvspan(*r["hiato"], color=CORES[1], alpha=0.15, label="hiato")
            ax.set(ylabel="T (dB)", title=f"Transmissão {p['id']}")
            ax.legend(loc="lower left", fontsize=8)
            jan = "1.05–1.35 × f_B" if gm else "f_B ± 10 %"
            txt = (f"Profundidade (média em {jan}): {r['profundidade']:.1f} ± "
                   f"{r['s_profundidade']:.1f} dB  (estat. {r['s_profundidade_est']:.1f}, ref. "
                   f"{r['s_profundidade_ref']:.1f})\n")
            if gm and np.isfinite(r["f_inf"]):
                txt += (f"Bordo de baixo do hiato f_inf = {r['f_inf']:.0f} ± {r['s_f_inf']:.0f} Hz "
                        f"= {r['f_inf'] / r['fb']:.3f} × {nome_b}"
                        + ("   (aberto por cima: normal na diagonal — difração lateral)"
                           if r["aberto"] else f",  bordo de cima {r['f_sup']:.0f} Hz") + "\n")
            elif np.isfinite(r["fc"]):
                txt += (f"Hiato: {r['f_inf']:.0f}–{r['f_sup']:.0f} Hz,  centro {r['fc']:.0f} ± "
                        f"{r['s_fc']:.0f} Hz,  largura {r['largura']:.0f} ± {r['s_largura']:.0f} Hz\n")
            else:
                txt += f"Não há hiato abaixo do limiar perto de {nome_b} (normal para N pequeno).\n"
            txt += (f"Referência: média de {r['n_refs']} ref. — {r['modo_ref']}.   "
                    f"Sinal/fundo no hiato ≥ {r['snr_min']:.0f} dB.   "
                    f"Repetições usadas: {len(r['meta']['_usadas'])}/{r['n_reps']}")
            if r["avisos"]:
                txt += "\n⚠ " + " · ".join(r["avisos"])
            self.lbl_res.config(text=txt)
        ax.set_xlabel("Frequência (Hz)")
        ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- resultados
    def calcular(self):
        linhas = []
        for p in self.plano:
            ref = f"ref_a{p['a_mm']:g}"
            if p["tipo"] == "cristal" and self.feito(p["id"]) and self.feito(ref):
                try:
                    r = A.analisar(self.pasta(p["id"]), self.pasta(ref))
                except Exception:
                    continue
                r["a_mm"], r["N"] = p["a_mm"], p["N"]
                linhas.append(r)
        return linhas

    def resultados(self):
        L = self.calcular()
        if not L:
            messagebox.showinfo("Resultados", "Ainda não há medições com cristal.")
            return
        janela = tk.Toplevel(self)
        janela.title("Resultados")
        janela.geometry("1100x720")
        barra_b = ttk.Frame(janela)
        barra_b.pack(side="bottom", fill="x")
        caixa = tk.Text(janela, height=9, font=("Consolas", 10))
        caixa.pack(side="bottom", fill="x")
        nb = ttk.Notebook(janela)
        nb.pack(fill="both", expand=True)
        figs = {}
        for nome in ("T(f) por a", "Profundidade vs N", "f do hiato vs 1/a"):
            fr = ttk.Frame(nb)
            nb.add(fr, text=f"  {nome}  ")
            fig = Figure(figsize=(8, 4.5), dpi=100)
            cv = FigureCanvasTkAgg(fig, master=fr)
            tb = NavigationToolbar2Tk(cv, fr, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
            cv.get_tk_widget().pack(fill="both", expand=True)
            figs[nome] = fig
        texto = self.graficos_resultados(L, figs)
        caixa.insert("1.0", texto)

        def guardar():
            out = PASTA / "resultados"
            out.mkdir(exist_ok=True)
            for nome, fig in figs.items():
                fig.savefig(out / (nome.replace("/", "_").replace(" ", "_") + ".png"), dpi=200)
            (out / "resultados.txt").write_text(texto, encoding="utf-8")
            with open(out / "tabela.csv", "w", encoding="utf-8") as fh:
                fh.write("dir,a_mm,N,prof_dB,s_prof,s_prof_est,s_prof_ref,fc_Hz,s_fc,s_fc_est,s_fc_ref,"
                         "largura_Hz,s_largura,f_inf_Hz,s_f_inf,f_sup_Hz,aberto,f_bragg_Hz,T0_dB,"
                         "snr_min_dB,n_refs,modo_ref,reps_usadas,avisos\n")
                for r in L:
                    fh.write(f"{r['dir']},{r['a_mm']},{r['N']},{r['profundidade']:.2f},"
                             f"{r['s_profundidade']:.2f},{r['s_profundidade_est']:.2f},"
                             f"{r['s_profundidade_ref']:.2f},{r['fc']:.1f},{r['s_fc']:.1f},"
                             f"{r['s_fc_est']:.1f},{r['s_fc_ref']:.1f},"
                             f"{r['largura']:.1f},{r['s_largura']:.1f},{r['f_inf']:.1f},"
                             f"{r['s_f_inf']:.1f},{r['f_sup']:.1f},{int(r['aberto'])},"
                             f"{r['fb']:.1f},{r['T0']:.2f},{r['snr_min']:.1f},"
                             f"{r['n_refs']},\"{r['modo_ref']}\",{len(r['meta']['_usadas'])},"
                             f"\"{' · '.join(r['avisos'])}\"\n")
            messagebox.showinfo("Guardado", f"Gráficos, tabela e texto em\n{out}", parent=janela)
            abrir(out)

        ttk.Button(barra_b, text="💾 Guardar tudo (PNG + CSV + TXT)", command=guardar).pack(pady=6)

    def graficos_resultados(self, L, figs):
        As = sorted({r["a_mm"] for r in L})
        temp = np.mean([r["meta"]["temp_C"] for r in L])
        c_teo = A.velocidade_som(temp)
        txt = [f"Temperatura média {temp:.1f} °C → c teórico = {c_teo:.1f} m/s", ""]

        # 1) T(f) por a
        fig = figs["T(f) por a"]
        cols = 2 if len(As) > 1 else 1
        rows = int(np.ceil(len(As) / cols))
        for k, a in enumerate(As):
            ax = fig.add_subplot(rows, cols, k + 1)
            R = sorted([r for r in L if r["a_mm"] == a], key=lambda r: r["N"])
            for j, r in enumerate(R):
                tom = 0.35 + 0.65 * (j + 1) / len(R)
                ax.plot(r["f"], r["Ts"], color=(0.1, 0.35, 0.75, tom), lw=1.2, label=f"N = {r['N']}")
            ax.axvline(R[0]["fb"], color=CORES[4], ls="--", lw=1)
            if not A.eh_gm(R[0]["dir"]):
                ax.axvline(2 * R[0]["fb"], color=CORES[4], ls=":", lw=1)
            ax.set(title=f"g = {a / 8 - 2:g}  (a = {a:g} mm)", xlabel="f (Hz)", ylabel="T (dB)")
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(alpha=0.3)
        fig.tight_layout()

        # 2) profundidade vs N
        fig = figs["Profundidade vs N"]
        ax = fig.add_subplot(111)
        txt.append("PROFUNDIDADE vs N   (D = α·N + b)")
        for k, a in enumerate(As):
            R = sorted([r for r in L if r["a_mm"] == a], key=lambda r: r["N"])
            N = np.array([r["N"] for r in R])
            D = np.array([r["profundidade"] for r in R])
            sD = np.array([r["s_profundidade"] for r in R])
            ax.errorbar(N, D, sD, fmt="o", color=CORES[k % 6], capsize=3, label=f"a = {a:g} mm")
            if len(R) >= 3:
                aj = A.reta(N, D, sD)
                xx = np.linspace(0, N.max(), 20)
                ax.plot(xx, aj["m"] * xx + aj["b"], color=CORES[k % 6], lw=1)
                txt.append(f"  a = {a:g} mm: α = {aj['m']:.2f} ± {aj['sm']:.2f} dB/fila, "
                           f"b = {aj['b']:.1f} ± {aj['sb']:.1f} dB, χ²/ν = {aj['chi2nu']:.1f}"
                           + escala(aj, aj["sm"], "α"))
        ax.set(xlabel="Número de filas N", ylabel="Profundidade (dB)",
               title="Profundidade do hiato vs número de filas")
        ax.set_xlim(left=0)
        ax.axhline(0, color="k", lw=0.6)    # profundidades negativas = medição falhada: mostrar
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        # 3) f do hiato vs 1/a  (para cada a: média de f_centro sobre todos os N ≥ N_MIN_FC;
        #    a incerteza é a dispersão entre N, que inclui montagem e referência)
        fig = figs["f do hiato vs 1/a"]
        # ΓX: f_centro = c/(2a) → c = 2·declive;  ΓM (diagonal): f_inf = c/(√2·a) → c = √2·declive
        dirs = sorted({r["dir"] for r in L})
        dir_ = dirs[0] if len(dirs) == 1 else self.defs["dir"]
        if len(dirs) > 1:
            txt.append(f"⚠ há medições em {', '.join(dirs)}: no ajuste de c só entram as de {dir_}.")
        Ld = [r for r in L if r["dir"] == dir_]
        gm = A.eh_gm(dir_)
        K = np.sqrt(2) if gm else 2.0
        nome_k = "√2" if gm else "2"
        nome_b = A.nome_bragg(dir_)
        nome_f = "f_inf (bordo de baixo)" if gm else "f_centro"
        ax = fig.add_subplot(111)
        txt += ["", f"{nome_f} vs 1/a   (f = m·(1/a) + b,  c = {nome_k}·m;  valor de cada a = média "
                    f"dos N ≥ {N_MIN_FC}, erro = desvio-padrão entre N / √n)"]
        x, y, sy, rot = [], [], [], []
        for a in As:
            v = np.array([r["f_hiato"] for r in Ld if r["a_mm"] == a and r["N"] >= N_MIN_FC
                          and np.isfinite(r["f_hiato"])])
            if v.size == 0:
                continue
            x.append(1000 / a)
            y.append(v.mean())
            sy.append(v.std(ddof=1) / np.sqrt(v.size) if v.size >= 2 else np.nan)
            rot.append(f"  a = {a:g} mm: {nome_f.split()[0]} = {v.mean():.0f} ± {sy[-1]:.0f} Hz "
                       f"(n = {v.size}; / f_Bragg = {v.mean() / (c_teo / K * 1000 / a):.3f})")
        txt += rot
        x, y, sy = map(np.array, (x, y, sy))
        if len(x):
            ax.errorbar(x, y, sy, fmt="o", color=CORES[0], capsize=3,
                        label=f"medido (média N ≥ {N_MIN_FC})")
            xx = np.linspace(0, x.max() * 1.1, 20)
            ax.plot(xx, c_teo / K * xx, "--", color=CORES[4], label=f"teoria {nome_b} (c = {c_teo:.0f} m/s)")
            if len(x) >= 3:
                aj = A.reta(x, y, sy)
                ax.plot(xx, aj["m"] * xx + aj["b"], color=CORES[1],
                        label=f"ajuste: c = {K * aj['m']:.0f} ± {K * aj['sm']:.0f} m/s")
                z = abs(K * aj["m"] - c_teo) / (K * aj["sm"])
                txt.append(f"  reta livre: c = {K * aj['m']:.1f} ± {K * aj['sm']:.1f} m/s "
                           f"(teórico {c_teo:.1f}; z = {z:.1f}), b = {aj['b']:.0f} ± "
                           f"{aj['sb']:.0f} Hz, χ²/ν = {aj['chi2nu']:.1f}" + escala(aj, K * aj["sm"], "c"))
            else:
                txt.append("  reta livre: precisa de pelo menos 3 valores de a.")
            if len(x) >= 2:
                a0 = A.reta_origem(x, y, sy)
                z0 = abs(K * a0["m"] - c_teo) / (K * a0["sm"])
                ax.plot(xx, a0["m"] * xx, ":", color=CORES[2],
                        label=f"pela origem: c = {K * a0['m']:.0f} ± {K * a0['sm']:.0f} m/s")
                txt.append(f"  reta pela origem (f = m/a): c = {K * a0['m']:.1f} ± "
                           f"{K * a0['sm']:.1f} m/s (z = {z0:.1f}), χ²/ν = {a0['chi2nu']:.1f}"
                           + escala(a0, K * a0["sm"], "c"))
        ax.set(xlabel="1/a (m⁻¹)", ylabel=f"{nome_f} (Hz)", title=f"{nome_f} vs 1/a  ({dir_})")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        txt += ["", "TABELA", f"  {'passo':<9}{'prof. (dB)':>14}{'f_hiato (Hz)':>16}"
                f"{'largura (Hz)':>15}{'f_Bragg (Hz)':>13}   (f_hiato: ΓX centro, ΓM bordo de baixo)"]
        for r in sorted(L, key=lambda r: (r["a_mm"], r["N"])):
            def fm(v, s):
                return f"{v:.0f} ± {s:.0f}" if np.isfinite(v) else "—"
            txt.append(f"  {r['meta']['id']:<9}{r['profundidade']:>8.1f} ± {r['s_profundidade']:<4.1f}"
                       f"{fm(r['f_hiato'], r['s_f_hiato']):>16}"
                       f"{'aberto' if r['aberto'] else fm(r['largura'], r['s_largura']):>15}"
                       f"{r['fb']:>11.0f}"
                       + (f"   ⚠ {' · '.join(r['avisos'])}" if r["avisos"] else "")
                       + ("   ⚠ ref. de outro dia" if r["modo_ref"].startswith("OUTRO") else "")
                       + (f"   ⚠ sinal/fundo {r['snr_min']:.0f} dB" if r["snr_min"] < 10 else ""))
        txt += ["", "Incerteza = √(σ_estat² + σ_ref²): σ_estat das 5 repetições, σ_ref da dispersão "
                    "entre referências (ver documento de métodos)."]
        return "\n".join(txt)

    # ---------------------------------------------------------------- passo extra
    def extra(self):
        j = tk.Toplevel(self)
        j.title("Passo extra")
        j.transient(self)
        vs = {k: tk.StringVar(value=v) for k, v in
              (("nome", "remont"), ("a", "32"), ("N", "8"), ("dir", self.defs["dir"]), ("instr", ""))}
        linhas = (("Nome curto", "nome", None), ("a (mm)", "a", None), ("N", "N", None),
                  ("Direção", "dir", ["ΓX", "ΓM"]), ("Instrução", "instr", None))
        for i, (rot, k, op) in enumerate(linhas):
            ttk.Label(j, text=rot).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            w = ttk.Combobox(j, textvariable=vs[k], values=op, width=8, state="readonly") \
                if op else ttk.Entry(j, textvariable=vs[k], width=46 if k == "instr" else 12)
            w.grid(row=i, column=1, sticky="w", padx=8)
        ttk.Label(j, text="Exemplos: ΓM = cristal rodado 45° (hiato em c/(√2·a));\n"
                          "defeito = tirar a coluna do meio; remontagem = desmontar e montar\n"
                          "de novo a mesma configuração (incerteza sistemática).\n"
                          "Usa a referência do mesmo a. Não entra nas regressões.",
                  foreground="#666").grid(row=5, column=0, columnspan=2, padx=8, pady=6)

        def ok():
            try:
                a, N = float(vs["a"].get()), int(vs["N"].get())
            except ValueError:
                messagebox.showerror("Passo extra", "a e N têm de ser números.", parent=j)
                return
            if not self.feito(f"ref_a{a:g}") and f"ref_a{a:g}" not in [p["id"] for p in self.plano]:
                self.plano.append(dict(id=f"ref_a{a:g}", tipo="ref", a_mm=a, N=None,
                                       instr=f"Tira todas as colunas. Referência para a = {a:g} mm."))
            base = "x_" + "".join(c for c in vs["nome"].get() if c.isalnum())[:12] + f"_a{a:g}_N{N}"
            pid, k = base, 2
            while pid in [p["id"] for p in self.plano]:
                pid, k = f"{base}_{k}", k + 1
            self.plano.append(dict(id=pid, tipo="extra", a_mm=a, N=N, dir=vs["dir"].get(),
                                   instr=vs["instr"].get() or f"Passo extra: {vs['nome'].get()}"))
            self.guardar()
            self.atualizar_lista()
            j.destroy()

        ttk.Button(j, text="Adicionar", command=ok).grid(row=6, column=1, sticky="e", padx=8, pady=8)

    # ---------------------------------------------------------------- definições
    def definicoes(self):
        j = tk.Toplevel(self)
        j.title("Definições")
        j.transient(self)
        ent, sai = dispositivos()
        vs = {}
        campos = [("entrada", "Microfone", ent), ("saida", "Coluna", sai),
                  ("reps", "Repetições", None), ("dur", "Duração (s)", None),
                  ("amp", "Amplitude (0–1)", None), ("temp_C", "Temperatura (°C)", None),
                  ("g", "g: pinos livres entre tijolos (lista)", None), ("N", "Números de filas N", None),
                  ("dir", "Direção (ΓM = diagonal, ΓX = eixo)", ["ΓM", "ΓX"]),
                  ("primeira", "Diagonal: 1.ª fila do lado da coluna (A = 4, B = 3)", ["A", "B"]),
                  ("placa", "Placa (pinos de lado): 32 = uma, 64 = 2×2", ["32", "64"]),
                  ("W", "Tijolos por fila (só ΓX)", None)]
        for i, (k, rot, op) in enumerate(campos):
            ttk.Label(j, text=rot).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            v = tk.StringVar(value=str(self.defs[k]))
            w = ttk.Combobox(j, textvariable=v, values=op, width=50) if op is not None \
                else ttk.Entry(j, textvariable=v, width=20)
            w.grid(row=i, column=1, sticky="w", padx=8)
            vs[k] = v
        vsim = tk.BooleanVar(value=self.defs["simulacao"])
        ttk.Checkbutton(j, text="Modo simulação (treinar sem hardware)",
                        variable=vsim).grid(row=len(campos), column=1, sticky="w", padx=8)
        ttk.Label(j, text="Bluetooth: escolhe a saída «Headphones/Stereo», nunca «Hands-Free».\n"
                          "Microfone e coluna da mesma família ([MME] de preferência).",
                  foreground="#666").grid(row=len(campos) + 1, column=0, columnspan=2, padx=8,
                                          pady=6, sticky="w")

        def ler():
            try:
                for k, v in vs.items():
                    t = type(DEF[k])
                    self.defs[k] = t(float(v.get())) if t in (int, float) else v.get()
            except ValueError:
                messagebox.showerror("Definições", "Há um campo numérico inválido.", parent=j)
                return False
            if self.defs["dir"] not in ("ΓM", "ΓX") or self.defs["primeira"] not in ("A", "B") \
                    or self.defs["placa"] not in (32, 64):
                messagebox.showerror("Definições", "Direção: ΓM ou ΓX; 1.ª fila: A ou B; placa: 32 ou 64.",
                                     parent=j)
                return False
            self.defs["simulacao"] = vsim.get()
            self.guardar()
            return True

        def testar():
            if not ler():
                return
            if self.defs["simulacao"] or sd is None:
                messagebox.showinfo("Teste", "Em simulação não há som.", parent=j)
                return
            try:
                # o mesmo sinal da medição (ruído gaussiano normalizado ao pico), 3 s
                x = np.random.default_rng().standard_normal(3 * A.FS)
                x = self.defs["amp"] * x / np.abs(x).max()
                x = np.concatenate([x, np.zeros(int(1.5 * A.FS))]).astype(np.float32)
                y = sd.playrec(np.column_stack([x, x]), samplerate=A.FS, channels=1,
                               dtype="float32", device=(indice(self.defs["entrada"]),
                                                        indice(self.defs["saida"])))
                sd.wait()
                q = A.qualidade(y[:, 0], 3.0)
                msg = (f"Ruído na banda 0.5–12 kHz: {q['nivel_db']:.0f} dBFS\n"
                       f"Pico: {q['pico']:.2f}   (zumbido < 100 Hz: {100 * q['zumbido']:.0f} % da potência)\n\n")
                av = A.avisos_qc([q])
                msg += "\n".join(av) if av else "Nível bom ✔ (alvo: ruído na banda −30 a −15 dBFS, pico < 0.8)"
                messagebox.showinfo("Teste de som", msg, parent=j)
            except Exception as e:
                messagebox.showerror("Teste de som", str(e), parent=j)

        def ok():
            chave = ("g", "N", "W", "dir", "primeira", "placa")
            antigo = [self.defs[k] for k in chave]
            if ler():
                if [self.defs[k] for k in chave] != antigo:
                    self.gerar_plano()
                self.atualizar_lista()
                j.destroy()

        b = ttk.Frame(j)
        b.grid(row=len(campos) + 2, column=0, columnspan=2, pady=8)
        ttk.Button(b, text="🔊 Testar som", command=testar).pack(side="left", padx=4)
        ttk.Button(b, text="OK", command=ok).pack(side="left", padx=4)


if __name__ == "__main__":
    App().mainloop()
