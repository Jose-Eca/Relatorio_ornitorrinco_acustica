#!/usr/bin/env python3
"""
app.py — Cristal sónico de LEGO (LFE): análise CONJUNTA das duas experiências.
  Lê as gravações das duas apps de medição (ΓX: cristal_simples; ΓM: cristal_lego_g), sem lhes
  mexer, e faz a mesma análise às duas direções (contas em analise.py, gráficos em graficos.py).

  python app.py                (Windows: duplo clique em INICIAR.bat)
  python app.py --sem-janela   (só calcula e guarda tudo em resultados/)
"""
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import matplotlib

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import analise as A  # noqa: E402
import graficos as G  # noqa: E402
import teoria as T  # noqa: E402
import documento as D  # noqa: E402

CONFIG = AQUI / "config.json"
CACHE = AQUI / "_cache"
SAIDA = AQUI / "resultados"
DEF = {"conjuntos": [
    {"nome": "perpendicular", "dir": "ΓX",
     "pasta": str(AQUI.parent.parent / "cristal_simples" / "cristal_simples" / "projeto" / "dados")},
    {"nome": "diagonal", "dir": "ΓM",
     "pasta": str(AQUI.parent / "cristal_lego_g" / "projeto" / "dados")}],
    "modo_ref": "bloco"}


def ler_config():
    cfg = json.loads(json.dumps(DEF))
    if CONFIG.exists():
        lido = json.loads(CONFIG.read_text(encoding="utf-8"))
        cfg["modo_ref"] = lido.get("modo_ref", cfg["modo_ref"])
        for c, novo in zip(cfg["conjuntos"], lido.get("conjuntos", [])):
            c["pasta"] = novo.get("pasta") or c["pasta"]
    return cfg


def guardar_config(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def calcular(cfg, progresso=None):
    """Lê as duas pastas e faz a análise toda. Devolve dict(R, fora, texto, modo, teoria, doc)."""
    meds = A.carregar(cfg["conjuntos"], cache=CACHE, progresso=progresso)
    linhas, fora, _ = A.analisar_tudo(meds, cfg["modo_ref"])
    if not linhas:
        raise ValueError("Não há nenhuma medição com cristal (e referência) nas pastas escolhidas.")
    R = A.resultados(linhas)
    res = dict(R=R, fora=fora, modo=cfg["modo_ref"], texto=A.relatorio(R, fora, cfg["modo_ref"]))
    if progresso:
        progresso(1, 1, "")
    try:        # a teoria não pode impedir que se vejam as medições
        res["teoria"] = T.comparar(R)
    except Exception as e:
        res["teoria"], res["erro_teoria"] = None, e
    res["doc"] = D.documento(res)
    return res


def valores_a(R):
    return sorted({r["a_mm"] for r in R["linhas"]})


def guardar_tudo(res, pasta=SAIDA):
    """PNG de todos os gráficos + resultados.txt + tabela.csv + ajustes.csv + teoria e método."""
    from matplotlib.figure import Figure
    R = res["R"]
    pasta.mkdir(exist_ok=True)
    figs = {"profundidade_vs_N": G.fig_prof, "f_vs_1_2d": G.fig_freq, "comparacao_GX_GM": G.fig_comp}
    for nome, desenhar in figs.items():
        fig = Figure(figsize=(12, 7), layout="constrained")
        desenhar(fig, R)
        fig.savefig(pasta / f"{nome}.png", dpi=200)
    for a in valores_a(R):
        fig = Figure(figsize=(12, 5.5), layout="constrained")
        G.fig_T(fig, R, a)
        fig.savefig(pasta / f"T_a{a:g}.png", dpi=200)
    (pasta / "resultados.txt").write_text(res["texto"], encoding="utf-8")
    (pasta / "tabela.csv").write_text(A.tabela_csv(R["linhas"]), encoding="utf-8")
    (pasta / "ajustes.csv").write_text(A.ajustes_csv(R), encoding="utf-8")
    (pasta / "teoria_e_metodo.txt").write_text(D.texto_simples(res["doc"]), encoding="utf-8")
    if res["teoria"]:
        fig = Figure(figsize=(16, 8), layout="constrained")
        T.figura(fig, R, res["teoria"])
        fig.savefig(pasta / "teoria_vs_medido.png", dpi=160)
        (pasta / "teoria.txt").write_text(T.texto(res["teoria"]), encoding="utf-8")
    return pasta


def abrir(caminho):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(caminho))  # noqa
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(caminho)])
    except Exception:
        pass


def sem_janela():
    matplotlib.use("Agg")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = calcular(ler_config())
    print(res["texto"])
    print(f"\nGuardado em {guardar_tudo(res)}")


# ====================================================================== janela
def criar_app():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Cristal sónico — análise conjunta ΓX + ΓM")
            self.geometry("1320x860")
            self.cfg = ler_config()
            self.res = None
            self.ocupado = False
            self.fila = queue.Queue()
            self._interface()
            self.after(100, self.recalcular)

        # ------------------------------------------------------------ interface
        def _interface(self):
            st = ttk.Style(self)
            st.configure("Titulo.TLabel", font=("Segoe UI", 15, "bold"))
            topo = ttk.Frame(self, padding=8)
            topo.pack(fill="x")
            ttk.Label(topo, text="Cristal sónico de LEGO — ΓX e ΓM, a mesma análise",
                      style="Titulo.TLabel").pack(side="left")
            for txt, cmd in (("📂 Abrir resultados", lambda: abrir(SAIDA) if SAIDA.exists() else None),
                             ("💾 Guardar tudo", self.guardar), ("⚙ Pastas de dados", self.pastas),
                             ("🔄 Reler dados", self.recalcular)):
                ttk.Button(topo, text=txt, command=cmd).pack(side="right", padx=3)

            linha = ttk.Frame(self, padding=(8, 0))
            linha.pack(fill="x")
            ttk.Label(linha, text="Referência:").pack(side="left")
            self.v_modo = tk.StringVar(value=A.MODOS_REF[self.cfg["modo_ref"]])
            cb = ttk.Combobox(linha, textvariable=self.v_modo, values=list(A.MODOS_REF.values()),
                              state="readonly", width=68)
            cb.pack(side="left", padx=6)
            cb.bind("<<ComboboxSelected>>", lambda e: self.mudar_modo())
            self.lbl_estado = ttk.Label(linha, text="", foreground="#555")
            self.lbl_estado.pack(side="left", padx=10)
            self.barra = ttk.Progressbar(linha, length=220)

            self.nb = ttk.Notebook(self)
            self.nb.pack(fill="both", expand=True, padx=8, pady=8)
            # resumo
            fr = ttk.Frame(self.nb)
            self.nb.add(fr, text="  Resumo  ")
            rolo = ttk.Scrollbar(fr)
            rolo.pack(side="right", fill="y")
            rolo_x = ttk.Scrollbar(fr, orient="horizontal")
            rolo_x.pack(side="bottom", fill="x")
            self.caixa = tk.Text(fr, font=("Consolas", 10), wrap="none", yscrollcommand=rolo.set,
                                 xscrollcommand=rolo_x.set)
            self.caixa.pack(fill="both", expand=True)
            rolo.config(command=self.caixa.yview)
            rolo_x.config(command=self.caixa.xview)
            # gráficos
            self.figs = {}
            for nome in ("T(f)", "Profundidade vs N", "f vs 1/(2d)", "ΓX ↔ ΓM"):
                fr = ttk.Frame(self.nb)
                self.nb.add(fr, text=f"  {nome}  ")
                if nome == "T(f)":
                    cima = ttk.Frame(fr, padding=4)
                    cima.pack(fill="x")
                    ttk.Label(cima, text="a (mm):").pack(side="left")
                    self.v_a = tk.StringVar()
                    self.cb_a = ttk.Combobox(cima, textvariable=self.v_a, state="readonly", width=8)
                    self.cb_a.pack(side="left", padx=6)
                    self.cb_a.bind("<<ComboboxSelected>>", lambda e: self.desenhar_T())
                    ttk.Label(cima, text="As duas direções com os mesmos eixos (f em unidades de f_B). "
                                         "Ponteado: medição sem referência do mesmo dia.",
                              foreground="#555").pack(side="left", padx=10)
                self.figs[nome] = self._figura(fr)
            # uma medição
            fr = ttk.Frame(self.nb)
            self.nb.add(fr, text="  Uma medição  ")
            esq = ttk.Frame(fr)
            esq.pack(side="left", fill="y")
            self.lista = ttk.Treeview(esq, columns=("ref", "ond"), height=30, selectmode="browse")
            self.lista.heading("#0", text="medição")
            self.lista.column("#0", width=130)
            for c, t, w in (("ref", "referência", 95), ("ond", "ond. (dB)", 65)):
                self.lista.heading(c, text=t)
                self.lista.column(c, width=w, anchor="center")
            self.lista.tag_configure("fora", foreground="#999")
            rolo = ttk.Scrollbar(esq, command=self.lista.yview)
            self.lista.configure(yscrollcommand=rolo.set)
            rolo.pack(side="right", fill="y")
            self.lista.pack(fill="y", expand=True)
            self.lista.bind("<<TreeviewSelect>>", lambda e: self.desenhar_uma())
            dir_ = ttk.Frame(fr, padding=(8, 0))
            dir_.pack(side="left", fill="both", expand=True)
            self.lbl_uma = ttk.Label(dir_, font=("Segoe UI", 10), justify="left", wraplength=980)
            self.lbl_uma.pack(side="bottom", anchor="w", pady=6)
            self.figs["uma"] = self._figura(dir_)
            # teoria e método: um documento com a figura da teoria lá dentro
            fr = ttk.Frame(self.nb)
            self.nb.add(fr, text="  📖 Teoria e método  ")
            rolo = ttk.Scrollbar(fr)
            rolo.pack(side="right", fill="y")
            self.doc = tk.Text(fr, wrap="word", font=("Segoe UI", 11), padx=44, pady=18, relief="flat",
                               background="white", foreground="#0b0b0b", cursor="arrow",
                               yscrollcommand=rolo.set, spacing1=2, spacing3=4)
            self.doc.pack(fill="both", expand=True)
            rolo.config(command=self.doc.yview)
            for nome, opc in (
                    ("h1", dict(font=("Segoe UI", 19, "bold"), spacing3=6)),
                    ("h2", dict(font=("Segoe UI", 14, "bold"), spacing1=24, spacing3=8)),
                    ("h3", dict(font=("Segoe UI", 11, "bold"), spacing1=10, spacing3=2)),
                    ("p", dict(spacing3=8)),
                    ("item", dict(lmargin1=14, lmargin2=30, spacing3=5)),
                    ("formula", dict(font=("Cambria", 13), lmargin1=44, lmargin2=44,
                                     spacing1=8, spacing3=14, wrap="none")),
                    ("mono", dict(font=("Consolas", 10), lmargin1=14, lmargin2=14, wrap="none",
                                  spacing1=0, spacing3=0, background="#f4f4f1")),
                    ("nota", dict(font=("Segoe UI", 9), foreground="#52514e", spacing3=10))):
                self.doc.tag_configure(nome, **opc)

        def escrever_doc(self):
            """Enche o separador «Teoria e método» a partir de res['doc']."""
            res, t = self.res, self.doc
            t.config(state="normal")
            t.delete("1.0", "end")
            for estilo, txt in res["doc"]:
                if estilo == "figura":
                    if not res["teoria"]:
                        continue
                    fig = Figure(figsize=(11.6, 6.2), dpi=100, layout="constrained")
                    T.figura(fig, res["R"], res["teoria"])
                    w = FigureCanvasTkAgg(fig, master=t).get_tk_widget()
                    # a roda do rato por cima da figura continua a rolar o documento
                    w.bind("<MouseWheel>", lambda e: t.yview_scroll(-e.delta // 120 * 3, "units"))
                    t.window_create("end", window=w, pady=8)
                    t.insert("end", "\n")
                else:
                    t.insert("end", ("•  " if estilo == "item" else "") + txt + "\n", estilo)
            t.config(state="disabled")
            t.yview_moveto(0)

        def _figura(self, pai):
            fig = Figure(figsize=(9, 5), dpi=100, layout="constrained")
            cv = FigureCanvasTkAgg(fig, master=pai)
            tb = NavigationToolbar2Tk(cv, pai, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
            cv.get_tk_widget().pack(fill="both", expand=True)
            return fig

        # ------------------------------------------------------------ cálculo
        def recalcular(self):
            if self.ocupado:
                return
            falta = [c for c in self.cfg["conjuntos"] if not Path(c["pasta"]).is_dir()]
            if falta:
                messagebox.showwarning("Pastas de dados", "Não encontro a pasta de dados de:\n\n"
                                       + "\n".join(f"{c['nome']} ({c['dir']}): {c['pasta']}" for c in falta)
                                       + "\n\nEscolhe-a em ⚙ Pastas de dados.")
                self.pastas()
                return
            self.ocupado = True
            self.barra.pack(side="right")
            self.lbl_estado.config(text="A ler as gravações (a primeira vez demora ~1 min)…")
            threading.Thread(target=self._calcular, daemon=True).start()
            self.after(50, self._ouvir)

        def _calcular(self):
            # corre noutra thread: só fala com a janela pela fila (o Tk não é seguro entre threads)
            try:
                res = calcular(self.cfg, lambda k, n, txt: self.fila.put(("progresso", k, n, txt)))
                self.fila.put(("pronto", res, None))
            except Exception as e:      # mostrar o erro na janela em vez de a fechar
                self.fila.put(("pronto", None, e))

        def _ouvir(self):
            while not self.fila.empty():
                msg = self.fila.get()
                if msg[0] == "pronto":
                    self._pronto(msg[1], msg[2])
                    return
                _, k, n, txt = msg
                self.barra.config(maximum=max(n, 1), value=k)
                self.lbl_estado.config(text=f"A ler {txt}…" if txt else "A analisar…")
            self.after(50, self._ouvir)

        def _pronto(self, res, erro):
            self.ocupado = False
            self.barra.pack_forget()
            if erro:
                self.lbl_estado.config(text="Erro.")
                messagebox.showerror("Erro na análise", str(erro))
                return
            self.res = res
            R = res["R"]
            n_val = sum(r["valida"] for r in R["linhas"])
            self.lbl_estado.config(text=f"{n_val} medições nas regressões"
                                        + (f" · {len(R['linhas']) - n_val + len(res['fora'])} fora "
                                           f"(ver Resumo)" if len(R["linhas"]) - n_val + len(res["fora"]) else ""))
            self.caixa.delete("1.0", "end")
            self.caixa.insert("1.0", res["texto"])
            self.caixa.yview_moveto(0)
            As = valores_a(R)
            comuns = [c["a"] for c in R["comparar"]]
            self.cb_a.config(values=[f"{a:g}" for a in As])
            if self.v_a.get() not in [f"{a:g}" for a in As]:
                self.v_a.set(f"{(max(comuns) if comuns else As[0]):g}")
            self.desenhar_T()
            for nome, desenhar in (("Profundidade vs N", G.fig_prof), ("f vs 1/(2d)", G.fig_freq),
                                   ("ΓX ↔ ΓM", G.fig_comp)):
                desenhar(self.figs[nome], R)
                self.figs[nome].canvas.draw_idle()
            sel = self.lista.selection()
            self.lista.delete(*self.lista.get_children())
            for k, r in enumerate(R["linhas"]):
                self.lista.insert("", "end", iid=str(k), text=f"{r['dir']}  {r['id']}",
                                  values=(r["ref"], f"{r['ondulacao']:.1f}"),
                                  tags=() if r["valida"] else ("fora",))
            alvo = sel[0] if sel and self.lista.exists(sel[0]) else "0"
            self.lista.selection_set(alvo)
            self.escrever_doc()

        def mudar_modo(self):
            self.cfg["modo_ref"] = next(k for k, v in A.MODOS_REF.items() if v == self.v_modo.get())
            guardar_config(self.cfg)
            self.recalcular()

        # ------------------------------------------------------------ desenhar
        def desenhar_T(self):
            if self.res and self.v_a.get():
                G.fig_T(self.figs["T(f)"], self.res["R"], float(self.v_a.get()))
                self.figs["T(f)"].canvas.draw_idle()

        def desenhar_uma(self):
            sel = self.lista.selection()
            if not (self.res and sel):
                return
            r = self.res["R"]["linhas"][int(sel[0])]
            G.fig_uma(self.figs["uma"], r)
            self.figs["uma"].canvas.draw_idle()
            self.lbl_uma.config(text=G.texto_uma(r))

        # ------------------------------------------------------------ guardar / pastas
        def guardar(self):
            if not self.res:
                return
            try:
                pasta = guardar_tudo(self.res)
            except Exception as e:
                messagebox.showerror("Guardar", str(e))
                return
            messagebox.showinfo("Guardado", "Gráficos (PNG), resultados.txt, tabela.csv, ajustes.csv e "
                                            f"teoria_e_metodo.txt em\n{pasta}")
            abrir(pasta)

        def pastas(self):
            j = tk.Toplevel(self)
            j.title("Pastas de dados")
            j.transient(self)
            vs = []
            ttk.Label(j, text="Pasta «dados» de cada experiência (a que tem a32_N10, ref_a32, …).\n"
                              "A app só lê: não altera nem apaga nada nessas pastas.",
                      foreground="#555").grid(row=0, column=0, columnspan=3, padx=8, pady=8, sticky="w")
            for i, c in enumerate(self.cfg["conjuntos"], start=1):
                ttk.Label(j, text=f"{c['dir']} — {c['nome']}").grid(row=i, column=0, sticky="w", padx=8, pady=4)
                v = tk.StringVar(value=c["pasta"])
                ttk.Entry(j, textvariable=v, width=90).grid(row=i, column=1, padx=4)
                ttk.Button(j, text="Escolher…", command=lambda v=v: v.set(
                    filedialog.askdirectory(parent=j, initialdir=v.get() or str(AQUI)) or v.get())
                           ).grid(row=i, column=2, padx=8)
                vs.append(v)

            def ok():
                for c, v in zip(self.cfg["conjuntos"], vs):
                    c["pasta"] = v.get()
                guardar_config(self.cfg)
                j.destroy()
                self.recalcular()

            ttk.Button(j, text="OK", command=ok).grid(row=len(vs) + 1, column=2, padx=8, pady=8, sticky="e")

    return App()


if __name__ == "__main__":
    if "--sem-janela" in sys.argv:
        sem_janela()
    else:
        criar_app().mainloop()
