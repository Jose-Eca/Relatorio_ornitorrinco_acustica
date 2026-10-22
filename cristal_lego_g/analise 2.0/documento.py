"""
documento.py — o texto do separador «Teoria e método»: a teoria, como se mede, como se analisa
e de onde vêm as incertezas. Os parâmetros (janelas, limiar, …) e os números (c, tabela da
teoria contra a medição) são lidos de analise.py e dos resultados, por isso o texto acompanha
sempre o que a app está de facto a fazer.

documento(res) devolve uma lista de (estilo, texto), com estilo em:
  "h1", "h2", "h3", "p", "item", "formula", "mono" (tabela de largura fixa), "nota", "figura".
"""
import numpy as np

import analise as A
import teoria as T


def documento(res):
    R, C = res["R"], res.get("teoria")
    d = []

    def h2(t): d.append(("h2", t))
    def h3(t): d.append(("h3", t))
    def p(t): d.append(("p", t))
    def item(t): d.append(("item", t))
    def formula(t): d.append(("formula", t))

    cs = [R[x]["c"] for x in R["dirs"] if np.isfinite(R[x]["c"])]
    c, temp = float(np.mean(cs)), float(np.mean([R[x]["temp"] for x in R["dirs"] if np.isfinite(R[x]["c"])]))
    df = A.FS / A.NPERSEG
    As = sorted({r["a_mm"] for r in R["linhas"]})

    d.append(("h1", "Cristal sónico de LEGO: teoria, medição e análise"))
    d.append(("nota", "Os parâmetros e os números deste texto são os que a app está a usar agora; "
                      "mudam se os dados ou as constantes de analise.py mudarem."))

    # ------------------------------------------------------------------ 1
    h2("1. A experiência")
    p("Um cristal sónico é uma rede periódica de obstáculos rígidos no ar. Aqui os obstáculos são "
      f"colunas de tijolos LEGO 2×2 (lado s = {T.LADO_MM} mm) numa rede quadrada de passo a. Os pinos "
      f"da placa estão a 8 mm, por isso a é múltiplo de 8: {', '.join(f'{a:g}' for a in As)} mm. "
      "De um lado está uma coluna de som, do outro um microfone, e mede-se quanto som atravessa o "
      "cristal em cada frequência.")
    p("Mede-se em duas direções. O que as distingue é a distância d entre planos de tijolos "
      "perpendiculares ao som:")
    item("ΓX (perpendicular): o som viaja ao longo de um eixo da rede. Os planos estão a d = a e, "
         "dentro de cada plano, os tijolos estão a L = a uns dos outros.")
    item("ΓM (diagonal): o som viaja ao longo da diagonal. Os planos estão mais juntos, d = a/√2, e "
         "dentro de cada plano os tijolos estão mais afastados, L = √2·a.")

    # ------------------------------------------------------------------ 2
    h2("2. O que a teoria prevê")
    h3("Hiato de Bragg")
    p("Numa rede periódica, as ondas refletidas em planos sucessivos somam-se em fase quando o "
      "comprimento de onda é o dobro da distância entre planos. À volta dessa frequência não há "
      "ondas que se propaguem no cristal: é o hiato (band gap).")
    formula("λ = 2d   ⟹   f_B = c / (2d)          ΓX: f_B = c / (2a)          ΓM: f_B = c / (√2·a)")
    p("No ar uniforme, em k = π/d há duas ondas estacionárias com a mesma frequência f_B: "
      "p₁ ∝ cos(πx/d), com máximos de pressão nos planos de tijolos, e p₂ ∝ sin(πx/d), com máximos de "
      "pressão no ar entre planos. Com tijolos, deixam de ter a mesma frequência. A que tem a pressão "
      "máxima em cima dos tijolos sobe (o tijolo não se comprime e sobra menos ar para comprimir): é o "
      "bordo de cima, f_sup. A que tem a velocidade do ar máxima nos planos de tijolos desce (o ar tem "
      "de passar a correr pelas frestas, como se tivesse mais massa): é o bordo de baixo, f_inf. O "
      "hiato é o intervalo entre as duas; f_B fica lá dentro.")
    formula("f_inf  <  f_B  <  f_sup          (sem tijolos: f_inf = f_sup = f_B)")
    p("Para o mesmo a, o Bragg puro põe o hiato da diagonal √2 vezes mais acima. A velocidade do som vem "
      f"da temperatura da sala: c = 331.3·√(1 + T/273.15) = {c:.1f} m/s a {temp:.1f} °C.")
    h3("Bandas")
    p("f_B é uma estimativa de rede vazia. Onde o hiato fica e que largura tem depende da fração "
      "de enchimento, ff = s²/a²: quanto mais cheio, mais largo o hiato e mais abaixo de f_B começa. "
      "Por isso a frequência do hiato não é exatamente proporcional a 1/a. A app calcula as bandas "
      "ω(k) de um cristal 2D infinito de hastes quadradas rígidas, resolvendo a equação de onda com "
      f"a pressão escrita como soma de ondas planas ({(2 * T.N_ONDAS + 1) ** 2} ondas):")
    formula("∇·(ρ⁻¹ ∇p) + ω² p / (ρc²) = 0          p(r) = Σ_G  p_G · exp[ i (k + G)·r ]")
    p("Substituindo, cada coeficiente p_G fica ligado aos outros e, para cada k, sai um problema de "
      "valores próprios; as soluções ω₁(k), ω₂(k), … são as bandas:")
    formula("Σ_G'  σ(G−G') (k+G)·(k+G') p_G'  =  ω² Σ_G'  η(G−G') p_G'          σ = 1/ρ,  η = 1/(ρc²)")
    p("σ(G) e η(G) são os coeficientes de Fourier do cristal: para uma haste quadrada de lado s numa "
      "célula de lado a, (valor no tijolo − valor no ar)·ff·sinc(G_x s/2)·sinc(G_y s/2). Ficando só com as "
      "duas ondas e^(±iπx/d), a matriz é 2×2 e as soluções são o cos e o sin de cima; a diferença das "
      "frequências é proporcional ao coeficiente de Fourier em G = 2π/d. Sem tijolos é zero e as duas "
      "coincidem em f_B.")
    p("O cristal também torna o som mais lento em média (o ar tem de contornar os tijolos), e isso "
      "desce o hiato inteiro. Para hastes rígidas, a baixa frequência:")
    formula("c_ef ≈ c / √(1 + ff)     ⟹     f_hiato ≈ f_B / √(1 + ff)")
    p("Com ff = 0.24 (a = 32 mm) isso é 0.90 f_B; com ff = 0.08 (a = 56 mm), 0.96 f_B. A teoria de "
      "bandas dá o centro do hiato de ΓX em 0.925 e 0.968 f_B.")
    p("O hiato teórico vai do topo da 1.ª banda ao fundo da 2.ª, ao longo da direção do som. Só "
      "contam os modos simétricos em relação ao plano que contém essa direção: os antissimétricos "
      "(«bandas surdas») não são excitados por uma onda que chega de frente.")
    h3("Atenuação")
    p("Dentro do hiato o número de onda é complexo e a pressão decai exponencialmente ao longo do "
      "cristal, p ∝ exp(−κx). Em decibéis isso é uma reta que passa na origem:")
    formula("D(N) = α·N          α = 20·log10(e)·κ·d     (dB por fila)")
    h3("Difração lateral")
    p("Uma fila de tijolos com período L é também uma rede de difração: a partir de f = c/L parte "
      "do som sai desviada para os lados e deixa de chegar ao microfone, que está no eixo. Em ΓX, "
      "L = a e isso só acontece em 2·f_B, longe do hiato. Em ΓM, L = √2·a e acontece exatamente em "
      "f_B. É a razão provável para, na diagonal, T não voltar a subir depois do hiato.")
    h3("Hiato completo")
    p("Há hiato para todas as direções do plano se os de ΓX e ΓM se sobrepuserem, isto é, se "
      "f_sup(ΓX) > f_inf(ΓM). Como os centros diferem de √2, só acontece com hiatos largos (ff alto). "
      "Pela teoria de bandas, com tijolos 2×2 não acontece: f_sup(ΓX) − f_inf(ΓM) = −143, −357, −505 Hz "
      "para a = 32, 40, 48 mm (e −7 Hz para a = 24 mm). O hiato de ΓM é estreito porque, ao longo da "
      "diagonal, a diagonal do tijolo (22.3 mm) ocupa quase toda a distância entre planos (a/√2 = 22.6 mm "
      "para a = 32 mm): o meio parece quase uniforme nessa direção e cada plano reflete pouco. Por isso a "
      "razão esperada f_inf(ΓM)/f_inf(ΓX) não é √2: é 1.59, 1.51 e 1.47 para a = 32, 40, 48 mm.")

    # ------------------------------------------------------------------ 3
    h2("3. Como se mede")
    item("A coluna emite ruído branco durante 10 s (todas as frequências ao mesmo tempo) e o "
         f"microfone grava a {A.FS // 1000} kHz. Cada passo são 5 gravações seguidas.")
    item("Referência: a mesma gravação sem tijolos, com a coluna e o microfone nos mesmos sítios. "
         "Dividir pela referência tira da conta a resposta da coluna, do microfone e da sala.")
    item("Um bloco é um valor de a: referência, cristal com o máximo de filas, e depois tira-se uma "
         "fila de cada vez do lado do microfone, até N = 1. Na diagonal o bloco acaba com uma "
         "segunda referência, que mede o quanto a montagem mudou entretanto.")
    item(f"A coluna é Bluetooth e atrasa o som cerca de 0.55 s: de cada gravação só se usa o troço "
         f"{A.T_INI}–{A.T_FIM} s.")

    # ------------------------------------------------------------------ 4
    h2("4. Da gravação a T(f)")
    item(f"Espectro de potência de cada gravação pelo método de Welch (janela de Hann, {A.NPERSEG} "
         f"pontos, 50 % de sobreposição; Δf = {df:.1f} Hz), entre {A.FMIN} Hz e {A.FMAX // 1000} kHz.")
    item("Repetições com saturação ou com cortes do Bluetooth são postas de parte.")
    item("Os espectros são suavizados em potência numa janela de 1/6 de oitava antes de dividir, "
         "para um buraco estreito da referência não virar um pico falso.")
    formula("T(f) = 10·log10 [ P_cristal(f) / P_ref(f) ]     (dB)")

    # ------------------------------------------------------------------ 5
    h2("5. As grandezas que se tiram de T(f)")
    p("As janelas do hiato são definidas em unidades de f_B. É isso que torna a análise igual nas duas "
      "direções: muda d, não mudam as contas. A exceção é a banda de controlo, que tem de ficar onde "
      "o cristal é transparente em todas as direções.")
    item(f"Nível base T0: mediana de T na banda de controlo, {A.BASE[0]}–{A.BASE[1]} × c/(2a), nas duas "
         "direções. Fica por baixo do 1.º hiato da rede em qualquer direção (o de ΓX, perto de c/(2a)): "
         "na diagonal, uma banda em unidades de f_B = c/(√2·a) iria até 0.99 × c/(2a) e apanharia o "
         "hiato de ΓX, que o som que chega de lado também vê. Absorve mudanças de volume entre medições.")
    item("Ondulação: desvio-padrão robusto (1.4826 × desvio absoluto mediano) de T na banda de "
         "controlo. Aí T devia ser plano; o que sobra é o «ruído» de T(f) que as repetições não veem "
         "(reflexões na sala, referência que não acerta).")
    item(f"Profundidade D = T0 − média de T em {A.JAN_PROF[0]}–{A.JAN_PROF[1]} × f_B.")
    item(f"Hiato: zona contínua com T < T0 − {A.LIMIAR_DB} dB que toca em {A.PROCURA[0]}–{A.PROCURA[1]} "
         "× f_B. Os bordos f_inf e f_sup são os cruzamentos do limiar, interpolados entre pontos.")
    item(f"O hiato «fecha» se T voltar a subir acima do limiar antes de {A.FECHO_MAX} × f_B. Só então "
         "há f_sup, f_centro = (f_inf + f_sup)/2 e largura = f_sup − f_inf.")
    item("A frequência principal é f_inf, porque é a única que existe sempre nas duas direções.")
    p(f"De cada bloco (um a) fica um valor de cada frequência: a média ponderada dos N ≥ {A.N_MIN_F} "
      "(com menos filas o hiato mal existe). Um bloco só tem f_sup, f_centro e largura se o hiato "
      "fechar em mais de metade das medições.")

    # ------------------------------------------------------------------ 6
    h2("6. Que referência se usa")
    p("A referência é a montagem sem tijolos, por isso não depende de a: são candidatas todas as "
      "referências da mesma sessão (mesma pasta de dados, mesmo dia). Qual se usa escolhe-se na "
      "caixa «Referência»:")
    for k, v in A.MODOS_REF.items():
        item(("▶ " if k == res["modo"] else "") + v + ("   (em uso)" if k == res["modo"] else ""))
    p("Entre candidatas, fica a que deixa a banda de controlo mais plana (menor mediana da "
      "ondulação nas medições do bloco). As outras entram na incerteza σ_ref. Trocar de modo é um "
      "teste de robustez: os resultados não devem mudar mais do que as incertezas.")
    item("Medição sem referência do mesmo dia: fica na tabela, mas não entra em nenhum ajuste.")
    item(f"Medição «com cristal» igual a uma referência (menos de {A.IGUAL_REF_DB} dB de desvio em toda "
         "a banda): foi gravada sem tijolos e é posta de parte.")
    fora = [f"{m['conj']} / {m['id']} (sem tijolos)" for m, _ in res["fora"]] \
        + [f"{r['conj']} / {r['id']} (sem referência do dia)" for r in R["linhas"] if not r["valida"]]
    if fora:
        p("Fora das regressões com os dados de agora: " + ", ".join(fora) + ".")

    # ------------------------------------------------------------------ 7
    h2("7. Incertezas")
    h3("De cada número")
    formula("σ² = σ_estat² + σ_ond² + σ_ref²     (+ σ_res² nas frequências)")
    item("σ_estat: desvio-padrão entre repetições / √n. É quase sempre desprezável, porque as 5 "
         "repetições são a mesma montagem e só veem o ruído da gravação.")
    item("σ_ond: a ondulação propagada para o número. Na profundidade, ondulação × "
         f"√(π/2·1/n_base + 1/n_jan) = {A.K_OND_PROF:.2f} × ondulação, com n = largura da janela / largura "
         "da suavização (número de pontos independentes; π/2 porque T0 é uma mediana). Nos bordos, "
         "ondulação / |dT/df| no cruzamento: um bordo pouco inclinado tem a frequência mal definida.")
    item("σ_ref: dispersão do número calculado com cada referência candidata, em torno do valor "
         "central, com pesos 1/ondulação² (uma referência má pesa pouco). Com uma só candidata "
         "não se pode estimar.")
    item(f"σ_res = Δf/√12 = {df / np.sqrt(12):.1f} Hz: resolução do espectro.")
    h3("De cada bloco")
    formula("σ² = σ_disp² + σ_sis²")
    item("σ_disp: erro da média, tirado da dispersão observada entre os N.")
    item("σ_sis: média das partes σ_ond e σ_ref. A referência e a montagem são as mesmas para todos "
         "os N do bloco, por isso esta parte não diminui ao fazer a média.")
    h3("Fixas")
    item(f"σ_a = {A.SIGMA_A_MM} mm (posição dos tijolos nos pinos);  σ_T = {A.SIGMA_TEMP:g} °C, que dá "
         f"σ_c = {A.sigma_velocidade_som(temp):.1f} m/s.")

    # ------------------------------------------------------------------ 8
    h2("8. Regressões")
    p("Todos os ajustes são mínimos quadrados ponderados: cada ponto pesa 1/σ², e os parâmetros "
      "são os que minimizam")
    formula("χ² = Σ (y_i − m·x_i − b)² / σ_i²          ν = n.º de pontos − n.º de parâmetros")
    item("As incertezas dos parâmetros saem da matriz de covariância do ajuste.")
    item("A incerteza de a entra por variância efetiva: σ_i² = σ_y² + (m·σ_x)².")
    item("χ²/ν ≈ 1 quer dizer que a reta descreve os pontos dentro das barras de erro. O valor-p é "
         "a probabilidade de obter um χ² tão grande ou maior se o modelo estiver certo.")
    item("Se χ²/ν > 1, as incertezas dos parâmetros são multiplicadas por S = √(χ²/ν), e o texto "
         "diz quando. Se χ²/ν < 1 não se mexe: nunca se encolhem incertezas.")
    item("Nos gráficos, a faixa à volta da reta é ±1σ do ajuste e o painel de baixo mostra os "
         "resíduos divididos por σ.")
    p("Fazem-se três coisas, iguais para cada direção:")
    item("Profundidade contra N:  D = α·N + b.  α é a atenuação por fila; b deve ser compatível com "
         "zero, porque sem filas não há atenuação.")
    item("Frequência contra 1/(2d):  f = m·x + b, com x = 1/(2d), para f_inf e para f_centro. O "
         "declive é uma velocidade; se f fosse exatamente f_B dava c. Faz-se a reta livre (pelo "
         "menos 3 valores de a) e a reta pela origem (pelo menos 2).")
    item("Para o mesmo a nas duas direções:  f_inf(ΓM)/f_inf(ΓX), que se compara com √2, e "
         "f_sup(ΓX) − f_inf(ΓM), que é positivo se os hiatos se sobrepuserem.")
    p("Um ajuste com valor-p abaixo de 0.05 não é uma medição falhada: quer dizer que a reta é "
      "simples demais ou que as barras de erro são pequenas demais. Um χ²/ν muito abaixo de 1 quer "
      "dizer barras largas ou correlacionadas. Na profundidade contra N é o caso: a parte da "
      "referência é comum a todos os N, o que desloca a reta toda mas não muda o declive.")

    # ------------------------------------------------------------------ 9
    h2("9. A teoria contra as medições")
    if C:
        p("Para cada bloco: os bordos do hiato e a atenuação por fila calculados pelas bandas, ao "
          "lado do que se mediu. z é o número de desvios-padrão entre os dois; abaixo de 2 são "
          "compatíveis.")
        d.append(("mono", T.tabela(C)))
        p(f"A incerteza da teoria junta o lado do tijolo (± {T.S_LADO_MM} mm) e o erro de truncatura "
          f"da expansão em ondas planas ({100 * T.S_TRUNC:.1f} %, tirado de comparar 225 com 729 ondas). "
          "A atenuação teórica é o máximo dentro do hiato; a medida é a média em f_B ± 10 %, que na "
          "diagonal só apanha parte do hiato, por isso essa coluna compara ordens de grandeza.")
        p("Duas previsões que não são as do Bragg puro: (1) o hiato fica abaixo de f_B nas duas "
          "direções, porque os tijolos rígidos tornam o som mais lento dentro do cristal; em ΓM o "
          "hiato teórico fica quase todo abaixo de 0.9 f_B; (2) por isso a razão f_inf(ΓM)/f_inf(ΓX) "
          "esperada é maior que √2 (≈ 1.47–1.59 para a = 48–32 mm), e com estas frações de "
          "enchimento o cristal infinito NÃO tem hiato completo: f_sup(ΓX) fica abaixo de f_inf(ΓM).")
        d.append(("figura", "teoria"))
        d.append(("nota", "Curvas: T − T0 medido para N ≥ %d, em cima ΓX e em baixo ΓM. Uma curva a "
                          "ponteado é de um bloco sem referência do mesmo dia (provisório)." % A.N_MIN_F))
    else:
        p("Não foi possível calcular a teoria: " + str(res.get("erro_teoria", "")))

    # ------------------------------------------------------------------ 10
    h2("10. O que o modelo e as medições não apanham")
    item("O modelo é um cristal 2D infinito atingido por uma onda plana. O cristal real tem 3 ou 4 "
         "tijolos de largura, altura finita e uma fonte próxima: há som que passa à volta e por "
         "cima, e isso limita a profundidade que se consegue medir.")
    item("O modelo dá os bordos das bandas. A medição dá o sítio onde T cruza um limiar de "
         f"{A.LIMIAR_DB} dB, que só coincide com o bordo quando o hiato está bem formado (N grande).")
    item("O modelo não inclui a difração lateral: prevê que T volte a subir depois do hiato também "
         "na diagonal. A transmissão de um cristal finito não foi calculada.")
    item("A fonte está perto e o cristal é estreito: chega ao microfone som que atravessou o cristal "
         "na diagonal da rede, isto é, perto de ΓX. É a explicação provável para, em ΓM, T começar a "
         "descer já na zona do hiato de ΓX (≈ 0.5–0.8 × f_B de ΓM, mais visível em a = 32 mm com N "
         "grande) e para o f_inf medido em ΓM ficar 8–11 % abaixo do teórico. Pela mesma razão, a "
         "«sobreposição» medida dos hiatos não prova um hiato completo.")
    item("A referência é o ponto fraco das medições: mexer na coluna, no microfone ou no volume "
         "entre a referência e o cristal muda T(f) em vários dB. É o que domina as incertezas.")
    return d


def texto_simples(doc):
    """O documento em texto corrido (para guardar em ficheiro)."""
    out = []
    for estilo, t in doc:
        if estilo == "h1":
            out += [t.upper(), "=" * len(t), ""]
        elif estilo == "h2":
            out += ["", "", t, "-" * len(t)]
        elif estilo == "h3":
            out += ["", t + ":"]
        elif estilo == "item":
            out.append("  • " + t)
        elif estilo == "formula":
            out += ["", "        " + t, ""]
        elif estilo == "mono":
            out += ["", t, ""]
        elif estilo == "figura":
            out.append("  [figura: teoria_vs_medido.png]")
        else:
            out.append(t)
    return "\n".join(out) + "\n"
