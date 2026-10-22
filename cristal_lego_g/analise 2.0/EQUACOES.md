# Cristal sónico de LEGO: todas as equações da análise

Este ficheiro junta, por ordem, todas as equações que a análise 2.0 usa: a física do cristal, o caminho da gravação até T(f), as grandezas medidas, as incertezas e os ajustes. Para cada uma diz-se de onde vem e como se lê. Os números de exemplo são os da corrida de 14/out (pasta `resultados/`).

Notação:

| símbolo | significado |
|---|---|
| a | distância entre centros de tijolos (32, 40, 48, 56 mm) |
| s | lado do tijolo 2×2 (15,8 mm) |
| d | distância entre planos de tijolos perpendiculares ao som: d = a (ΓX), d = a/√2 (ΓM) |
| L | período dos tijolos dentro de um plano: L = a (ΓX), L = √2·a (ΓM) |
| N | número de filas |
| ff | fração de enchimento, s²/a² |

---

## 1. Física do cristal

### 1.1 Velocidade do som

$$c(T) = 331{,}3\,\sqrt{1 + \frac{T}{273{,}15}}\ \text{m/s}, \qquad \sigma_c = \frac{c}{2\,(273{,}15 + T)}\,\sigma_T$$

Com σ_T = 1 °C, σ_c = 0,6 m/s. A segunda fórmula é a derivada dc/dT vezes σ_T.

### 1.2 Bragg

Cada plano de tijolos reflete um pouco de som. A onda refletida no plano seguinte percorre mais 2d (ida e volta). As reflexões somam-se em fase quando 2d = λ:

$$\lambda = 2d \quad\Longrightarrow\quad f_B = \frac{c}{2d}, \qquad f_B^{\Gamma X} = \frac{c}{2a}, \qquad f_B^{\Gamma M} = \frac{c}{\sqrt{2}\,a}$$

O mesmo resultado em número de onda: $k = \pi/d$, que é o bordo da zona de Brillouin na direção do som.

### 1.3 Porque é que o hiato é um intervalo e não um ponto

Em $k = \pi/d$ há duas ondas estacionárias com o mesmo comprimento de onda:

$$p_1 \propto \cos\!\left(\frac{\pi x}{d}\right)\ \text{(pressão máxima nos planos de tijolos)}, \qquad p_2 \propto \sin\!\left(\frac{\pi x}{d}\right)\ \text{(pressão máxima no ar entre planos)}$$

- **Sem tijolos**, as duas têm a mesma frequência, $f_B$.
- **Com tijolos**, separam-se:
  - A onda com **pressão máxima nos tijolos** sobe. O tijolo não se comprime e sobra menos ar para comprimir, por isso o meio fica mais rígido. É o **bordo de cima, f_sup**.
  - A onda com **velocidade do ar máxima nos planos de tijolos** (é onde a sua pressão é zero) desce. O ar tem de passar a correr pelas frestas, o que é como ter mais massa. É o **bordo de baixo, f_inf**.

$$f_\text{inf} < f_B < f_\text{sup} \qquad (\text{sem tijolos: } f_\text{inf} = f_\text{sup} = f_B)$$

Entre f_inf e f_sup não há nenhuma onda com k real. Uma onda que entre nessa faixa decai, e é isso o hiato.

### 1.4 Equação de onda num meio periódico

Com densidade ρ(r) e velocidade c(r) que mudam de ponto para ponto (ar ou tijolo):

$$\nabla\cdot\!\left(\frac{1}{\rho}\,\nabla p\right) + \frac{\omega^2}{\rho c^2}\,p = 0$$

Com ρ e c constantes, isto reduz-se a $\nabla^2 p + (\omega/c)^2 p = 0$.

**Teorema de Bloch.** Num meio periódico, a solução é uma onda plana vezes uma função com o período da rede:

$$p(\mathbf r) = e^{i\mathbf k\cdot\mathbf r}\,u(\mathbf r), \qquad u(\mathbf r) = \sum_{\mathbf G} p_{\mathbf G}\, e^{i\mathbf G\cdot\mathbf r}, \qquad \mathbf G = \frac{2\pi}{a}(m, n)$$

### 1.5 Expansão em ondas planas (`teoria.py`)

Escreve-se também $\sigma = 1/\rho$ e $\eta = 1/(\rho c^2)$ em séries de Fourier. Para uma haste quadrada de lado s numa célula de lado a, com $v$ o valor da grandeza e $\operatorname{sinc}(x) = \sin x / x$:

$$\sigma(\mathbf G) = (v_\text{tijolo} - v_\text{ar})\; ff\; \operatorname{sinc}\!\left(\frac{G_x s}{2}\right)\operatorname{sinc}\!\left(\frac{G_y s}{2}\right), \qquad \sigma(\mathbf 0) = v_\text{ar} + (v_\text{tijolo} - v_\text{ar})\,ff$$

Ao substituir na equação de onda, para cada k sai um problema de valores próprios generalizado:

$$\sum_{\mathbf G'} \sigma(\mathbf G - \mathbf G')\,(\mathbf k + \mathbf G)\cdot(\mathbf k + \mathbf G')\; p_{\mathbf G'} \;=\; \omega^2 \sum_{\mathbf G'} \eta(\mathbf G - \mathbf G')\; p_{\mathbf G'}$$

As soluções $\omega_1(\mathbf k) \le \omega_2(\mathbf k) \le \dots$ são as **bandas**. A app:

- usa $(2\cdot 7 + 1)^2 = 225$ ondas planas;
- percorre k de Γ até X, $|\mathbf k| = \pi/a$, e de Γ até M, $|\mathbf k| = \sqrt 2\,\pi/a = \pi/d$;
- só usa as bandas **simétricas** em relação ao plano que contém a direção do som (y → −y em ΓX, x ↔ y em ΓM). As antissimétricas («bandas surdas») não são excitadas por uma onda que chega de frente;
- define o hiato como

$$f_\text{inf} = \max_k f_1(k), \qquad f_\text{sup} = \min_k f_{n\ge 2}(k)$$

**Erro de truncatura.** Com 729 ondas em vez de 225, os bordos mudam 0,1–0,4 %. Entra na incerteza teórica como 0,5 %:

$$\sigma_\text{teo} = \sqrt{\big(f(s + 0{,}1\ \text{mm}) - f(s)\big)^2 + (0{,}005\,f)^2}$$

### 1.6 Modelo de duas ondas (para perceber, não para calcular)

Se na expansão se ficar só com $e^{+i\pi x/d}$ e $e^{-i\pi x/d}$, a matriz é 2×2. As soluções são exatamente o cos e o sin de 1.3, e a separação das duas frequências é proporcional ao coeficiente de Fourier do cristal em $G = 2\pi/d$:

$$f_\text{sup} - f_\text{inf} \;\propto\; \big|\,\sigma(2\pi/d)\,\big|$$

Daqui se lê:
- **mais tijolo por plano** dá um coeficiente maior e um **hiato mais largo**;
- **sem tijolos** o coeficiente é zero e as duas frequências **coincidem em f_B**.

### 1.7 Fração de enchimento e som mais lento

$$ff = \frac{s^2}{a^2} \qquad (0{,}24;\ 0{,}16;\ 0{,}11;\ 0{,}08 \text{ para } a = 32, 40, 48, 56\ \text{mm})$$

A baixa frequência, o cristal comporta-se como um meio uniforme com:
- menos ar para comprimir: $\kappa_\text{ef} = (1-ff)\,\kappa$;
- mais inércia, porque o ar contorna os tijolos: $\rho_\text{ef} \approx \rho\,(1+ff)/(1-ff)$, a aproximação para hastes rígidas.

Daí:

$$c_\text{ef}^2 = \frac{1}{\rho_\text{ef}\,\kappa_\text{ef}} = \frac{c^2}{1 + ff} \qquad\Longrightarrow\qquad f_\text{hiato} \approx \frac{f_B}{\sqrt{1 + ff}}$$

| a (mm) | ff | $1/\sqrt{1+ff}$ | centro do hiato ΓX (teoria de bandas) / f_B |
|---|---|---|---|
| 32 | 0,24 | 0,897 | 0,925 |
| 40 | 0,16 | 0,930 | 0,948 |
| 48 | 0,11 | 0,950 | 0,960 |
| 56 | 0,08 | 0,962 | 0,968 |

**Consequência:** como o fator depende de a, f não é proporcional a 1/a, e a reta f contra 1/(2d) pela origem não descreve os pontos.

### 1.8 Atenuação dentro do hiato

Dentro do hiato o número de onda é complexo, $k = \pi/d + i\kappa$, e a pressão decai como $e^{-\kappa x}$. Ao fim de N filas, isto é $x = N d$:

$$D(N) = 20\log_{10}\!\left(e^{\kappa N d}\right) = \alpha N, \qquad \alpha = 20\log_{10}(e)\,\kappa d \approx 8{,}69\,\kappa d \quad \text{(dB por fila)}$$

A app resolve o problema ao contrário (dá-se ω, sai k). Fica um problema quadrático em k, que se transforma num linear de tamanho dobro:

$$\big(k^2 M_2 + k\,M_1 + M_0(\omega)\big)\,\mathbf p = 0$$

κ é o menor $|\operatorname{Im} k|$ dentro da 1.ª zona.

### 1.9 Difração lateral

Uma fila de tijolos com período L ao longo da fila é uma rede de difração:

$$\sin\theta = \frac{\lambda}{L} \quad\Longrightarrow\quad \text{1.ª ordem só existe se } \lambda < L, \text{ i.e. } f > \frac{c}{L}$$

| direção | L | começo da difração | posição relativa |
|---|---|---|---|
| ΓX | a | c/a | 2 f_B, longe do hiato |
| ΓM | √2·a | c/(√2·a) | f_B, mesmo no hiato |

Acima de c/L parte do som sai em ângulo e não chega ao microfone, que está no eixo. Em ΓM, T não volta a subir depois do hiato: o hiato fica «aberto».

### 1.10 Hiato completo e razão ΓM/ΓX

Há hiato para todas as direções se os hiatos de ΓX e ΓM se sobrepuserem:

$$f_\text{sup}(\Gamma X) > f_\text{inf}(\Gamma M)$$

No Bragg puro, $f_\text{inf}(\Gamma M)/f_\text{inf}(\Gamma X) = \sqrt 2$. A teoria de bandas dá outra coisa, porque o hiato de ΓX é muito mais largo que o de ΓM. Ao longo da diagonal, a diagonal do tijolo (22,3 mm) ocupa quase toda a distância entre planos, que para a = 32 mm é 22,6 mm. O meio parece quase uniforme nessa direção.

| a (mm) | f_inf(ΓM)/f_inf(ΓX), teoria | f_sup(ΓX) − f_inf(ΓM), teoria |
|---|---|---|
| 24 | — | −7 Hz |
| 32 | 1,59 | −143 Hz |
| 40 | 1,51 | −357 Hz |
| 48 | 1,47 | −505 Hz |

Valores negativos querem dizer que **não há hiato completo** com tijolos 2×2.

---

## 2. Da gravação a T(f)

### 2.1 Troço útil e espectro de Welch

Usa-se o troço de 0,8 s a 10,3 s, porque o Bluetooth atrasa ~0,55 s. O espectro calcula-se pelo método de Welch:

1. O troço é cortado em segmentos $x_m$ de $N_s = 4096$ amostras, com 50 % de sobreposição.
2. Cada segmento é multiplicado pela janela de Hann $w(n) = \sin^2(\pi n / N_s)$.
3. Faz-se a média de $|\mathrm{FFT}|^2$ sobre os M segmentos:

$$P(f_j) = \frac{2}{M}\sum_{m=1}^{M} \frac{\big|\sum_n w(n)\,x_m(n)\,e^{-2\pi i j n/N_s}\big|^2}{f_s\sum_n w(n)^2}, \qquad \Delta f = \frac{f_s}{N_s} = \frac{48\,000}{4096} = 11{,}7\ \text{Hz}$$

Analisa-se a banda 0,5–12 kHz. A constante de normalização, o 2 de espectro de um só lado e o denominador, cancela-se em T(f).

### 2.2 Suavização em 1/6 de oitava

Para cada f, faz-se a média da **potência** de todos os pontos entre $f\cdot 2^{-1/12}$ e $f\cdot 2^{+1/12}$, que são ±6 % (≈ 40 pontos a 4 kHz):

$$\bar P(f) = \frac{1}{n_f}\sum_{f\,2^{-1/12}\ \le\ f_j\ \le\ f\,2^{1/12}} P(f_j)$$

Faz-se ao espectro com cristal e ao da referência, **antes** de dividir. Assim um buraco estreito da referência não vira um pico falso em T.

### 2.3 Transmissão

$$T(f) = 10\log_{10}\frac{\bar P_\text{cristal}(f)}{\bar P_\text{ref}(f)}\quad(\text{dB})$$

Porque é que dividir funciona: $P_\text{cristal} \approx S\cdot H_\text{sala}\cdot M\cdot |t|^2$ e $P_\text{ref} \approx S\cdot H_\text{sala}\cdot M$, com S a coluna, $H_\text{sala}$ a sala e M o microfone. Na divisão, coluna, sala e microfone cancelam-se.

---

## 3. Grandezas tiradas de T(f)

### 3.1 Nível base e ondulação (banda de controlo)

A banda de controlo é **0,35–0,70 × c/(2a), nas duas direções**. Tem de ficar abaixo do 1.º hiato da rede em qualquer direção, que é o de ΓX.

> **Novo na 2.0.** A análise antiga usava 0,35–0,70 × f_B. Em ΓM isso ia até 0,99 × c/(2a), dentro do hiato de ΓX, que o som oblíquo também vê. Puxava T0 para baixo.

$$T_0 = \operatorname{mediana}\{T(f)\}_{\text{banda}}, \qquad \text{ondulação} = 1{,}4826\cdot\operatorname{mediana}\big|T - \operatorname{mediana}(T)\big|$$

O fator 1,4826 faz com que, para ruído gaussiano, a ondulação seja igual ao desvio-padrão. A mediana não se deixa puxar por picos isolados.

### 3.2 Profundidade

$$D = T_0 - \big\langle T(f) \big\rangle_{0{,}9 f_B\ \le\ f\ \le\ 1{,}1 f_B}$$

### 3.3 Bordos do hiato

O hiato é a zona contínua com $T < T_0 - 6$ dB que toca em $0{,}75$–$1{,}25\,f_B$. Os bordos são os cruzamentos do limiar, por interpolação linear entre os pontos $f_i$, $f_{i+1}$ que o rodeiam:

$$f_\text{cruz} = f_i + \frac{(T_0 - 6) - T_i}{T_{i+1} - T_i}\,(f_{i+1} - f_i)$$

O hiato «fecha» se $f_\text{sup} < 1{,}5\,f_B$. Só então se calculam:

$$f_\text{centro} = \frac{f_\text{inf} + f_\text{sup}}{2}, \qquad \text{largura} = f_\text{sup} - f_\text{inf}$$

---

## 4. Incertezas

### 4.1 De cada número

$$\sigma^2 = \sigma_\text{estat}^2 + \sigma_\text{ond}^2 + \sigma_\text{ref}^2\ (+\,\sigma_\text{res}^2 \text{ nas frequências})$$

**Estatística.** Dispersão entre as n repetições:

$$\sigma_\text{estat} = \frac{s_\text{rep}}{\sqrt n}$$

**Ondulação na profundidade.** $D = T_0 - M$, com as duas partes vindas de zonas diferentes do espectro:

1. **Pontos independentes.** Pontos mais próximos do que a largura de suavização (1/6 de oitava) não são independentes. Em cada janela contam-se

$$n_\text{base} = \frac{\log_2(0{,}70/0{,}35)}{1/6} = 6, \qquad n_\text{jan} = \frac{\log_2(1{,}1/0{,}9)}{1/6} = 1{,}74$$

2. **Variância de cada parte.** M é uma média de $n_\text{jan}$ pontos. T0 é uma mediana de $n_\text{base}$ pontos, e a variância da mediana é π/2 vezes a da média para ruído gaussiano.

$$\operatorname{Var}(D) = \operatorname{Var}(T_0) + \operatorname{Var}(M) = \frac{\pi}{2}\frac{\sigma^2}{6} + \frac{\sigma^2}{1{,}74} = (0{,}26 + 0{,}58)\,\sigma^2$$

3. **Resultado.**

$$\sigma_\text{ond}(D) = \sqrt{0{,}84}\;\text{ondulação} = 0{,}92\;\text{ondulação}$$

**Ondulação nos bordos.** Um erro δT em T desloca o cruzamento do limiar de δf = δT / |dT/df|. O declive dT/df ajusta-se numa janela de 1/6 de oitava à volta do bordo.

$$\sigma_\text{ond}(f_\text{inf}) = \frac{\text{ondulação}}{|dT/df|}, \qquad \sigma_\text{ond}(f_\text{centro}) = \tfrac12\sqrt{\sigma_\text{inf}^2 + \sigma_\text{sup}^2}, \qquad \sigma_\text{ond}(\text{largura}) = \sqrt{\sigma_\text{inf}^2 + \sigma_\text{sup}^2}$$

**Referência.** O número $x_k$ calculado com cada referência candidata k, em torno do valor central $x_0$. O peso de cada candidata é $w_k = 1/\text{ondulação}_k^2$, por isso uma referência má pesa pouco.

$$\sigma_\text{ref} = \sqrt{\frac{\sum_k w_k (x_k - x_0)^2}{\sum_k w_k}}$$

**Resolução.** O valor verdadeiro está algures num intervalo de largura Δf com igual probabilidade. A variância de uma distribuição uniforme de largura w é w²/12:

$$\sigma_\text{res} = \frac{\Delta f}{\sqrt{12}} = 3{,}4\ \text{Hz}$$

### 4.2 De cada bloco (um a, média dos N ≥ 4)

Média ponderada, com $w_i = 1/\sigma_i^2$:

$$\bar x = \frac{\sum w_i x_i}{\sum w_i}, \qquad \sigma_\text{disp} = \sqrt{\frac{\sum w_i (x_i - \bar x)^2}{(n-1)\sum w_i}}, \qquad \sigma_\text{sis} = \frac{\sum w_i \sqrt{\sigma_{\text{ond},i}^2 + \sigma_{\text{ref},i}^2}}{\sum w_i}$$

$$\sigma_\text{bloco}^2 = \sigma_\text{disp}^2 + \sigma_\text{sis}^2$$

As duas partes comportam-se de forma diferente quando se faz a média:
- **σ_disp diminui** com a média, como qualquer erro aleatório.
- **σ_sis não diminui.** É a parte comum a todos os N do bloco: a mesma referência e a mesma montagem. Um enviesamento partilhado por todas as medições não desaparece com a média.

### 4.3 Comparação com um valor esperado

$$z = \frac{|x - x_0|}{\sqrt{\sigma_x^2 + \sigma_0^2}} \qquad (z < 2:\ \text{compatíveis})$$

---

## 5. Ajustes

### 5.1 Mínimos quadrados ponderados

Para y = m·x + b, minimiza-se

$$\chi^2 = \sum_i \frac{(y_i - m x_i - b)^2}{\sigma_i^2}$$

Com $w_i = 1/\sigma_i^2$, $S = \sum w$, $S_x = \sum w x$, $S_y = \sum w y$, $S_{xx} = \sum w x^2$, $S_{xy} = \sum wxy$ e $\Delta = S S_{xx} - S_x^2$:

$$m = \frac{S S_{xy} - S_x S_y}{\Delta}, \quad b = \frac{S_{xx} S_y - S_x S_{xy}}{\Delta}, \quad \sigma_m = \sqrt{\frac{S}{\Delta}}, \quad \sigma_b = \sqrt{\frac{S_{xx}}{\Delta}}, \quad \operatorname{cov}(m,b) = -\frac{S_x}{\Delta}$$

**Reta pela origem** (y = m·x):

$$m = \frac{S_{xy}}{S_{xx}}, \qquad \sigma_m = \frac{1}{\sqrt{S_{xx}}}$$

**Incerteza em x** (a, com σ_a = 0,1 mm), por variância efetiva:

$$\sigma_{i,\text{ef}}^2 = \sigma_{y,i}^2 + (m\,\sigma_{x,i})^2$$

Como σ_ef depende de m, minimiza-se χ²(m) numericamente.

**Faixa ±1σ da reta** desenhada nos gráficos:

$$\sigma_y(x) = S_f\sqrt{\sigma_b^2 + x^2\sigma_m^2 + 2x\operatorname{cov}(m,b)}$$

### 5.2 Qualidade do ajuste e fator de escala

$$\nu = n - n_\text{par}, \qquad \chi^2/\nu, \qquad p = P(\chi^2_\nu \ge \chi^2_\text{obs})$$

$$S_f = \begin{cases}\sqrt{\chi^2/\nu} & \text{se } \chi^2/\nu > 1\\ 1 & \text{caso contrário}\end{cases} \qquad \sigma_m^\text{final} = S_f\,\sigma_m, \quad \sigma_b^\text{final} = S_f\,\sigma_b$$

**Porque se multiplica.** Se χ²/ν = 30, os pontos estão √30 ≈ 5,5 vezes mais longe da reta do que as barras dizem: há erros que não foram contados. Sem a correção, a incerteza do declive herda as barras pequenas e fica otimista. Por exemplo, ΓX f_inf pela origem dava m = 277,6 ± 3,7 m/s sem correção e ± 20,4 m/s com ela.

**Porque nunca se encolhe.** Com poucos pontos, χ²/ν baixo acontece muitas vezes por sorte: com ν = 2, $P(\chi^2/\nu < 0{,}3) = 26\,\%$.

χ²/ν e p são sempre impressos, para se ver quando a reta é simples demais.

### 5.3 Os três ajustes

**1. Profundidade contra N.** α em dB/fila. b deve ser ≈ 0.

$$D = \alpha N + b$$

**2. Frequência contra x = 1/(2d), para f_inf e f_centro.** Como $f_B = c\,x$, o declive é uma velocidade.

$$f = m\,x + b$$

- **Reta pela origem:** testa «f ∝ 1/d». Daria m = c no Bragg puro.
- **Reta livre:** descreve os pontos. Por causa do fator $1/\sqrt{1+ff}$ (1.7), m ≠ c.

**3. Mesmo a nas duas direções.** Compara-se a razão com √2 (Bragg puro) e com a teoria de bandas (1.10). Uma diferença positiva seria sobreposição dos hiatos.

$$\frac{f_\text{inf}(\Gamma M)}{f_\text{inf}(\Gamma X)}, \qquad f_\text{sup}(\Gamma X) - f_\text{inf}(\Gamma M)$$
