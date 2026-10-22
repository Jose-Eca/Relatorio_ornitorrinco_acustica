# Cristal sónico — como usar

## Abrir
1. Instala Python (python.org, marca «Add Python to PATH»).
2. Windows: duplo clique em `INICIAR.bat` (da 1.ª vez instala o que falta; precisa de internet).
   Mac/Linux: `python3 app.py`.

Tudo fica na pasta `projeto/` (gravações, plano, resultados). Leva-a num USB no fim de cada sessão.

## Antes de medir (uma vez por sessão)
1. **⚙ Definições** → escolhe o microfone e a coluna.
   Bluetooth: a saída «Headphones/Stereo» da coluna, nunca «Hands-Free».
2. Mete a temperatura da sala.
3. **🔊 Testar som** sem cristal: tem de dizer «Nível bom». Depois disso **não mexas no volume**.
4. Coluna de som, microfone e face do cristal sempre nos mesmos sítios (marca com fita-cola).

## Medir
- À direita aparece o passo a fazer e a instrução (ex. «Monta 4 × 6 colunas com a = 48 mm»).
- Monta e carrega em **▶ MEDIR**. A app grava 5 vezes e mostra logo T(f) e os números do hiato.
- Para cada a: primeiro a **referência** (sem colunas), depois N = 6, e vais tirando filas do lado do microfone.
- Correu mal? Clica no passo e **▶ MEDIR** outra vez (o antigo fica em `projeto/dados/_antigos`).
- Aviso «SATURAÇÃO» → baixa o volume e repete. «Sinal fraco» → sobe o volume e repete.

## Extras (com o tempo que sobrar)
**+ Passo extra**: dá um nome, a, N e a direção.
- **ΓM:** roda o cristal 45°. O hiato deve passar para c/(√2·a), cerca de 1.4 vezes mais alto.
- **Defeito:** tira a coluna do meio. Deve aparecer um pico estreito dentro do hiato.
- **Remontagem:** desmonta e volta a montar a mesma configuração (ex. a = 48, N = 4) 2–3 vezes. A diferença entre montagens é a incerteza sistemática.

Os extras não entram nas regressões. Aparecem na lista e podes vê-los um a um.

## Resultados
**📊 Resultados** mostra:
1. **T(f) por a**: um gráfico por a, com todos os N.
2. **Profundidade vs N**: com a reta ajustada para cada a (dB por fila).
3. **f_centro vs 1/a**: com a reta ajustada. O declive vale c/2, portanto **c = 2 × declive**.

Por baixo aparece o texto com todos os ajustes e a tabela. **💾 Guardar tudo** grava os PNG, `tabela.csv` e `resultados.txt` em `projeto/resultados/`.

**Treinar em casa:** ⚙ Definições → «Modo simulação». Apaga a pasta `projeto/` antes das medições a sério.
