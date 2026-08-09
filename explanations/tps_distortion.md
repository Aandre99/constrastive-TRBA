# Por que o TPS está "distorcendo" imagens em vez de corrigir?

## Resumo rápido

A saída do módulo TPS mostra que imagens de placas de carro — que **já estão praticamente retas** — sofrem uma deformação sutil (bordas esticadas, leve curvatura). Isso é **esperado** e não é um bug. A explicação se resume a três fatores complementares.

---

## 1. O TPS é treinado end-to-end — não há supervisão geométrica

O `TPS_SpatialTransformerNetwork` é um **Spatial Transformer Network (STN)** treinado de forma end-to-end junto com o restante do pipeline (ResNet → BiLSTM → Attention). Isso significa que:

- **Não existe ground-truth geométrico.** Ninguém diz ao modelo "a imagem correta é *esta* retificação". O único sinal de supervisão é a **loss de reconhecimento de texto** (CrossEntropy do decoder Attention).
- O TPS aprende **qualquer deformação que reduza a loss do reconhecimento**, mesmo que isso não corresponda a uma "retificação" visualmente intuitiva.
- Se distorcer levemente a imagem ajudar o extrator de features a produzir representações melhores, o modelo fará isso.

> O TPS não minimiza uma "loss de retificação" — ele minimiza a loss de reconhecimento. Qualquer deformação que melhore o OCR será aprendida, mesmo que pareça contra-intuitiva.

---

## 2. As placas de carro já são retilíneas — o TPS não tem o que corrigir

O TPS (do paper RARE) foi projetado para lidar com **texto curvo em cena** — placas de loja arqueadas, texto em garrafas, etc. No dataset RodoSol de placas de carro:

- As placas são **rígidas e planas**.
- A perspectiva é geralmente **frontal ou quase-frontal**.
- Não há curvatura significativa para o TPS corrigir.

Quando a imagem de entrada já é geometricamente "boa", o módulo TPS não tem utilidade real — mas ele **ainda precisa produzir alguma saída**. A `LocalizationNetwork` prevê pontos fiduciais C' que, para imagens retas, deveriam ser aproximadamente iguais aos pontos-alvo C. Na prática, o modelo aprende pequenos desvios dos pontos identidade que **ajudam marginalmente o reconhecimento** (ou pelo menos não o prejudicam o suficiente para serem penalizados pela loss).

---

## 3. A inicialização do bias favorece uma transformação "quase-identidade"

Na `LocalizationNetwork` (modules/transformation.py L42-L83):

```python
# Pesos da fc2 inicializados em zero
self.localization_fc2.weight.data.fill_(0)

# Bias: pontos fiduciais distribuídos ao longo das bordas superior e inferior
ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
```

Com pesos zero e bias fixo, a rede **começa** produzindo sempre os mesmos pontos C' (independente da entrada). Durante o treino, os pesos são levemente ajustados. Mas como a loss de reconhecimento no dataset de placas não exige grandes retificações, os pesos permanecem **próximos de zero** e a transformação resultante é uma **perturbação leve da identidade**.

Note que os pontos y da inicialização vão de `0.0 → -1.0` (topo) e `1.0 → 0.0` (base), formando um "V" — não uma grade retangular. Isso já é um desvio sutil da identidade pura, o que explica a leve deformação visível mesmo no início do treino.

---

## 4. Isso prejudica o reconhecimento?

**Provavelmente não de forma significativa**, mas pode não ajudar. Para datasets de placas retas, há duas opções:

| Abordagem | Prós | Contras |
|-----------|------|---------|
| **Manter TPS** (`--Transformation TPS`) | Compatível com o checkpoint pré-treinado; pode ajudar em placas com perspectiva extrema | Adiciona parâmetros desnecessários; pode gerar artefatos nas bordas |
| **Remover TPS** (`--Transformation None`) | Pipeline mais simples; sem distorções espúrias | Perde a compatibilidade com checkpoints TPS; pior em dados com texto curvo |

---

## Conclusão

O comportamento observado é **normal e esperado** para um STN treinado end-to-end em dados que não contêm distorção geométrica significativa. O TPS aprende a fazer "quase nada", e o "quase" resulta em perturbações visuais leves que são irrelevantes para a acurácia do reconhecimento. Se o objetivo for eliminar essas distorções, basta treinar com `--Transformation None`.
