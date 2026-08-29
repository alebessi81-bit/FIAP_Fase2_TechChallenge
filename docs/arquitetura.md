# Arquitetura e decisões de implementação

Tech Challenge Fase 2 — Projeto 1: Otimização de Modelos de Diagnóstico

---

## 1. Visão geral

O sistema tem três camadas, e a separação entre elas é a decisão de
projeto mais importante do trabalho.

```
┌─────────────────────────────────────────────────────────────┐
│  CAMADA 1 — OTIMIZAÇÃO (offline, executada uma vez)         │
│                                                             │
│   diabetes.csv → Algoritmo Genético → modelo campeão        │
│                   ├── hiperparâmetros do SVM                │
│                   ├── seleção de features                   │
│                   └── quantil do ponto de corte             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  CAMADA 2 — INFERÊNCIA (determinística, sem rede)           │
│                                                             │
│   paciente → pipeline → escore → classificação              │
│                          │                                  │
│                          └→ contribuições por oclusão       │
│                                                             │
│   Toda a estatística acontece aqui, em Python auditável.    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  CAMADA 3 — TRADUÇÃO (LLM, opcional)                        │
│                                                             │
│   evidência (JSON) → prompt → Gemini → texto                │
│                                         │                   │
│                                         ▼                   │
│                              avaliação em 2 camadas         │
└─────────────────────────────────────────────────────────────┘
```

**O LLM nunca vê o modelo nem os dados brutos.** Ele recebe um dicionário
fechado com números já calculados, e sua única tarefa é transformar isso
em texto. Se o laudo sair errado, o erro está na redação, nunca na conta.
Num contexto clínico, essa separação é o que torna o sistema defensável.

---

## 2. Fluxo de uma requisição

```
Cliente
   │
   ▼
┌──────────────────────────────────────────┐
│ Middleware                               │
│  • gera request_id                       │
│  • cronometra                            │
│  • registra métricas                     │
│  • loga em JSON                          │
└──────────────────────────────────────────┘
   │
   ├──── POST /predict ──────────────────────────────┐
   │        validação Pydantic                       │
   │        zeros → NaN                              │
   │        StandardScaler → KNNImputer → SVC        │
   │        escore vs. ponto de corte                │
   │        contribuições por oclusão                │
   │        ~5 ms, sem dependência externa           │
   │                                                 ▼
   │                                            resposta
   │
   ├──── POST /explain ──────────────────────────────┐
   │        tudo de /predict                         │
   │        + monta evidência                        │
   │        + chama o Gemini                         │
   │        + roda as 8 verificações                 │
   │        ~2 s, depende de serviço externo         │
   │        falha → 503 (não 500)                    │
   │                                                 ▼
   │                                            resposta
   │
   ├──── GET /health   → usado pelo Cloud Run
   └──── GET /metrics  → métricas de serviço e de modelo
```

### Por que `/predict` e `/explain` são rotas separadas

Elas têm perfis de desempenho e de falha completamente diferentes:

| | `/predict` | `/explain` |
|---|---|---|
| Latência | ~5 ms | ~2 s |
| Dependência externa | nenhuma | Gemini |
| Determinístico | sim | não |
| Custo por chamada | zero | consome cota |
| Criticidade | alta (triagem) | complementar |

Separadas, uma indisponibilidade do Gemini nunca derruba a triagem. Se
fossem um parâmetro da mesma rota, a tentação de aplicar o mesmo timeout
e o mesmo alerta às duas seria grande, e o resultado seria acoplar a
função crítica a um serviço de terceiros.

---

## 3. Decisões de implementação

### 3.1 Fitness baseada em sensibilidade e especificidade

A primeira versão usava `0.6 · recall + 0.4 · F1`. O algoritmo genético
encontrou a solução degenerada: baixar o limiar até classificar **todos**
os pacientes como diabéticos. Recall 1.0, F1 em torno de 0.52 — suficiente
para vencer qualquer modelo honesto.

O problema é que o F1 ignora os verdadeiros negativos, então um modelo que
diz "sim" para todo mundo nunca é punido. A especificidade fecha a brecha,
porque a solução degenerada tem especificidade exatamente zero.

Fitness final:

```
fitness = 0.65 · sensibilidade + 0.35 · especificidade
          − 0.02 · (features usadas / total)
          − 2.0 · max(0, 0.50 − especificidade)
```

O último termo é uma restrição: um exame de triagem que marca mais da
metade dos saudáveis como suspeitos perde utilidade clínica, por mais
alto que seja o recall.

### 3.2 Ponto de corte codificado como quantil

A primeira versão guardava o limiar como valor absoluto do
`decision_function`. Ele era calibrado durante a validação cruzada, onde
cada modelo vê 80% do treino, e depois aplicado ao modelo final, treinado
com 100%. A escala muda entre os dois, e o corte acabou ficando abaixo de
todos os escores — todo paciente classificado como diabético.

A correção foi guardar um **quantil**. O gene diz "corte no percentil 40
dos escores de treino", e o valor absoluto é recalculado sempre que o
modelo é retreinado.

### 3.3 Validação cruzada sobre o treino, nunca sobre o teste

O notebook da Fase 1 escolheu o `k` do KNN medindo o erro no conjunto de
teste. Numa busca de 14 valores é um deslize pequeno; num algoritmo
genético que avalia milhares de configurações seria fatal.

Todo o fitness aqui vem de validação cruzada estratificada de 5 folds
sobre o treino, com scaler e imputer reajustados dentro de cada fold. O
conjunto de teste é tocado uma única vez, na avaliação final.

**Consequência prática:** o modelo escolhido foi o do experimento B, que
teve o melhor fitness em validação cruzada — e não o do C, que teve o
melhor desempenho no teste. Escolher pelo teste seria repetir o erro da
Fase 1.

### 3.4 Teto de iterações no SVM

Sem `max_iter`, combinações patológicas (kernel `poly` com C alto e gamma
alto) faziam o SVM rodar indefinidamente e travavam o experimento. Com o
teto de 200 mil iterações, o SVM devolve a melhor solução encontrada até
ali, e um indivíduo que não converge tende a ter fitness ruim e desaparece
da população sozinho.

### 3.5 Contribuições por oclusão, e não SHAP

Para cada exame, substituímos o valor do paciente pela média da população
e medimos o quanto o escore se move:

```
contribuição_j = f(paciente) − f(paciente com exame j na média)
```

Escolhemos oclusão por ser **determinística** (dois laudos do mesmo
paciente nunca divergem) e por rodar em milissegundos, o que viabiliza
explicar pacientes sob demanda numa API.

**Limitação:** a oclusão distribui pior o crédito entre variáveis
correlacionadas do que o SHAP.

### 3.6 Modelo treinado na subida, sem `.pkl`

O treino leva cerca de um segundo e evita carregar um artefato
serializado, que criaria dependência da versão exata do scikit-learn
usada para gerá-lo. O custo é um cold start ligeiramente maior; o
benefício é não ter incompatibilidade silenciosa de versão.

### 3.7 Zeros tratados como ausentes também na API

No dataset original, zero em Glicose, Insulina ou IMC significa "não
medido". A API reproduz essa convenção. Fazer diferente seria divergência
treino-serviço: o modelo veria em produção um valor que nunca viu no
treino.

---

## 4. Monitoramento

### Métricas de serviço
Requisições, taxa de erro, latência p50/p95/p99, quebra por rota.
Respondem: *a API está de pé e rápida?*

### Métricas de modelo
Taxa de classificações "risco elevado", distribuição dos escores,
proporção de campos ausentes na entrada. Respondem: *o modelo continua
vendo o mesmo tipo de paciente que viu no treino?*

A segunda categoria é a que costuma faltar. Um modelo pode degradar
silenciosamente sem que nada quebre: se a taxa de risco elevado sair dos
~40% observados em validação para 80%, ou a população mudou ou algo se
alterou no pré-processamento. Nenhum monitoramento de infraestrutura
detectaria isso — a latência continuaria ótima e o status continuaria 200.

### Logs

JSON estruturado, uma linha por evento, com os campos `severity` e
`message`. O Cloud Logging faz o parsing automático desse formato, então
os campos extras viram campos pesquisáveis no console. Cada requisição
carrega um `request_id`, o que permite juntar todos os eventos de uma
mesma chamada.

### Limitação conhecida

As métricas vivem na memória do processo. No Cloud Run, com várias
instâncias e scale-to-zero, cada instância tem seus próprios contadores e
tudo se perde quando ela é desligada. Servem para inspeção pontual e
demonstração. Em produção real, os números iriam para o Cloud Monitoring
— viável justamente porque já emitimos logs estruturados, e cada linha
pode virar uma métrica baseada em log.

---

## 5. Escalabilidade

O Cloud Run resolve o requisito de escalabilidade automática por
configuração, sem cluster:

| Parâmetro | Valor | Motivo |
|---|---|---|
| `min-instances` | 0 | Sem tráfego, custo zero |
| `max-instances` | 10 | Teto de custo |
| `concurrency` | 40 | `/predict` é leve; uma instância atende várias |
| `cpu` | 1 | Suficiente para SVM em 5 features |
| `memory` | 512Mi | Modelo e dados de treino cabem folgados |
| `timeout` | 60s | `/explain` depende do Gemini |

Um worker por container é proposital: o Cloud Run escala criando mais
instâncias, não mais processos dentro da mesma. Vários workers
multiplicariam o modelo na memória sem ganho de vazão.

**Trade-off do scale-to-zero:** a primeira requisição depois de um período
ocioso paga o cold start (imagem + treino do modelo). Para uma triagem
que não é de emergência, é um preço aceitável em troca de custo zero
ocioso. Se o requisito fosse latência garantida, `min-instances=1`
resolveria, ao custo de uma instância sempre ligada.

---

## 6. Segurança

- A chave do Gemini vem de variável de ambiente, nunca do código. Em
  produção, do Secret Manager.
- Container roda com usuário sem privilégios.
- Validação de faixas na entrada: o modelo não tem como saber que uma
  glicose de 5000 é erro de digitação — produziria um escore de aparência
  normal. Rejeitar na porta é mais seguro que devolver predição sem
  sentido.
- Nenhum dado de paciente é persistido. Os logs registram classificação e
  escore, não os valores dos exames.

---

## 7. Limitações do sistema

1. **Ganho é de calibração, não de discriminação.** O AUC caiu levemente
   nos três experimentos (0.809 → 0.780–0.801). Os modelos otimizados não
   separam melhor; operam num ponto mais favorável ao recall.

2. **Seleção de features instável.** Cada experimento escolheu um
   conjunto diferente; apenas Glicose e Idade aparecem nos três.

3. **Explicabilidade individual frágil.** Com `C = 0.042`, a fronteira de
   decisão é quase plana e as contribuições ficam pequenas e instáveis.
   Bom desempenho populacional não garante explicação local confiável.

4. **População de referência restrita.** Mulheres de origem Pima com 21
   anos ou mais. Os percentis não são válidos fora desse grupo.

5. **Amostra de teste pequena.** 154 pacientes, 54 diabéticos. Os três
   experimentos estão separados por 3 pacientes — ruído, não diferença
   real.
