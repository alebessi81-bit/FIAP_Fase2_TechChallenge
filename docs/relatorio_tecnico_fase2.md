# Relatório Técnico — Tech Challenge Fase 2

**Projeto 1: Otimização de Modelos de Diagnóstico**

PosTech FIAP — IA para Desenvolvedores

Repositório: `https://github.com/mparra137/FIAP_Fase1_TechChallenge` (Fase 1)

---

## Sumário executivo

Partimos do modelo SVM de triagem de diabetes entregue na Fase 1 e
aplicamos um algoritmo genético para otimizá-lo. O ganho principal foi em
**recall: de 0,722 para 0,870**. Traduzindo para pacientes: num grupo de
100 diabéticos, o modelo original deixava 28 sem diagnóstico; o otimizado
deixa 13.

Sobre esse resultado, três ressalvas honestas que este relatório
desenvolve nas seções finais:

1. O ganho é de **calibração**, não de capacidade discriminativa — o AUC
   caiu levemente.
2. A **accuracy caiu** de 0,714 para 0,682, consequência esperada de
   priorizar recall.
3. Os três experimentos estão separados por **3 pacientes** no conjunto
   de teste, o que é ruído e não diferença real de qualidade.

Além da otimização, integramos uma LLM (Gemini) para traduzir os
diagnósticos em linguagem natural para profissionais de saúde, com um
sistema de avaliação de qualidade em duas camadas independentes.

---

## 1. Ponto de partida: auditoria do modelo da Fase 1

Antes de otimizar, auditamos o notebook original. Três achados mudaram o
desenho da solução.

### 1.1 Vazamento na escolha de hiperparâmetro

A escolha do melhor `k` do KNN foi feita medindo o erro no **conjunto de
teste**:

```python
for i in range(1, 15):
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)   # teste usado para escolher k
```

Numa busca de 14 valores, é um deslize pequeno. Num algoritmo genético
que avalia milhares de configurações, seria fatal: o AG encontraria o
indivíduo que melhor decora o conjunto de teste, e as métricas finais não
significariam nada.

**Decisão:** todo o fitness da Fase 2 vem de validação cruzada
estratificada de 5 folds sobre o conjunto de treino. O teste é tocado uma
única vez, na avaliação final.

### 1.2 Comparação desigual entre modelos

O Random Forest foi treinado em `X_train` cru, sem escalonamento e sem
imputação, enquanto KNN e SVM receberam os dados tratados pelo
`KNNImputer`. A comparação entre os três, portanto, não era justa.

### 1.3 Pré-processamento correto

Uma verificação que resultou negativa e vale registrar: o `train_test_split`
vem antes do `StandardScaler` e do `KNNImputer`, e ambos são ajustados
apenas no treino. Não há vazamento no pré-processamento.

### 1.4 Modelo de referência

O SVM vencedor da Fase 1 **não teve nenhum hiperparâmetro ajustado** —
`C=1.0` e `gamma='scale'` são os padrões do scikit-learn. Isso significa
espaço amplo de melhora para o algoritmo genético.

Reproduzimos esse baseline no código da Fase 2 e obtivemos recall 0,7222
e accuracy 0,7143 — praticamente idênticos aos 0,72 do notebook original,
o que confirma a fidelidade da reprodução.

---

## 2. Implementação do algoritmo genético

### 2.1 Codificação do cromossomo

Todos os genes são números reais no intervalo [0, 1], independentemente
do que representam. Um gene contínuo (`C`), um categórico (`kernel`) e um
binário (usar ou não uma feature) são armazenados da mesma forma e só
ganham significado na decodificação.

Isso traz duas vantagens: os operadores de cruzamento e mutação funcionam
sobre o vetor inteiro sem tratamento especial por tipo; e **qualquer
vetor em [0,1]¹³ é sempre um indivíduo válido**, então o AG nunca
desperdiça avaliações com configurações impossíveis de treinar.

| Índice | Gene | Decodificação | Faixa |
|---|---|---|---|
| 0 | C | 10^(−2 + 4g) | 0,01 – 100 |
| 1 | gamma | 10^(−4 + 5g) | 1e−4 – 10 |
| 2 | kernel | categórico | rbf, poly, sigmoid |
| 3 | class_weight | categórico | None, balanced |
| 4–11 | máscara de features | g ≥ 0,5 | 8 bits |
| 12 | quantil do corte | 0,20 + 0,70g | 0,20 – 0,90 |

O cromossomo otimiza três coisas ao mesmo tempo: hiperparâmetros do SVM
(requisito obrigatório), seleção de features e ponto de operação do
classificador.

### 2.2 Função fitness

```
fitness = 0,65 · sensibilidade
        + 0,35 · especificidade
        − 0,02 · (features usadas / total)
        − 2,00 · max(0; 0,50 − especificidade)
```

Avaliada por validação cruzada estratificada de 5 folds sobre o treino,
com scaler e imputer reajustados dentro de cada fold.

**Justificativa dos componentes:**

- Os pesos 0,65/0,35 preservam a prioridade clínica estabelecida na Fase 1
  (falso negativo custa mais que falso positivo).
- Sensibilidade e especificidade são o par padrão em medicina
  diagnóstica, o que torna a escolha fácil de defender.
- O termo de parcimônia prefere, entre modelos equivalentes, o que usa
  menos exames — triagem mais barata e mais rápida.
- A restrição de especificidade mínima existe porque um exame que sinaliza
  mais da metade dos saudáveis sobrecarrega o serviço, por mais alto que
  seja o recall.

A seção 6.1 explica por que essa não foi a primeira formulação.

### 2.3 Operadores

**Seleção por torneio (k=3):** sorteia k indivíduos e devolve o melhor. A
pressão seletiva depende apenas de k, e não da escala do fitness.

**Seleção por roleta:** probabilidade proporcional ao fitness. Sensível a
escala — quando os indivíduos convergem, vira quase um sorteio uniforme.

**Cruzamento uniforme:** cada gene vem de um dos pais, sorteado
independentemente. Bom para genes categóricos e para a máscara de
features, porque preserva valores que já funcionam em vez de criar médias
sem sentido.

**Cruzamento BLX-α (α=0,5):** cada gene é sorteado num intervalo que se
estende além dos valores dos pais. Bom para `C` e `gamma`, porque permite
explorar fora do segmento entre os pais.

**Mutação gaussiana (σ=0,15):** ruído gaussiano em vez de substituição
aleatória. Faz busca local em torno do indivíduo, preservando boas
soluções sem impedir a fuga de ótimos locais.

**Elitismo:** os melhores indivíduos passam intactos para a geração
seguinte.

---

## 3. Experimentos

### 3.1 Configurações

| | Exp. A | Exp. B | Exp. C |
|---|---|---|---|
| Estratégia | Equilíbrio | Exploração | Intensificação |
| População | 30 | 60 | 20 |
| Gerações | 20 | 20 | 40 |
| Taxa de mutação | 0,10 | 0,30 | 0,05 |
| Taxa de cruzamento | 0,80 | 0,60 | 0,90 |
| Seleção | Torneio | Torneio | Roleta |
| Cruzamento | Uniforme | BLX-α | BLX-α |
| Elite | 2 | 2 | 4 |
| **Tempo de execução** | 339,7 s | 677,0 s | 273,2 s |

Os três variam não apenas os números, mas a **estratégia de busca**.

### 3.2 Soluções encontradas

| | Exp. A | Exp. B | Exp. C |
|---|---|---|---|
| Fitness (CV) | 0,7909 | **0,7959** | 0,7828 |
| Sensibilidade (CV) | 0,9238 | 0,9300 | 0,9019 |
| Especificidade (CV) | 0,5925 | 0,5825 | 0,5975 |
| Kernel | sigmoid | rbf | poly |
| C | 0,0735 | 0,0424 | 0,0193 |
| gamma | 0,34899 | 0,00090 | 0,12615 |
| class_weight | balanced | balanced | balanced |
| Quantil do corte | 0,422 | 0,404 | 0,434 |
| Features usadas | 6 de 8 | 5 de 8 | 5 de 8 |

**Features selecionadas:**

- **A:** Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age
- **B:** Pregnancies, Glucose, Insulin, BMI, Age
- **C:** Glucose, BloodPressure, SkinThickness, DiabetesPedigreeFunction, Age

### 3.3 Convergência do experimento B

| Geração | Melhor fitness | Fitness médio | Desvio padrão |
|---|---|---|---|
| 1 | 0,7781 | 0,5417 | 0,1730 |
| 5 | 0,7851 | 0,7394 | 0,0479 |
| 10 | 0,7878 | 0,7481 | 0,0664 |
| 15 | 0,7959 | 0,7587 | 0,0800 |
| 20 | 0,7959 | 0,7583 | 0,0559 |

Três leituras:

1. **O fitness médio sobe rápido nas quatro primeiras gerações** (0,54 →
   0,73) e depois estabiliza. A maior parte do ganho vem cedo.
2. **O desvio padrão cai de 0,173 para 0,056, mas não vai a zero.** É a
   taxa de mutação de 0,30 mantendo diversidade — comportamento
   pretendido para uma configuração exploratória.
3. **O melhor fitness ainda subiu na geração 15.** O experimento B não
   havia convergido quando as 20 gerações acabaram, o que sugere que mais
   gerações renderiam algum ganho adicional.

### 3.4 Padrões estáveis entre os experimentos

O que se repete nos três, e por isso merece interpretação:

- **C sempre baixíssimo** (0,019 a 0,073): regularização forte. Com 768
  pacientes e poucas features, o SVM generaliza melhor sendo simples.
- **Quantil sempre entre 0,40 e 0,43:** o corte no percentil 40, em vez do
  meio implícito, é a origem da maior parte do ganho de recall.
- **`class_weight='balanced'` nos três.**

O que **não** é estável, e por isso não deve ser interpretado:

- **Kernel diferente nos três** (sigmoid, rbf, poly), com desempenho
  parecido. Reforça que o ponto de operação pesa mais que a forma do
  modelo.
- **Seleção de features instável:** apenas Glucose e Age aparecem nos
  três. Atribuir significado clínico a uma escolha em boa parte aleatória
  seria incorreto.

---

## 4. Comparativo: original vs. otimizados

Todas as métricas abaixo foram medidas **uma única vez**, no conjunto de
teste de 154 pacientes (54 diabéticos), que não participou de nenhuma
etapa da evolução.

| Modelo | Accuracy | Recall | Precision | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Baseline (Fase 1)** | **0,7143** | 0,7222 | **0,5735** | 0,6393 | **0,8087** |
| AG — Experimento A | 0,6753 | 0,8333 | 0,5233 | 0,6429 | 0,8006 |
| AG — Experimento B | 0,6818 | 0,8704 | 0,5281 | 0,6573 | 0,7989 |
| AG — Experimento C | 0,6883 | **0,8889** | 0,5333 | **0,6667** | 0,7798 |

### 4.1 Tradução clínica

| Modelo | Diabéticos identificados (de 54) |
|---|---|
| Baseline | 39 |
| Experimento A | 45 |
| Experimento B | 47 |
| Experimento C | 48 |

**Os três experimentos do AG estão separados por 3 pacientes.** Isso é
ruído, não diferença real de qualidade. A distância para o baseline, de 6
a 9 pacientes, é a conclusão que se sustenta.

### 4.2 Qual modelo foi escolhido, e por quê

**O experimento B**, porque teve o melhor fitness em validação cruzada
(0,7959).

O experimento C teve desempenho melhor no conjunto de teste. Escolhê-lo
por isso seria usar o teste para decidir — exatamente o erro identificado
na seção 1.1. O conjunto de teste serve para estimar o desempenho do
modelo escolhido, nunca para escolhê-lo.

Este é o ponto metodológico mais importante do trabalho, e ele nos custou
o modelo que parecia melhor.

### 4.3 O que o ganho realmente é

**O AUC caiu nos três experimentos** (0,8087 → 0,7798–0,8006). O AUC mede
o poder de separação independentemente do limiar, então essa queda
significa que os modelos otimizados **não separam melhor** que o
baseline — eles operam num ponto mais favorável ao recall.

O ganho é real e clinicamente útil, mas é ganho de **calibração**, não de
capacidade discriminativa. Parte dele poderia ter sido obtida apenas
ajustando o limiar do modelo original. A contribuição do algoritmo
genético foi encontrar essa combinação sozinho, junto com hiperparâmetros
e features, sem que ninguém precisasse suspeitar de antemão que o limiar
era a alavanca principal.

---

## 5. Integração com LLM

### 5.1 Arquitetura da integração

**O LLM nunca vê o modelo nem os dados brutos.** Ele recebe um dicionário
fechado com números já calculados em Python, e sua única tarefa é
transformar isso em texto.

```
paciente → pipeline → escore → contribuições  ┐
                                              ├→ evidência (JSON) → LLM → texto
métricas de validação do modelo               ┘
```

Toda a estatística acontece em código auditável antes da chamada. Se o
laudo sair errado, o erro está na redação, nunca na conta. Num contexto
clínico, essa separação é o que torna o sistema defensável.

Consequência prática: **trocar de modelo de LLM é uma variável de
ambiente.** O prompt, a evidência e a avaliação não mudam.

### 5.2 Contribuições por oclusão

Para cada exame, substituímos o valor do paciente pela média da população
(zero no espaço padronizado) e medimos o deslocamento do escore:

```
contribuição_j = f(paciente) − f(paciente com exame j na média)
```

Escolhemos oclusão em vez de SHAP KernelExplainer porque é
**determinística** — dois laudos do mesmo paciente nunca divergem — e roda
em milissegundos, o que viabiliza explicar pacientes sob demanda numa API.

**Limitação:** a oclusão distribui pior o crédito entre variáveis
correlacionadas do que o SHAP.

### 5.3 Técnicas de prompt engineering aplicadas

| Técnica | Implementação | Motivo |
|---|---|---|
| **Persona** | Assistente de apoio à decisão clínica falando com o médico, não com o paciente | Muda vocabulário, nível de detalhe e tipo de recomendação |
| **Ancoragem** | Números num bloco JSON delimitado; proibição de usar qualquer valor fora dele | Defesa principal contra alucinação em contexto clínico |
| **Esquema de saída** | Quatro seções fixas obrigatórias | Sem formato estável, avaliação automática é impossível |
| **Restrições negativas** | Não afirmar diagnóstico, não prescrever, não omitir limitações | Segurança clínica |
| **Exemplo one-shot** | Um caso resolvido com outros números | Calibra tom e extensão melhor que descrever o formato |
| **Temperatura baixa** | 0,2 | Fidelidade importa mais que variedade |
| **Ordenação do prompt** | Persona → regras → formato → exemplo → dados | Dados por último reduzem a chance de o modelo perdê-los de vista |

O modelo utilizado foi o **Gemini** (`gemini-3.5-flash-lite`), acessado
pela camada gratuita do Google AI Studio.

### 5.4 Estrutura do prompt

**Persona:**
> Você é um assistente de apoio à decisão clínica que trabalha junto a
> médicos de atenção primária em um hospital universitário. Seu papel é
> traduzir a saída de um modelo estatístico de triagem de diabetes tipo 2
> para uma linguagem que o médico consiga usar em consulta. Você não fala
> com o paciente. Você fala com o profissional de saúde.

**Regras obrigatórias (resumo):**

1. Usar exclusivamente os números do bloco DADOS.
2. Nunca afirmar que o paciente tem ou não tem diabetes.
3. Nunca prescrever medicamento nem definir dose.
4. Sempre mencionar as limitações estatísticas do modelo.
5. Declarar explicitamente quando um valor foi imputado.
6. Informar sempre a população de referência do modelo.
7. Português do Brasil, 150 a 300 palavras.
8. Exatamente as quatro seções do formato.

**Seções de saída:** RESUMO · FATORES QUE MAIS PESARAM · O QUE ISSO
SUGERE NA PRÁTICA · LIMITAÇÕES

### 5.5 Avaliação de qualidade em duas camadas

#### Camada 1 — Verificações determinísticas (8 regras)

| Regra | O que verifica |
|---|---|
| `classificacao_consistente` | O texto concorda com o veredito do modelo |
| `todas_secoes_presentes` | As quatro seções existem |
| `sem_expressoes_proibidas` | Não afirma diagnóstico nem prescreve |
| `sem_numeros_inventados` | Todo número citado existe na evidência |
| `cita_top3_fatores` | Os três fatores de maior contribuição aparecem |
| `extensao_adequada` | Entre 130 e 340 palavras |
| `menciona_limitacoes` | Cita sensibilidade ou especificidade |
| `declara_populacao_referencia` | Declara a restrição da população Pima |

A verificação de **ancoragem numérica** é a mais valiosa: extrai todo
número do texto e confere se ele existe na evidência, com tolerância para
arredondamentos legítimos e derivações válidas (58,25% de especificidade
autoriza falar em ~42% de falsos positivos).

#### Camada 2 — LLM como avaliador

Uma segunda chamada, com prompt de auditoria, atribui notas de 1 a 5 em
fidelidade, clareza, acionabilidade e segurança. Captura problemas
subjetivos que regra nenhuma pega.

### 5.6 Resultados da avaliação

Na execução final, os textos gerados atingiram **8/8** nas verificações
determinísticas, e o avaliador atribuiu 5/5 em todos os quatro critérios.

**Exemplo de saída (paciente 36, sinal limítrofe):**

> **RESUMO** — O modelo classificou esta paciente como de risco elevado
> para diabetes tipo 2. O escore de 0,8653 ficou apenas 0,0167 acima do
> ponto de corte (0,8486), o que caracteriza um sinal limítrofe de
> triagem e não um caso de alto desvio estatístico.
>
> **O QUE ISSO SUGERE NA PRÁTICA** — Como o escore ultrapassou o corte
> devido exclusivamente ao IMC elevado, enquanto os demais fatores tendem
> a reduzir o risco, o caso sugere cautela.

Esse paciente era, na verdade, **não diabético** — um falso positivo do
modelo. Mas a explicação sinalizou espontaneamente que o caso era frágil.
Um médico lendo isso trataria o resultado com o ceticismo apropriado.

**É a demonstração concreta do valor da camada de LLM:** a interpretação
em linguagem natural carregou informação que o rótulo binário sozinho não
carregava.

### 5.7 Conclusão sobre a avaliação

As duas camadas erram, mas erram coisas diferentes:

- **O avaliador encontrou uma falha real que as 7 regras da época não
  pegavam:** a omissão de que a população de referência são mulheres Pima
  com 21 anos ou mais. Um médico lendo "percentil 86,8 da população de
  referência" assumiria população geral. Isso virou a oitava regra.

- **O avaliador errou de forma reproduzível.** Para o mesmo número
  derivado da especificidade, deu fidelidade 2 num paciente e 5 em outro.
  A crítica que fez ("o número 42 não consta nos dados") estava
  factualmente errada nos dois casos.

**Conclusão:** as regras determinísticas são confiáveis e cegas para o que
não foi previsto; o avaliador é perspicaz e instável. Nenhuma das duas
sozinha seria suficiente.

---

## 6. Desafios enfrentados e soluções implementadas

Esta seção registra os problemas reais encontrados. Todos foram
descobertos **executando o código**, não raciocinando sobre ele — o que é,
em si, uma das conclusões do trabalho.

### 6.1 A função fitness premiava um classificador inútil

**Problema.** A primeira formulação era `0,6 · recall + 0,4 · F1`. O
algoritmo genético encontrou a solução degenerada: baixar o limiar até
classificar **todos** os pacientes como diabéticos. Recall 1,0, precisão
igual à prevalência, F1 suficiente para vencer qualquer modelo honesto.

**Diagnóstico.** O F1 ignora os verdadeiros negativos. Um modelo que diz
"sim" para todo mundo nunca é punido por isso. O AG não estava errado; a
função fitness é que premiava um classificador sem utilidade.

**Solução.** Substituir F1 por especificidade. A solução degenerada tem
especificidade exatamente zero, então morre sozinha. Posteriormente
adicionamos também a restrição de especificidade mínima de 0,50.

### 6.2 O limiar não transferia entre treinos

**Problema.** Mesmo após a correção anterior, a solução degenerada voltou
no conjunto de dados real: recall 1,0 e precisão 0,3506, exatamente a
prevalência.

**Diagnóstico.** O AUC do modelo era 0,8139 — melhor que o baseline. O
modelo estava bom; o limiar é que estava errado. O gene guardava um valor
absoluto do `decision_function`, calibrado durante a validação cruzada
(onde cada modelo vê 80% do treino) e aplicado ao modelo final (treinado
com 100%). A escala muda entre os dois, e o corte acabou ficando abaixo de
todos os escores.

**Solução.** Codificar o corte como **quantil**, não como valor absoluto.
O gene passa a dizer "corte no percentil 40 dos escores de treino", e o
valor numérico é recalculado a cada retreino. A faixa começa em 0,20 para
impedir estruturalmente a degeneração.

### 6.3 O SVM travava a evolução

**Problema.** O experimento B parecia congelado na primeira geração por
tempo indefinido.

**Diagnóstico.** Sem `max_iter`, combinações patológicas (kernel `poly`
com C e gamma altos) fazem o SVM rodar sem convergir. O experimento B, com
mutação de 0,30 e cruzamento BLX, é justamente o que mais explora regiões
extremas do espaço de busca.

**Solução.** Teto de 200 mil iterações. O SVM devolve a melhor solução
encontrada até ali, e um indivíduo que não converge tende a ter fitness
ruim e desaparece da população.

### 6.4 O verificador automático produzia falsos positivos

Quatro rodadas de calibração, cada uma disparada por um caso real:

| Rodada | Falso positivo | Causa | Correção |
|---|---|---|---|
| 1 | "Insulina sérica" acusada de ausente | Rótulos internos sem acento | Normalização Unicode |
| 2 | 58,25 e 42 acusados de inventados | Arredondamento para 2 casas antes de derivar percentuais | Precisão de 4 casas e tolerância |
| 3 | 21 acusado de inventado | Números dentro de strings não eram extraídos | Extração também de texto |
| 4 | Percentil falso de 99,4 **não** detectado | `100 − valor` derivado também para proporções gerava 99,4175; tolerância de 2% cobria o resto | Derivação condicional e tolerância de 0,5% |

As três primeiras foram descobertas em execução real; a quarta, por um
teste automatizado que injetava um valor falso de propósito.

**Conclusão para o projeto:** verificações determinísticas são
conservadoras por natureza e tendem a acusar o que não previram. Elas
precisam ser calibradas contra saídas reais antes de serem confiáveis.

### 6.5 Nenhuma regra conferia coerência com o modelo

**Problema.** Durante um teste em modo simulado, um paciente classificado
como risco **baixo** recebeu um texto afirmando risco **elevado**, e as
seis verificações da época passaram todas.

**Diagnóstico.** Nenhuma delas comparava o veredito do texto com o do
modelo. É a inconsistência mais grave possível num laudo, e era a única
que nenhuma regra pegava.

**Solução.** Regra `classificacao_consistente`, restrita à seção RESUMO —
é legítimo a seção de limitações mencionar as duas classificações.

### 6.6 Limite de cota da API

**Problema.** Erro 429 no meio de uma execução, com perda de todo o
trabalho já gerado.

**Solução.** Três ajustes: o programa passou a salvar resultados parciais
quando a cota estoura; falha apenas na avaliação preserva a explicação
(que é o entregável principal); e o backoff passou a ler o tempo de espera
sugerido pela própria API em vez de adivinhar. Erros definitivos (modelo
inexistente, chave inválida) deixaram de ser retentados.

Migramos também do `gemini-3-flash-preview` (20 requisições diárias) para
o `gemini-3.5-flash-lite`, estável e com cota maior.

---

## 7. Arquitetura da solução

### 7.1 Três camadas

```
CAMADA 1 — OTIMIZAÇÃO (offline)
  diabetes.csv → Algoritmo Genético → modelo campeão

CAMADA 2 — INFERÊNCIA (determinística, sem rede)
  paciente → pipeline → escore → classificação + contribuições

CAMADA 3 — TRADUÇÃO (LLM, opcional)
  evidência → prompt → Gemini → texto → avaliação
```

### 7.2 API

| Rota | Função | Latência | Dependência externa |
|---|---|---|---|
| `GET /health` | Verificação de vida (Cloud Run) | ~1 ms | — |
| `GET /metrics` | Métricas de serviço e modelo | ~1 ms | — |
| `POST /predict` | Classificação + contribuições | ~5 ms | — |
| `POST /explain` | O anterior + linguagem natural | ~2 s | Gemini |

**Por que rotas separadas.** Perfis de falha diferentes. Separadas, uma
indisponibilidade do Gemini nunca derruba a triagem, que é a função
crítica. Falha do LLM retorna **503** e não 500: o problema é dependência
externa indisponível, e a distinção diz a quem consome a API que vale
tentar de novo.

### 7.3 Monitoramento

**Métricas de serviço:** requisições, taxa de erro, latência p50/p95/p99,
quebra por rota.

**Métricas de modelo:** taxa de classificações "risco elevado",
distribuição dos escores, proporção de campos ausentes.

A segunda categoria é a que costuma faltar. Um modelo pode degradar
silenciosamente sem que nada quebre: se a taxa de risco elevado sair dos
~40% observados em validação para 80%, ou a população mudou ou algo se
alterou no pré-processamento. Nenhum monitoramento de infraestrutura
detectaria isso — a latência continuaria ótima e o status continuaria 200.

**Logs:** JSON estruturado com campos `severity` e `message`, formato que
o Cloud Logging parseia automaticamente, transformando campos extras em
campos pesquisáveis. Cada requisição carrega um `request_id`.

**Limitação:** as métricas vivem na memória do processo. No Cloud Run, com
várias instâncias e scale-to-zero, cada instância tem seus próprios
contadores. Em produção, os números iriam para o Cloud Monitoring — viável
justamente porque já emitimos logs estruturados.

### 7.4 Escalabilidade

| Parâmetro | Valor | Motivo |
|---|---|---|
| `min-instances` | 0 | Sem tráfego, custo zero |
| `max-instances` | 10 | Teto de custo |
| `concurrency` | 40 | `/predict` é leve |
| `cpu` / `memory` | 1 / 512Mi | Suficiente para SVM em 5 features |
| `timeout` | 60 s | `/explain` depende do Gemini |

Um worker por container é proposital: o Cloud Run escala criando mais
instâncias, não mais processos. Vários workers multiplicariam o modelo na
memória sem ganho de vazão.

**Trade-off do scale-to-zero:** a primeira requisição após período ocioso
paga o cold start. Para triagem não emergencial, é preço aceitável em
troca de custo zero ocioso.

### 7.5 Testes

28 testes automatizados, todos passando. Os mais valiosos são os de
**regressão** — cada um corresponde a um bug que realmente aconteceu:

- `test_gene_de_corte_e_quantil_e_nao_valor_absoluto`
- `test_teto_de_iteracoes_presente`
- `test_acento_nao_causa_falso_positivo`
- `test_percentual_derivado_e_aceito`
- `test_numero_dentro_de_string_e_aceito`
- `test_contradicao_com_o_modelo_e_detectada`

Todos os outros verificam o que imaginamos que poderia quebrar; esses
verificam o que de fato quebrou.

### 7.6 Segurança

- Chave da API em variável de ambiente, nunca no código. Em produção, no
  Secret Manager.
- Container roda com usuário sem privilégios.
- Validação de faixas na entrada: o modelo não tem como saber que uma
  glicose de 5000 é erro de digitação — produziria um escore de aparência
  normal.
- Nenhum dado de paciente é persistido. Os logs registram classificação e
  escore, não os valores dos exames.

---

## 8. Limitações

1. **O ganho é de calibração, não de discriminação.** O AUC caiu nos três
   experimentos.

2. **A seleção de features é instável.** Apenas Glucose e Age aparecem nos
   três experimentos.

3. **A explicabilidade individual é frágil.** Com `C = 0,042`, a fronteira
   de decisão é quase plana e as contribuições ficam pequenas e instáveis.
   Num dos casos analisados, a glicose de 120 mg/dL apareceu *reduzindo* o
   escore — clinicamente contraintuitivo. Bom desempenho populacional não
   garante explicação local confiável.

4. **A população de referência é restrita.** Mulheres de origem Pima com
   21 anos ou mais. Os percentis não são válidos fora desse grupo.

5. **A amostra de teste é pequena.** 154 pacientes, 54 diabéticos. Os três
   experimentos separados por 3 pacientes.

6. **A avaliação por LLM é enviesada e instável.** Documentamos casos de
   notas divergentes para o mesmo conteúdo.

7. **As métricas não são agregadas entre instâncias.**

---

## 9. Conclusão

O algoritmo genético elevou o recall do modelo de triagem de 0,722 para
0,870 — de 39 para 47 diabéticos identificados a cada 54. O ganho é real
e clinicamente relevante, e veio ao custo de 4 pontos percentuais de
accuracy, um trade-off coerente com a prioridade estabelecida na Fase 1.

A análise honesta mostra que o ganho vem majoritariamente da calibração do
ponto de operação, e não de uma capacidade discriminativa superior. O
mérito do algoritmo genético foi descobrir isso sozinho, otimizando
simultaneamente hiperparâmetros, features e limiar, sem que a alavanca
principal precisasse ser conhecida de antemão.

A integração com LLM demonstrou valor concreto além do requisito formal:
num caso de falso positivo limítrofe, a explicação gerada sinalizou
espontaneamente a fragilidade do resultado — informação que o rótulo
binário não carregava.

Por fim, o achado metodológico mais transferível deste trabalho é que
**quase todos os problemas sérios apareceram apenas na execução.** A
função fitness parecia razoável no papel. O limiar absoluto parecia
correto. As verificações automáticas pareciam completas. Em todos os
casos, foi rodar contra dados reais que revelou o problema — e, em três
ocasiões, foi um teste ou uma camada de avaliação independente que
apontou o erro de outra.

---

## Apêndice A — Como reproduzir

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Algoritmo genético (os três experimentos)
python run_experiments.py --dados data/diabetes.csv

# Interpretação por LLM
$env:GEMINI_API_KEY="sua-chave"
$env:GEMINI_MODELO="gemini-3.5-flash-lite"
python explicar_pacientes.py --n 3

# Testes
python -m pytest -v

# API
uvicorn api.main:app --port 8000   # http://localhost:8000/docs
```

## Apêndice B — Estrutura do repositório

```
fase2/
├── src/
│   ├── data.py                  carga, split, pipeline
│   ├── modelo_final.py          configuração do modelo campeão
│   ├── genetic/                 cromossomo, operadores, fitness, motor
│   ├── llm/                     cliente, prompts, explicador, avaliação
│   └── monitoring/              logs estruturados, métricas
├── api/                         FastAPI
├── tests/                       28 testes automatizados
├── docs/arquitetura.md          diagramas e decisões
├── terraform/main.tf            IaC para Cloud Run
├── Dockerfile
├── run_experiments.py
├── explicar_pacientes.py
└── listar_modelos.py
```
