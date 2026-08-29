# Fase 2 — Núcleo do Algoritmo Genético

Otimização por algoritmo genético do SVM entregue na Fase 1
(Tech Challenge PosTech FIAP — IA para Devs).

## Estrutura

```
src/
├── data.py                  # carga, split e pipeline (scaler → imputer → SVM)
└── genetic/
    ├── chromosome.py        # codificação e decodificação dos 13 genes
    ├── operators.py         # seleção, cruzamento, mutação, elitismo
    ├── fitness.py           # avaliação por validação cruzada estratificada
    └── engine.py            # laço evolutivo
run_experiments.py           # baseline + os 3 experimentos obrigatórios
```

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# coloque diabetes.csv em data/
python run_experiments.py --dados data/diabetes.csv
```

Para rodar um único experimento:

```bash
python run_experiments.py --experimentos A
```

## Saídas

Na pasta `resultados/`:

| Arquivo | Conteúdo |
|---|---|
| `historico_exp_A.csv` | fitness por geração (curva de convergência) |
| `comparativo.csv` | baseline vs. modelos otimizados no conjunto de teste |
| `melhores_individuos.json` | hiperparâmetros e features do melhor indivíduo |

## Integração com LLM

```bash
$env:GEMINI_API_KEY="sua-chave"
$env:GEMINI_MODELO="gemini-3.5-flash-lite"

python listar_modelos.py                  # modelos disponíveis na sua chave
python explicar_pacientes.py --simular    # testa sem gastar cota
python explicar_pacientes.py --n 3        # execução real
```

Flags úteis: `--sem-juiz` corta pela metade as chamadas à API;
`--n` define quantos pacientes explicar.

A chave nunca deve ir para o código nem para o Git — use variável de
ambiente, que expira quando a janela do terminal fecha.

## Nota metodológica

O conjunto de teste não é usado em momento algum durante a evolução.
Todo o fitness é calculado por validação cruzada estratificada de 5 folds
sobre o conjunto de treino, com o scaler e o imputer reajustados dentro
de cada fold.
