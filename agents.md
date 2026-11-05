# Diretrizes de Desenvolvimento e Escopo do Projeto - Agents.md

Este documento define o escopo funcional, os requisitos de entrega, a arquitetura e os padrões de qualidade em Python (Clean Code e SOLID) a serem seguidos estritamente por agentes de IA e desenvolvedores neste projeto.

---

## 1. Visão Geral e Escopo do Projeto

### 1.1 Objetivo
Conectar e extrair os dados de perguntas e respostas médicas do dataset MedQuAD hospedado no **Hugging Face** (`lavita/MedQuAD`) utilizando a biblioteca oficial `datasets` e token de autenticação via `.env`. Em seguida, higienizar e estruturar os registros no formato **JSONL** (JSON Lines), pronto para ser consumido no **fine-tuning** de modelos base de linguagem (LLMs).

### 1.2 Formato Alvo (Chat Completion Fine-Tuning)
Cada linha do arquivo `.jsonl` gerado deve conter estritamente um objeto JSON com a seguinte estrutura:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a biomedical evidence assistant for healthcare professionals. Answer using the supplied scientific evidence. Distinguish findings from interpretation, state limitations, and do not invent evidence. You are not intended to diagnose, but to give assistance. The final word is always from a professional"
    },
    {
      "role": "user",
      "content": "<Question>"
    },
    {
      "role": "assistant",
      "content": "<Answer>"
    }
  ]
}
```

### 1.3 Estrutura Alvo de Diretórios
```text
TCv1/
├── data/
│   └── processed/                   # Destino dos arquivos gerados (finetuning_train.jsonl, finetuning_val.jsonl, discarded.json)
├── logs/
│   └── execucao.log                 # Registro persistente de execuções do logger
├── src/
│   ├── __init__.py
│   ├── config.py                    # Configurações globais (SYSTEM_PROMPT, HF token, dataset_name, caminhos com pathlib.Path)
│   ├── data_loader.py               # Conexão com Hugging Face via datasets, extração, filtragem e descarte
│   ├── jsonl_converter.py           # Conversão das mensagens para o formato de fine-tuning e validação
│   ├── Process_dataset.py           # Pipeline orquestrador de ponta a ponta
│   └── utils/
│       ├── __init__.py
│       └── logger.py                # Configuração de logging estruturado (console e arquivo)
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py          # Testes unitários para carga do Hugging Face e limpeza de dados
│   └── test_jsonl_converter.py      # Testes de integridade do formato JSONL gerado
├── .env                             # Credenciais e chaves de API (HUGGINGFACE_API_KEY)
├── agents.md                        # Documento de escopo e diretrizes
├── README.md                        # Documentação detalhada de execução
└── requirements.txt                 # Dependências do projeto (datasets, python-dotenv, pandas, pytest)
```

### 1.4 Backlog de Desenvolvimento (Roadmap)
- [x] **Fase 1: Configuração e Modelagem**
  - [x] Criar `src/config.py` centralizando constantes (`SYSTEM_PROMPT`, `HUGGINGFACE_DATASET_NAME`, leitura de token via `python-dotenv`).
  - [x] Definir modelos de dados com `dataclasses` para representar as mensagens e o schema de exportação.
- [x] **Fase 2: Conexão com Hugging Face e Limpeza dos Dados**
  - [x] Conectar com `lavita/MedQuAD` no Hugging Face utilizando `datasets.load_dataset`.
  - [x] Tratar valores nulos, registros sem resposta (`None` ou strings vazias) e inconsistências de texto em `question` e `answer`.
  - [x] Garantir sanitização de caracteres especiais para evitar JSON malformado.
- [x] **Fase 3: Exportação para JSONL e Separação (Train / Val)**
  - [x] Gerar linhas JSON válidas com encoding UTF-8 (`ensure_ascii=False`).
  - [x] Implementar divisão estratificada em treino e validação (90% treino, 10% validação).
  - [x] Salvar arquivos em `data/processed/`.
- [x] **Fase 4: Testes Automatizados e Validação**
  - [x] Criar validação do arquivo final gerado (garantir que 100% das linhas possuem `messages`, `system`, `user`, `assistant`).
  - [x] Escrever testes unitários em `tests/` com mocks para requisições externas e validações completas.

---

## 2. Princípios SOLID em Python

### 2.1 Single Responsibility Principle (SRP) — Responsabilidade Única
- Cada módulo, classe ou função deve ter **uma, e apenas uma, razão para mudar**.
- Separe tarefas de I/O (leitura de arquivos), validação/limpeza de dados e serialização JSON em componentes distintos.
- **Anti-padrão:** Um único script que lê o CSV, trata nulos, formata strings e grava o JSONL tudo junto.

### 2.2 Open/Closed Principle (OCP) — Aberto para Extensão, Fechado para Modificação
- Entidades de software devem ser abertas para extensão, mas fechadas para modificação.
- Interfaces e classes devem permitir novos formatos de saída ou novas fontes de dados sem alterar o código existente.

### 2.3 Liskov Substitution Principle (LSP) — Substituição de Liskov
- Subclasses devem poder substituir suas classes base sem quebrar o comportamento do programa.
- Respeite assinaturas e tipos de retorno esperados.

### 2.4 Interface Segregation Principle (ISP) — Segregação de Interfaces
- Prefira interfaces enxutas usando `typing.Protocol` ou classes abstratas (`abc.ABC`).
- Não force um módulo a depender de métodos que ele não usa (ex: leitor de dados não precisa conhecer serializador JSONL).

### 2.5 Dependency Inversion Principle (DIP) — Inversão de Dependência
- Módulos de alto nível devem depender de abstrações, não de implementações concretas.
- Utilize injeção de dependências: passe configurações, caminhos e transformadores por injeção nos construtores/funções.

---

## 3. Clean Code em Python

### 3.1 PEP 8 e Nomenclatura
- **Variáveis e Funções:** `snake_case` (ex: `convert_to_jsonl`, `cleaned_text`).
- **Classes:** `PascalCase` (ex: `MedQuadProcessor`, `JsonlWriter`).
- **Constantes:** `UPPER_SNAKE_CASE` (ex: `SYSTEM_PROMPT`, `DEFAULT_TRAIN_RATIO`).
- **Nomes Intencionais:** Evite abreviações (`df`, `temp`). Use nomes descritivos (`medquad_df`, `chat_messages`).

### 3.2 Funções Pequenas e Focadas
- Funções devem fazer apenas **uma coisa**.
- Limite o número de argumentos (máximo 3; use `dataclasses` se precisar de mais).
- Evite efeitos colaterais (*side effects*) e mutações inesperadas.

### 3.3 Tipagem Estática (Type Hints)
- Adicione *type hints* em todas as assinaturas de funções e métodos públicos (`def convert(df: pd.DataFrame) -> List[Dict[str, Any]]:`).
- Utilize o módulo `typing` (`Optional`, `List`, `Dict`, `Union`, `Tuple`).

### 3.4 Documentação e Docstrings
- Escreva docstrings no padrão **Google Style** para módulos, classes e funções públicas.

### 3.5 Tratamento de Exceções
- Evite `except Exception:` ou `except: pass`.
- Capture exceções específicas (`FileNotFoundError`, `ValueError`, `json.JSONDecodeError`).

### 3.6 Logging Estruturado
- Não utilize `print()`. Use o módulo padrão `logging` (`logger.info(...)`, `logger.error(...)`).

---

## 4. Boas Práticas para Manipulação de Dados

### 4.1 Manipulação de Caminhos
- Use sempre `pathlib.Path` para garantir compatibilidade multiplataforma (Windows/Linux/macOS).
- Nunca use caminhos absolutos (*hardcoded*).

### 4.2 Manipulação de DataFrames
- Trate cópias explícitas com `.copy()` ao filtrar DataFrames para evitar `SettingWithCopyWarning`.
- Remova valores `NaN` ou nulos nas colunas essenciais (`question`, `answer`) antes de processar.

---

## 5. Testes e Qualidade de Código

- **Testes Automatizados:** Implemente testes com `pytest`.
- **Testes Críticos:**
  - Validação de que nenhuma linha do JSONL resultante possui campos vazios.
  - Verificação de parsing correto do JSON em cada linha gerada.
  - Teste de consistência do prompt de sistema.

---

## 6. Checklist de Validação

Antes de considerar qualquer entrega finalizada, certifique-se de:
1. [x] A alteração respeita a responsabilidade única de cada arquivo?
2. [x] Foram adicionados *Type Hints* em todas as funções?
3. [x] Caminhos utilizam `pathlib.Path`?
4. [x] O código utiliza `logging` em vez de `print()`?
5. [x] O JSONL gerado foi validado com `json.loads()` linha a linha sem erros?
6. [x] Foram implementados testes unitários com `pytest` / `unittest`?
