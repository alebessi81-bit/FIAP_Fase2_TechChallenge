# Tech Challenge - Fase 3: Pipeline de Fine-Tuning Biomédico (MedQuAD via Hugging Face)

Este projeto foi desenvolvido como parte do **Tech Challenge (Fase 3)** da pós-graduação em **IA para Devs (FIAP)**.

O objetivo principal é conectar diretamente ao dataset médico **MedQuAD** hospedado no **Hugging Face** ([`lavita/MedQuAD`](https://huggingface.co/datasets/lavita/MedQuAD)) utilizando a biblioteca oficial `datasets`, filtrar os pares de perguntas e respostas médicas, higienizar os textos e estruturá-los no formato **JSONL** (*Chat Completion / Messages*), preparando-os para o **fine-tuning** de modelos de linguagem base (LLMs) como GPT-4o-mini, Llama ou Mistral.

---

## 📋 Sumário
- [Visão Geral e Formato de Saída](#-visão-geral-e-formato-de-saída)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Arquitetura e Boas Práticas (SOLID & Clean Code)](#-arquitetura-e-boas-práticas-solid--clean-code)
- [Integração com Hugging Face e Métricas](#-integração-com-hugging-face-e-métricas)
- [Configuração de Ambiente (.env)](#-configuração-de-ambiente-env)
- [Instalação](#-instalação)
- [Como Executar o Pipeline](#-como-executar-o-pipeline)
- [Execução dos Testes Automatizados](#-execução-dos-testes-automatizados)

---

## 🎯 Visão Geral e Formato de Saída

Cada linha dos arquivos gerados (`finetuning_train.jsonl` e `finetuning_val.jsonl`) contém estritamente um objeto JSON com o formato padrão para fine-tuning conversacional:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a biomedical evidence assistant for healthcare professionals. Answer using the supplied scientific evidence. Distinguish findings from interpretation, state limitations, and do not invent evidence. You are not intended to diagnose, but to give assistance. The final word is always from a professional"
    },
    {
      "role": "user",
      "content": "What are the treatments for Frontotemporal Dementia ?"
    },
    {
      "role": "assistant",
      "content": "No treatment has been shown to slow the progression of FTD. Behavior modification may help control unacceptable or dangerous behaviors..."
    }
  ]
}
```

### Papel das Partições:
- **`finetuning_train.jsonl` (90%):** Utilizado para o ajuste de pesos da rede neural durante o fine-tuning.
- **`finetuning_val.jsonl` (10%):** Utilizado para cálculo de perda de validação (*validation loss*), prevenção de *overfitting* e testes de generalização com perguntas nunca vistas no treino.

---

## 📂 Estrutura do Projeto

```text
TCv1/
├── data/
│   └── processed/                   # Arquivos gerados pelo pipeline
│       ├── finetuning_train.jsonl   # Partição de treino (14.766 amostras)
│       ├── finetuning_val.jsonl     # Partição de validação (1.641 amostras)
│       └── discarded.json           # Registros descartados na limpeza (31.034 amostras)
├── logs/
│   └── execucao.log                 # Histórico persistente de todas as execuções do logger
├── src/
│   ├── __init__.py
│   ├── config.py                    # Constantes globais, caminhos com Pathlib, System Prompt e leitura de .env
│   ├── data_loader.py               # Conexão com Hugging Face (datasets), extração, higienização e descarte
│   ├── jsonl_converter.py           # Modelagem de dados, conversão, split e validação sintática
│   ├── Process_dataset.py           # Script principal que orquestra todo o pipeline
│   └── utils/
│       ├── __init__.py
│       └── logger.py                # Configuração centralizada de logging estruturado (console e arquivo)
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py          # Testes unitários com mocks para API Hugging Face
│   └── test_jsonl_converter.py      # Testes unitários para conversão, split e integridade JSONL
├── .env                             # Arquivo de configuração de credenciais (HUGGINGFACE_API_KEY)
├── agents.md                        # Diretrizes de arquitetura, Clean Code e backlog
├── requirements.txt                 # Dependências do projeto (datasets, python-dotenv, pandas, pytest)
└── README.md                        # Documentação completa do projeto
```

---

## 🏛 Arquitetura e Boas Práticas (SOLID & Clean Code)

O código foi projetado seguindo estritamente os princípios de engenharia de software especificados no arquivo `agents.md`:

- **Single Responsibility Principle (SRP):**
  - `data_loader.py`: Cuida exclusivamente da conexão com a API do Hugging Face, extração das colunas `question` e `answer`, e descarte de dados nulos/vazios.
  - `jsonl_converter.py`: Responsável apenas pela conversão de formatos, particionamento e validação de saída.
  - `Process_dataset.py`: Apenas coordena as etapas do pipeline.
- **Dependency Inversion & Injeção de Dependências:**
  - Classes e funções recebem parâmetros configuráveis (como `dataset_name`, `token`, `system_prompt`, `train_ratio`, `seed`) em vez de valores fixos (*hardcoded*).
- **Manipulação Segura de Caminhos:**
  - Todo o I/O utiliza a biblioteca nativa `pathlib.Path`, garantindo compatibilidade multiplataforma (Windows, Linux e macOS).
- **Clean Code & Tipagem Estática:**
  - 100% das funções e classes contam com *Type Hints* completos (`typing`) e docstrings no padrão Google Style.
  - Uso de logging estruturado (`logging`) em substituição a chamadas diretas de `print()`.
  - Tratamento explícito de exceções (`FileNotFoundError`, `ValueError`, `json.JSONDecodeError`, `RuntimeError`).

---

## 🌐 Integração com Hugging Face e Métricas

O dataset remoto [`lavita/MedQuAD`](https://huggingface.co/datasets/lavita/MedQuAD) possui 47.441 linhas e 13 colunas (`document_id`, `question`, `answer`, `umls_cui`, etc.). Muitas linhas do dataset original representam documentos sem anotação de resposta médica completa (`answer` nulo ou vazio). O pipeline filtra os dados e mantém apenas os pares de pergunta e resposta clinicamente consistentes.

| Métrica | Valor |
| :--- | :--- |
| **Dataset Remoto** | `lavita/MedQuAD` (split `train`) |
| **Registros brutos baixados da API** | 47.441 |
| **Registros descartados (sem resposta ou nulos)** | 31.034 |
| **Registros válidos com pergunta e resposta** | 16.407 |
| **Conjunto de Treino (90%)** | 14.766 amostras |
| **Conjunto de Validação (10%)** | 1.641 amostras |
| **Integridade dos arquivos gerados** | 100% em conformidade com o schema JSON |

---

## 🔑 Configuração de Ambiente (.env)

Crie ou edite o arquivo `.env` na raiz do projeto com o seu token de acesso do Hugging Face:

```env
HUGGINGFACE_API_KEY=seu_token_aqui
```

*(O pipeline também reconhece a variável `HF_TOKEN`, caso já esteja definida em seu ambiente).*

---

## 🚀 Instalação

1. Clone ou acesse a pasta raiz do projeto:
   ```bash
   cd "d:\FIAP-IA_para_Devs\FASE 3\TechChallenge\TCv1"
   ```

2. *(Opcional, mas recomendado)* Crie e ative um ambiente virtual:
   ```bash
   # Criar ambiente virtual
   python -m venv venv

   # Ativar no Windows (PowerShell)
   .\venv\Scripts\Activate.ps1

   # Ativar no Linux/macOS
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Como Executar o Pipeline

Para rodar o processo completo de conexão ao Hugging Face, download, higienização, conversão para JSONL, divisão treino/validação e validação sintática:

```bash
python src/Process_dataset.py
```

### O que o pipeline executa:
1. Conecta à API do Hugging Face e obtém o dataset `lavita/MedQuAD` autenticado pelo token do `.env`.
2. Filtra e extrai apenas as colunas essenciais `question` e `answer`.
3. Remove registros onde a resposta ou a pergunta são nulas ou contêm apenas espaços vazios.
4. Converte os dados para a estrutura de mensagens do ChatML com o prompt de sistema biomédico.
5. Separa aleatoriamente os registros (90% treino / 10% validação) com semente determinística (`seed=42`).
6. Exporta os arquivos para a pasta `data/processed/` com codificação UTF-8 (`ensure_ascii=False`).
7. Valida linha por linha a integridade estrutural e de parsing dos arquivos gerados.

---

## 🧪 Execução dos Testes Automatizados

O projeto conta com testes unitários que utilizam *mocks* para a API do Hugging Face, permitindo testes rápidos e offline.

### Opção 1: Via Pytest
```bash
pytest -v
```

### Opção 2: Via Unittest (módulo nativo do Python)
```bash
python -m unittest discover tests -v
```

### Cobertura dos Testes:
- Conexão e extração de colunas com mock da biblioteca `datasets`.
- Tratamento de datasets vazios ou sem as colunas obrigatórias.
- Tratamento de falhas de rede/autenticação.
- Limpeza e remoção de registros nulos e espaços vazios.
- Estruturação correta dos papéis (`system`, `user`, `assistant`).
- Reprodutibilidade da separação treino/validação por semente.
- Detecção de arquivos corrompidos ou com esquemas inválidos.
