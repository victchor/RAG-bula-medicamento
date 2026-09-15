# Como rodar o projeto

Guia completo para colocar o projeto no ar pela primeira vez, do zero.

## Pré-requisitos

- **Python 3.12** instalado. Versões mais novas (3.13+) podem falhar ao instalar dependências de
  machine learning (`torch`, usado por `sentence-transformers`), porque essas bibliotecas
  costumam demorar a publicar builds para versões recentes do Python.
  Verifique com:
  ```bash
  py -0p
  ```
  Se `3.12` não aparecer na lista, instale-o antes de continuar (python.org ou Microsoft Store).
- Cerca de **2GB livres em disco** — a maior parte é o `torch`, usado pelos modelos locais de
  embeddings e reranking.
- Uma chave de API — **Google Gemini (gratuita)** ou **Anthropic Claude (paga)**. Veja a seção
  abaixo antes de decidir.

## Gemini (gratuito) vs. Claude (pago) — qual escolher

O projeto suporta os dois, trocáveis por uma variável de ambiente. Não há necessidade de decidir
"para sempre" — dá para começar com um e trocar depois.

| | Gemini | Claude |
|---|---|---|
| Custo | Camada gratuita, sem cartão de crédito | Pago por token (ainda que muito barato — veja `docs/ETAPAS.md`, Etapa 14) |
| Onde obter a chave | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| Limitação | Cota diária por modelo (pode esgotar em uso intenso) | Exige adicionar saldo pré-pago na conta |
| Recomendação | **Comece por aqui** para testar sem gastar nada | Use depois, se quiser validar com o modelo originalmente visado pelo projeto |

**Atenção a uma pegadinha real**: uma assinatura Claude Pro/Max (a mesma usada no Claude Code, por
exemplo) **não** dá acesso à API paga da Anthropic — são dois produtos de cobrança separados,
mesmo sendo a mesma empresa. Ter uma assinatura Claude não significa ter crédito de API.

## Passo a passo

### 1. Obter o projeto

Baixe ou clone os arquivos deste repositório para uma pasta local.

### 2. Criar e ativar o ambiente virtual

```bash
py -3.12 -m venv .venv
```

Ativar (Windows, PowerShell ou Git Bash):
```bash
.venv\Scripts\activate
```

Ativar (Linux/macOS):
```bash
source .venv/bin/activate
```

O prompt do terminal deve passar a mostrar `(.venv)` no início da linha.

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

Isso baixa ~2GB (a maior parte é `torch`, para os modelos locais de embeddings/reranking). Pode
levar alguns minutos, dependendo da conexão.

### 4. Configurar as variáveis de ambiente

```bash
copy .env.example .env
```
(Linux/macOS: `cp .env.example .env`)

Abra o `.env` em um editor de texto e:

1. Deixe `LLM_PROVIDER=google` (padrão, gratuito) ou troque para `LLM_PROVIDER=anthropic`.
2. Preencha a chave correspondente:
   - `GOOGLE_API_KEY=` — cole a chave gerada em aistudio.google.com/apikey
   - `ANTHROPIC_API_KEY=` — cole a chave gerada em console.anthropic.com (se for usar Claude)

Nunca compartilhe ou publique esse arquivo — ele contém sua chave real e já está no `.gitignore`.

### 5. Testar a conexão com a LLM

```bash
python -m scripts.test_llm_connection
```

Se aparecer uma resposta da LLM e a contagem de tokens, a configuração está correta. Erros comuns
nesta etapa:

| Erro | Causa | Solução |
|---|---|---|
| `ANTHROPIC_API_KEY não foi definida` / `GOOGLE_API_KEY não foi definida` | Chave não preenchida no `.env` para o provedor ativo | Preencha a chave correspondente ao `LLM_PROVIDER` escolhido |
| `credit balance is too low` | Conta Anthropic sem saldo | Adicione crédito em console.anthropic.com/settings/billing (não é o mesmo que assinatura Claude Pro) |
| `RESOURCE_EXHAUSTED` / `429` no Gemini | Cota diária gratuita do modelo esgotada | Aguarde o reset diário, ou troque `GEMINI_MODEL` no `.env` por outro modelo |

### 6. Rodar a aplicação

```bash
streamlit run app/ui/streamlit_app.py
```

Isso abre automaticamente `http://localhost:8501` no navegador (ou mostra o link no terminal, se
não abrir sozinho).

Aviso inofensivo: o terminal pode mostrar vários `ModuleNotFoundError: No module named 'torchvision'`
— é um comportamento conhecido do verificador de arquivos do Streamlit ao inspecionar a biblioteca
`transformers`, sem relação com o funcionamento da aplicação. Pode ignorar.

### 7. Primeiro uso

1. Na barra lateral, clique em **"Browse files"** e selecione um PDF de bula de medicamento.
2. Clique em **"Processar bula"** — aguarde o processamento (leitura do PDF, geração de embeddings,
   indexação).
3. No campo de pergunta, digite algo como *"Para que serve este medicamento?"*.
4. A resposta aparece com um expansor **"Fontes utilizadas"**, mostrando de qual página da bula
   veio a informação.

## Testando por linha de comando (alternativa ao Streamlit)

Depois de processar uma bula pelo menos uma vez pela interface (para criar o índice), é possível
conversar direto pelo terminal:

```bash
python -m scripts.ask
```

Digite `sair` para encerrar.
