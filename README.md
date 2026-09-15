# Chatbot RAG para Bulas de Medicamentos

Este projeto é o resultado de um estudo prático de RAG (Retrieval-Augmented Generation) — a
técnica que permite a um modelo de linguagem responder perguntas com base em documentos
específicos, em vez de depender só do que aprendeu em treinamento.

É um RAG simples, com o objetivo de auxiliar pessoas a consultar o conteúdo de uma bula de
medicamento em PDF por meio de perguntas em linguagem natural, em vez de precisar ler o texto
denso e técnico do documento inteiro. O usuário envia o PDF de uma bula, e o sistema responde
perguntas sobre ela citando a página de origem de cada informação.

> **Este é um projeto de estudo, não uma ferramenta médica.** Não deve ser usado para decisões
> reais sobre medicamentos, e não substitui orientação de um médico ou farmacêutico. Veja
> [docs/LIMITACOES_E_FUTURO.md](docs/LIMITACOES_E_FUTURO.md) para o aviso completo.

```text
PDF da bula → texto → chunks → embeddings → banco vetorial
                                                    │
Pergunta do usuário → busca semântica → reranking → contexto → LLM → resposta
```

## Documentação

| Documento | Conteúdo |
|---|---|
| [docs/COMO_RODAR.md](docs/COMO_RODAR.md) | Passo a passo completo de instalação e primeira execução, incluindo a escolha entre chave Gemini (gratuita) e Claude (paga) |
| [docs/COMO_FUNCIONA.md](docs/COMO_FUNCIONA.md) | Como o sistema funciona por dentro, explicado em 3 cenários de uso: indexação de um arquivo, primeira pergunta, e pergunta de acompanhamento |
| [docs/LIMITACOES_E_FUTURO.md](docs/LIMITACOES_E_FUTURO.md) | Limitações técnicas, riscos de segurança e de confiabilidade conhecidos, e o que seria necessário para evoluir isto para um uso real |
| [docs/ETAPAS.md](docs/ETAPAS.md) | Histórico completo de desenvolvimento: cada decisão de arquitetura, tecnologia escolhida e por quê, incluindo bugs reais encontrados e corrigidos pelo caminho |

## Início rápido

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edite o `.env` com sua chave de API (veja [docs/COMO_RODAR.md](docs/COMO_RODAR.md) para a escolha
entre Gemini gratuito e Claude pago), depois:

```bash
streamlit run app/ui/streamlit_app.py
```

## Estrutura do projeto

```text
app/
├── config.py            # configuração via variáveis de ambiente
├── llm.py                # fábrica do modelo de chat (Claude ou Gemini)
├── ingestion/             # PDF → chunks → embeddings → vector store
├── rag/                    # retriever, reranker, prompts, chain de resposta
├── evaluation/              # avaliação sistemática (Hit Rate + LLM como juiz)
├── observability/            # métricas de tempo, tokens, custo por pergunta
├── security/                  # validação e sanitização de upload
└── ui/                         # interface Streamlit

data/
├── uploads/          # PDFs enviados (não versionado)
└── vectorstore/      # índice vetorial persistido (não versionado)

scripts/               # ferramentas de teste, inspeção e avaliação por linha de comando
docs/                   # documentação (ver tabela acima)
```
