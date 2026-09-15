# Como o projeto funciona — explicado em 3 cenários

Este documento explica o sistema do jeito que você entenderia se estivesse aprendendo a construir
um RAG pela primeira vez: cada cenário primeiro explica o **conceito**, depois mostra o **código**
responsável, na ordem real em que ele executa.

Se você nunca mexeu com RAG antes, uma distinção vale ser fixada desde já, porque se repete o
tempo todo neste projeto:

```text
get_algo()      → MONTA uma ferramenta (modelo, conexão, configuração). Não processa nada.
algo.usar(...)  → USA essa ferramenta de fato, com um dado de entrada específico.
```

Carregar o modelo de embeddings, por exemplo, não gera vetor nenhum — só deixa o modelo pronto na
memória. O vetor só é calculado quando, mais tarde, alguém chama esse modelo com um texto
específico. Essa separação (montar vs. usar) é o que permite reaproveitar as mesmas ferramentas em
milhares de perguntas sem recriá-las a cada vez.

---

## Visão geral

```text
                    ┌─────────────────┐
                    │     Usuário     │
                    └────────┬────────┘
                             │ upload do PDF
                             ▼
              ══════ CENÁRIO 1: INDEXAÇÃO ══════
                    ┌─────────────────┐
                    │  Leitura do PDF │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │    Chunking     │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   Embeddings    │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Banco Vetorial  │
                    └─────────────────┘

              ══════ CENÁRIO 2/3: PERGUNTA ══════
                    ┌─────────────────┐
                    │     Pergunta     │
                    └────────┬────────┘
                             ▼
                 (com histórico? reformula)
                             ▼
                    ┌─────────────────┐
                    │    Retriever     │  → candidatos
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │    Reranker      │  → os melhores
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Prompt + LLM    │
                    └────────┬────────┘
                             ▼
                        Resposta
```

---

## Cenário 1 — Indexação de um arquivo

### Conceito

Antes de responder qualquer pergunta, o sistema precisa "aprender" o conteúdo da bula. Isso
acontece **uma vez por arquivo**, não a cada pergunta — é a fase mais cara computacionalmente
(embora ainda gratuita neste projeto, por rodar tudo localmente), mas a mais rara.

O processo transforma um PDF em algo que pode ser **buscado por significado**:

1. **Extrair o texto** do PDF, página por página.
2. **Dividir em pedaços menores** (chunks) — um PDF inteiro é grande demais para caber
   eficientemente no contexto de uma LLM, e chunks menores permitem recuperar só o trecho
   relevante para cada pergunta específica.
3. **Transformar cada chunk em um vetor** (embedding) — uma lista de números que representa o
   *significado* daquele texto, não as palavras exatas.
4. **Guardar tudo num banco vetorial**, pronto para busca por similaridade.

### Código responsável

**Upload** (`app/ui/streamlit_app.py`, função `processar_bula`) recebe o arquivo, valida (tamanho,
assinatura de PDF real, nome seguro) e salva em `data/uploads/`.

**Leitura do PDF** (`app/ingestion/pdf_loader.py`):
```python
def load_pdf(pdf_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()
```
Devolve um `Document` por página — cada um com o texto e a página de origem em metadata.

**Chunking** (`app/ingestion/splitter.py`):
```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, ...)
    return splitter.split_documents(documents)
```
Cada página vira vários chunks de até 1000 caracteres. O `overlap` de 150 caracteres repete o fim
de um chunk no início do próximo, para não cortar uma frase importante bem na fronteira entre dois
pedaços.

**Embeddings** (`app/ingestion/embeddings.py`):
```python
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
```
Isso só monta o modelo — nenhum vetor é calculado aqui.

**Banco vetorial** (`app/ingestion/vectorstore.py`):
```python
def create_new_vectorstore(documents, embedding, persist_directory, ...):
    ...
    return Chroma.from_documents(documents=documents, embedding=embedding, ...)
```
É **aqui**, dentro de `Chroma.from_documents` (biblioteca `langchain_chroma`, não código deste
projeto), que os vetores são de fato calculados: a função extrai o texto de cada chunk e chama
`embedding.embed_documents(textos)` internamente, guardando vetor + texto + página juntos.

Ao final do Cenário 1, existe um índice persistido em disco (`data/vectorstore/`) pronto para
busca — e a aplicação está pronta para receber perguntas.

---

## Cenário 2 — Primeira pergunta

### Conceito

Com a bula indexada, o usuário faz a primeira pergunta da conversa. Sem histórico prévio, o fluxo é
direto:

1. A pergunta vira um vetor (mesmo processo do chunk, mas para um texto só).
2. O banco vetorial devolve os pedaços de texto mais próximos matematicamente desse vetor —
   candidatos a conter a resposta.
3. Um segundo modelo, mais lento e mais criterioso, reordena esses candidatos comparando cada um
   lado a lado com a pergunta, e escolhe os melhores.
4. Os melhores trechos viram o "contexto" de um prompt, junto com regras de comportamento fixas, e
   são enviados à LLM.
5. A LLM lê tudo isso e escreve a resposta — sem inventar nada além do que está no contexto.

### Código responsável

**Retriever** (`app/rag/retriever.py`):
```python
K_CANDIDATOS = 15

def get_retriever(vectorstore: Chroma, k: int = K_CANDIDATOS) -> BaseRetriever:
    return vectorstore.as_retriever(search_kwargs={"k": k})
```
Um envelope fino sobre o `Chroma`, configurado para devolver 15 candidatos por busca. Buscar 15
(não 4) dá mais material para a próxima etapa escolher com mais precisão.

**Busca real e reranking** (`app/rag/chain.py`, dentro de `get_rag_chain`):
```python
def buscar_e_rerankear(entrada: dict) -> list[Document]:
    pergunta_busca = obter_pergunta_para_busca(entrada)   # sem histórico, é a pergunta original
    candidatos = retriever.invoke(pergunta_busca)          # aqui a pergunta vira vetor e é comparada
    return rerank(reranker, pergunta_busca, candidatos, k_final=k_final)
```
`retriever.invoke(pergunta_busca)` transforma a pergunta em vetor (`embed_query`), compara com os
vetores já indexados, e devolve os 15 chunks mais próximos.

**Reranking** (`app/rag/reranker.py`):
```python
def rerank(reranker: CrossEncoder, pergunta: str, documentos: list[Document], k_final: int) -> list[Document]:
    pares = [(pergunta, doc.page_content) for doc in documentos]
    scores = reranker.predict(pares)
    documentos_por_score = sorted(zip(scores, documentos), key=lambda par: par[0], reverse=True)
    return [doc for _, doc in documentos_por_score[:k_final]]
```
Diferença crucial em relação ao embedding: aqui a pergunta e cada chunk entram **juntos** no
modelo (não separadamente), permitindo uma comparação mais precisa — ao custo de ser mais lento,
por isso só é aplicado aos 15 candidatos, não a todos os chunks indexados. Sobram 4.

**Prompt e geração** (`app/rag/chain.py` + `app/rag/prompts.py`):
```python
gerar_resposta = (
    RunnableParallel(
        context=lambda etapa1: format_docs(etapa1["source_documents"]),
        question=lambda etapa1: etapa1["question"],
        chat_history=lambda etapa1: etapa1["chat_history"],
    )
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```
`format_docs` junta os 4 chunks finais numa string única, marcando a página de cada um.
`RAG_PROMPT` insere essa string, a pergunta e o histórico (vazio, neste cenário) num template com
regras fixas de comportamento — a mais importante sendo "responda só com base no contexto
fornecido, nunca invente". `llm` é a chamada real à API (Claude ou Gemini). `StrOutputParser`
normaliza a resposta para texto puro.

A chain devolve `{"answer": ..., "source_documents": [...]}` — a resposta e os mesmos 4 chunks que
a geraram, usados depois para exibir as fontes na interface.

---

## Cenário 3 — Segunda pergunta (com histórico)

### Conceito

Aqui está a diferença central em relação ao Cenário 2: a pergunta pode ser **elíptica** — depender
do que já foi dito antes. Um exemplo típico:

```text
Pergunta 1: "O que é este medicamento?"
Pergunta 2: "E quais são os efeitos colaterais?"
```

Sozinha, "e quais são os efeitos colaterais" não diz *de quê* — não tem sujeito. Se essa string
fosse direto para a busca vetorial, o resultado seria pior do que poderia ser. A solução: antes de
buscar, uma chamada extra à LLM **reescreve** a pergunta usando o histórico, transformando-a em algo
como "quais são os efeitos colaterais do paracetamol?" — só essa versão reescrita vai para a busca.
A resposta final, porém, continua sendo gerada em cima da pergunta **original** do usuário, para
soar natural.

Isso tem um custo real: uma pergunta de acompanhamento gasta **duas** chamadas de LLM (reformular +
responder) em vez de uma.

### Código responsável

A diferença aparece em `obter_pergunta_para_busca` (`app/rag/chain.py`):
```python
def obter_pergunta_para_busca(entrada: dict) -> str:
    if not entrada.get("chat_history"):
        return entrada["question"]              # Cenário 2: sem histórico, usa a pergunta direto
    return condensar_pergunta.invoke(entrada)    # Cenário 3: reformula usando o histórico
```

`condensar_pergunta` é outra mini-chain, com seu próprio prompt (`CONDENSE_QUESTION_PROMPT`, em
`app/rag/prompts.py`), cujo único trabalho é reescrever a pergunta — nunca respondê-la:
```python
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_QUESTION_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
```

O histórico em si é montado na interface (`app/ui/streamlit_app.py`, função `montar_chat_history`),
convertendo o que está guardado em `st.session_state.messages` (dicionários simples, usados para
exibir a conversa na tela) para `HumanMessage`/`AIMessage` — o formato que `MessagesPlaceholder`
sabe interpretar dentro dos dois prompts do sistema.

A partir daí, o restante do Cenário 3 é **idêntico** ao Cenário 2: a pergunta reescrita vai para o
retriever e o reranker, os 4 chunks finais viram contexto, e a LLM gera a resposta — só que agora
lendo também o histórico da conversa, o que permite respostas com continuidade
("como mencionei antes...", por exemplo), quando fizer sentido.
