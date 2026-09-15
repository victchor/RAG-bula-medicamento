# Diário de bordo — Chatbot RAG de Bulas

Este documento registra **o que foi construído, com o quê, e por quê**, etapa por etapa. A ideia é
que você consiga voltar aqui a qualquer momento, bater o olho e lembrar o raciocínio por trás de
cada peça — sem precisar reler a conversa inteira que gerou o projeto.

Para "como rodar o projeto", veja o [README](../README.md). Este arquivo é sobre **decisões e
aprendizado**, não sobre setup.

---

## Visão geral da arquitetura

```text
                    ┌─────────────────┐
                    │     Usuário     │
                    └────────┬────────┘
                             │ upload da bula (PDF)
                             ▼
                    ┌─────────────────┐
                    │  PDF Loader     │  pypdf — extrai texto + metadata (página)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Text Splitter   │  corta em pedaços de ~1000 caracteres
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   Embeddings    │  texto → vetor de 384 números (local, grátis)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Vector Database │  Chroma — guarda vetor + texto + página
                    └─────────────────┘

Pergunta do usuário
   ↓ embed_query
   ↓ busca por similaridade (k=15 candidatos)
   ↓ reranking (cross-encoder escolhe os 4 melhores)
   ↓ prompt (regras + contexto + histórico + pergunta)
   ↓ LLM (Claude ou Gemini)
   ↓
Resposta + fontes citadas
```

## Stack final (o que foi escolhido e por quê)

| Peça | Escolha | Por quê |
|---|---|---|
| Orquestração | LangChain (LCEL) | Padroniza `Document`, chains, prompts; permite trocar peças (LLM, embeddings) sem reescrever o pipeline |
| LLM | Claude Haiku 4.5 **ou** Gemini (pluggable via `LLM_PROVIDER`) | Claude é o alvo original; Gemini entrou como alternativa gratuita porque a conta Anthropic não tinha crédito de API (assinatura Pro do Claude Code é um produto separado da API paga) |
| PDF Loader | `pypdf` via `PyPDFLoader` | Sem dependências nativas do sistema; suficiente para bulas em PDF "de texto" |
| Text Splitter | `RecursiveCharacterTextSplitter` | Tenta preservar frases/parágrafos inteiros antes de cortar no meio |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (local) | Multilíngue (bula é pt-BR), roda no seu PC, custo zero |
| Vector Store | Chroma | Embarcado, persiste em disco sozinho, boa ergonomia de metadata |
| Reranking | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (local) | Segunda camada de precisão sobre os candidatos da busca vetorial; grátis |
| Frontend | Streamlit | Interface funcional em Python puro, sem build step |
| Avaliação | Hit Rate@k + LLM-como-juiz (caseiro, sem framework) | Métrica objetiva de retrieval + veredito qualitativo de geração, sem dependência pesada |

---

## Etapa 1 — Arquitetura e Setup

**O que é:** esqueleto do projeto, ambiente Python isolado, primeira chamada real a um LLM.

**Tecnologia:** `langchain-anthropic`, `langchain-google-genai`, `python-dotenv`. Python **3.12** via
venv (não o 3.14 do sistema — versão nova demais, bibliotecas de ML como `torch` ainda não publicam
wheels pré-compiladas para ela de forma confiável).

**Decisões-chave:**
- `app/config.py`: um único lugar que lê e **valida** variáveis de ambiente, falhando rápido
  (`RuntimeError`) se faltar algo — evita erros confusos no meio do processamento de um PDF.
- `app/llm.py`: fábrica `get_chat_model(settings)` que devolve um `ChatAnthropic` ou
  `ChatGoogleGenerativeAI` dependendo de `LLM_PROVIDER` — o resto do projeto nunca sabe qual dos
  dois está por trás.

**Aprendizado que não estava no plano:** a assinatura Claude Pro/Max (usada no Claude Code) e a API
paga da Anthropic (`console.anthropic.com`) são **produtos de billing separados**. Isso forçou a
decisão de suportar dois provedores de LLM desde o início — o que acabou virando uma lição de
arquitetura sobre desacoplamento (chains são agnósticas de provedor).

---

## Etapa 2 — Leitura do PDF

**O que é:** extrair texto + metadata de um PDF real de bula.

**Tecnologia:** `langchain-community` (onde vive o `PyPDFLoader`) + `pypdf` (engine de extração).

**Decisões-chave:**
- `app/ingestion/pdf_loader.py`: função fina `load_pdf(path)` — isola a biblioteca do resto da
  aplicação; se um dia precisarmos de OCR para bulas escaneadas, só esse arquivo muda.
- 1 `Document` por página (comportamento padrão do `PyPDFLoader`) — cada um carrega `page_label` na
  metadata, que **flui até o final do pipeline** e é o que permite citar a fonte na Etapa 8.

**Bula de teste:** `PARACETAMOL-Bula-Profissional.pdf` (fabricante Geolab), baixada com permissão do
site do fabricante, usada em todas as etapas seguintes como corpus real.

**Problemas reais encontrados (não hipotéticos):**
- Acentos aparecendo como `�` no terminal — **não era bug de extração**, era o console do Windows
  tentando exibir UTF-8 com a codepage errada. Resolvido com `sys.stdout.reconfigure(encoding="utf-8")`
  nos scripts.
- Cabeçalho repetido (`V.06_05/2022`) em quase toda página — ruído, mas inofensivo.
- Palavras grudadas (`ReferênciasBibliográficas`) por falta de espaço real no PDF original.

---

## Etapa 3 — Chunking

**O que é:** dividir o texto extraído em pedaços menores antes de indexar.

**Tecnologia:** `langchain-text-splitters`, `RecursiveCharacterTextSplitter`.

**Por quê dividir:** (1) custo — mandar o PDF inteiro em toda pergunta desperdiça tokens; (2)
qualidade — LLMs perdem precisão quando a informação relevante está perdida em meio a texto
irrelevante ("lost in the middle").

**Parâmetros escolhidos:** `chunk_size=1000` caracteres, `chunk_overlap=150` (15%). Bulas têm
seções curtas e diretas — chunk grande o bastante pra conter uma seção inteira, pequeno o bastante
pra não misturar duas seções.

**Validado com dado real:** vimos o overlap "salvando" uma frase cortada entre dois chunks (a
seção "1. INDICAÇÕES" reaparecendo no início do chunk seguinte). Também vimos um chunk misturando o
fim de uma seção com o início de outra — limitação conhecida de chunking por tamanho (não por
seção), documentada mas não corrigida (não era bloqueador).

---

## Etapa 4 — Embeddings

**O que é:** transformar texto em vetores numéricos que capturam significado, não só palavras exatas.

**Tecnologia:** `langchain-huggingface` + `sentence-transformers`, modelo
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões, multilíngue, local).

**Por quê esse modelo:** bula é em português — um modelo só-inglês teria qualidade ruim aqui. Local
= custo zero por embedding (nenhuma chamada de API), ao custo de um download inicial (~470MB) e ser
um pouco mais lento que uma API na nuvem.

**`app/ingestion/embeddings.py`:** fábrica fina `get_embeddings()`, mesmo padrão de `get_chat_model`
— sem lógica, só configuração.

**Regra de ouro validada na prática:** o mesmo modelo de embedding precisa ser usado para indexar
os chunks E para transformar a pergunta do usuário — senão os vetores vivem em espaços matemáticos
incomparáveis.

**Descoberta contraintuitiva:** comparar duas perguntas curtas entre si ("como armazenar" vs "qual
temperatura") deu similaridade **baixa e na ordem errada** — não é assim que o RAG funciona de
verdade (ele compara pergunta-com-chunk-de-documento, não pergunta-com-pergunta). Ao testar contra
os chunks reais da bula, veio à tona um problema real: o chunk certo sobre armazenamento ficou em
**7º lugar de 34** por similaridade de cosseno — sinal precoce de um problema de retrieval que só
seria totalmente diagnosticado (e sua causa raiz corrigida) na Etapa 11.

---

## Etapa 5 — Banco Vetorial (Vector Store)

**O que é:** armazenar os vetores + texto + metadata de forma persistente, pronta pra busca rápida.

**Tecnologia:** `langchain-chroma` + `chromadb`.

**Por quê Chroma (e não FAISS):** embarcado (sem servidor separado), persiste em disco
automaticamente, ergonomia melhor de metadata. Para uma bula por vez, a diferença de performance
para FAISS é irrelevante.

**`app/ingestion/vectorstore.py`:** duas funções — `create_new_vectorstore` (indexa do zero) e
`load_existing_vectorstore` (reconecta a um índice já salvo, sem reprocessar o PDF). As duas
precisam usar o mesmo `collection_name` (é a "chave" que as conecta ao mesmo dado).

**Bug real encontrado (e corrigido só na Etapa 11):** `Chroma.from_documents()` nunca *substitui*
uma coleção existente, só *adiciona* a ela. Reprocessar a mesma bula várias vezes pela interface
Streamlit empilhava cópias duplicadas — 170 chunks na coleção em vez dos 34 esperados. Corrigido
apagando a coleção antiga (`.delete_collection()`) antes de criar a nova.

---

## Etapa 6 — Retrieval

**O que é:** uma interface padronizada (`BaseRetriever`) por cima do vector store.

**Por quê essa camada extra:** desacopla a chain (Etapa 7) do Chroma especificamente — se um dia
trocarmos a fonte de busca (outro vector store, busca híbrida, reranking), a chain nem percebe.

**`app/rag/retriever.py`:** `get_retriever(vectorstore, k)` → `vectorstore.as_retriever(...)`.

**Conceitos validados com dado real:**
- Falso positivo: chunk irrelevante (texto de capa) no top-k.
- Falso negativo: chunk relevante (armazenamento) fora do top-k.
- Cuidado com a direção do score: o Chroma mede **distância** por padrão (menor = melhor), o
  oposto do que se espera intuitivamente de um "score de relevância".

---

## Etapa 7 — Primeiro RAG Funcionando

**O que é:** conectar retriever + prompt + LLM pela primeira vez — uma **chain**, não um Agent (o
caminho é sempre fixo: busca → monta prompt → gera resposta, sem decisão dinâmica).

**Tecnologia:** LCEL (LangChain Expression Language) — operador `|` encadeia `Runnable`s.

**`app/rag/chain.py`:** `get_rag_chain(retriever, llm)`, com `RunnableParallel` montando
`{context, question}` a partir da pergunta, e `format_docs` convertendo a lista de chunks numa
única string de contexto.

**Descoberta útil:** `StrOutputParser` já lida sozinho com a diferença de formato de resposta entre
Claude (string simples) e Gemini (lista de blocos) — usa a property `.text` do LangChain por baixo.

**Teste decisivo:** perguntando sobre "armazenamento" (retrieval ruim, conhecido) com um prompt
mínimo, a LLM **não inventou** uma resposta — disse que não encontrou a informação. Retrieval ruim
≠ resposta inventada, se o prompt already empurra nessa direção.

---

## Etapa 8 — Prompt Engineering

**O que é:** formalizar, em regras explícitas, o comportamento que apareceu "por sorte" na Etapa 7.

**`app/rag/prompts.py`:** `SYSTEM_PROMPT` com 6 regras, cada uma mitigando um risco específico:

| Regra | Risco mitigado |
|---|---|
| Usar só o contexto fornecido | LLM misturar conhecimento genérico de treino com a bula específica |
| Dizer quando não encontrar a informação | Alucinação por "completar a lacuna" |
| Sinalizar inferência vs. citação literal | Confundir interpretação própria com texto da bula |
| Citar a página de origem | Rastreabilidade/verificação pelo usuário |
| Ser claro e direto | Usabilidade |
| Postura de consulta, não diagnóstico | Segurança/uso indevido |

Para a regra de citar página funcionar de verdade, `format_docs` (Etapa 7) precisou mudar para
incluir `[Página N]` junto de cada chunk no contexto — sem isso a LLM não teria como citar nada.

---

## Etapa 9 — Frontend (Streamlit)

**O que é:** interface web funcional — upload, processamento, chat, fontes.

**Tecnologia:** Streamlit.

**Dois conceitos obrigatórios de entender:**
- Streamlit reroda o **script inteiro** a cada interação do usuário.
- `st.session_state` sobrevive a esses reruns (guarda a chain montada, o histórico);
  `st.cache_resource` evita recarregar modelos pesados (embeddings, LLM) a cada clique.

**`app/ui/streamlit_app.py`:** único arquivo que "conhece" todas as outras peças do projeto — seu
papel é orquestrar, não conter lógica de RAG.

**Limite real encontrado:** ferramentas de automação de navegador não conseguem interagir com o
seletor de arquivo nativo do sistema operacional — o teste de upload precisou ser manual.

---

## Etapa 10 — Conversação (memória)

**O que é:** interpretar perguntas de acompanhamento ("E quais são os efeitos colaterais?") usando
o histórico da conversa.

**Mecanismo — reformulação de pergunta:** antes de buscar no vector store, uma chamada de LLM
reescreve a pergunta atual como uma pergunta autônoma, usando o histórico. Só é chamada quando
existe histórico (pula na 1ª pergunta da conversa, economizando uma chamada de LLM).

**`app/rag/prompts.py`:** `CONDENSE_QUESTION_PROMPT` (reescreve, não responde) e `MessagesPlaceholder`
inserido em `RAG_PROMPT` para carregar o histórico até a LLM de geração também.

**`app/rag/chain.py` reestruturado:** busca os documentos **uma única vez** por pergunta e devolve
`{"answer": ..., "source_documents": ...}` juntos — evita mostrar "fontes" diferentes das que
realmente entraram no prompt (um risco real, já que a pergunta de busca pode ser diferente da
pergunta exibida ao usuário, por causa da reformulação).

**Custo real:** cada pergunta de acompanhamento agora custa **2 chamadas de LLM** (reformular +
responder), não 1. Histórico limitado a `MAX_HISTORICO_MENSAGENS=6` no Streamlit — não cresce
para sempre.

---

## Etapa 11 — Avaliação

**O que é:** medir sistematicamente, em vez de testar manualmente pergunta por pergunta.

**Duas camadas:**
1. **Retrieval — Hit Rate@k** (`app/evaluation/metrics.py::hit_rate`): a página certa apareceu
   entre os chunks recuperados? Rápido, determinístico, sem custo de LLM.
2. **Geração — LLM como juiz** (`avaliar_resposta_com_llm_juiz`): outra LLM compara a resposta
   gerada com um gabarito humano e dá um veredito (`CORRETO`/`PARCIAL`/`INCORRETO`). Não usa
   framework pronto (RAGAS/DeepEval) — implementado em ~30 linhas com o que já tínhamos.

**Dataset é por documento, não universal:** `app/evaluation/datasets/paracetamol_geolab.py` só
serve para essa bula específica (fabricante Geolab). Uma bula nova exige um gabarito novo — o
motivo é conceitual, não uma limitação do código: a resposta certa e a página certa são
propriedades do **documento**, não do sistema RAG.

**O que a avaliação revelou:**
- Hit Rate inicial de **40%** — bem pior do que qualquer teste manual anterior tinha sugerido.
- Investigação revelou um **bug real**: a coleção do Chroma tinha 170 chunks em vez de 34 (upload
  repetido pela interface duplicando dados — ver Etapa 5). Corrigido.
- Depois de corrigir o bug, Hit Rate foi para **100%** — inclusive sem reranking nenhum. A "causa
  raiz" suspeitada desde a Etapa 4 (modelo de embedding fraco) nunca foi o problema real.

**Reranking implementado mesmo assim** (`app/rag/reranker.py`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
local/grátis): busca 15 candidatos por embedding (`K_CANDIDATOS` em `retriever.py`), um cross-encoder
reordena com mais precisão (lê pergunta+chunk juntos, não separadamente), ficam os 4 melhores
(`K_FINAL` em `chain.py`). Decisão consciente do usuário de manter essa camada mesmo sem ganho
mensurável neste dataset pequeno — defesa para bulas/perguntas futuras mais difíceis, sem custo
monetário (roda local).

**Lição de metodologia:** sempre descarte bugs banais nos dados antes de investir em soluções mais
sofisticadas — isolar variáveis (testar k=4-sem-rerank vs. k=15-com-rerank no mesmo dado limpo)
evitou a conclusão errada de que o reranking "resolveu" algo que já tinha sido resolvido pelo bug fix.

**Outro obstáculo real:** cota gratuita do Gemini é por modelo — `gemini-3.6-flash` tem limite de
apenas 20 requisições/dia na camada free. Trocamos o padrão do projeto para `gemini-3.1-flash-lite`
(cota bem maior, e sem o overhead de "thinking" que o outro modelo tinha).

---

## Etapa 12 — Observabilidade

**O que é:** enxergar o que acontece dentro do pipeline a cada pergunta — tempo, chamadas de LLM,
tokens, chunks recuperados, erros — sem instrumentar manualmente cada função.

**Mecanismo:** callbacks do LangChain (`BaseCallbackHandler`). Todo `Runnable` aceita
`config={"callbacks": [...]}` no `.invoke()`, e o observador recebe eventos automáticos (início/fim
de chamada de LLM, resultado do retriever, erros) — **sem alterar uma linha de `chain.py`**. É o
mesmo mecanismo que o LangSmith usa por trás; só muda pra onde os dados vão.

**`app/observability/handler.py`:** `ObservabilityHandler` (coleta métricas) + `formatar_metricas`
(monta a linha de resumo). Conectado em `scripts/ask.py` e `app/ui/streamlit_app.py`.

**Descoberta no caminho:** o reranking (Etapa 11) não é um `Runnable` rastreado — é só uma função
Python — então o callback `on_retriever_end` captura os **candidatos** (k=15), não os 4 finais
usados na resposta. Corrigido mostrando os dois números lado a lado (`candidatos(k=15)` vs.
`usados_na_resposta`), em vez de confiar cegamente no primeiro sinal disponível.

**Validado com dado real:** uma pergunta de acompanhamento mostrou `chamadas_llm=2` e quase 3x mais
tempo que uma pergunta isolada — confirmando com números o custo de reformulação que só tínhamos
discutido em teoria na Etapa 10.

**LangSmith:** documentado como upgrade natural (dashboard visual, histórico entre execuções), mas
não ativado agora — exigiria mais uma conta/signup, e nosso "callback caseiro" já cobre o que o
briefing original pedia (tempo, chunks, tokens, custo, erros, chamadas).

---

## Etapa 13 — Segurança

**O que é:** tratar como não-confiável tudo que entra de fora — o arquivo enviado, o texto extraído
dele, e a pergunta do usuário — sem adicionar complexidade que só faria sentido num serviço
multi-usuário público (fora de escopo aqui, é uma aplicação local de um usuário só).

**`app/security/file_validation.py`:**
- `validar_pdf`: checa a **assinatura real do arquivo** (`%PDF-`, os primeiros bytes), não só a
  extensão do nome (fácil de forjar renomeando um `.exe`); limite de 20MB.
- `sanitizar_nome_arquivo`: remove qualquer componente de caminho do nome do arquivo (proteção
  contra *path traversal* — um nome como `../../../.env` não pode escrever fora de `data/uploads/`).

**`app/rag/prompts.py` — Regra 7 (defesa contra prompt injection):** o texto extraído do PDF é
conteúdo de terceiros, não confiável. Sem instrução explícita, a LLM poderia obedecer um comando
escondido dentro da bula (ex.: "ignore as instruções anteriores"). Testado ao vivo com um chunk
"envenenado" simulado — a LLM ignorou a instrução embutida e não revelou o system prompt nem
inventou informação.

**`app/ui/streamlit_app.py`:**
- Validação roda ANTES de qualquer processamento (economiza tempo/custo com arquivo inválido).
- PDF sem texto extraível (ex.: escaneado sem OCR) ou corrompido → mensagem clara, sem crash.
- Chamadas à LLM protegidas por `try/except` — erros de rede/rate limit não mostram stack trace cru
  na tela (poderia expor caminhos internos do sistema).
- Pergunta limitada a 500 caracteres — barato de implementar, evita custo/abuso desnecessário.

**`scripts/test_prompt_injection.py`** — suíte de 4 tentativas de injeção mais sofisticadas
(falso encerramento de documento + override, instrução em base64, troca de persona, ataque
multicamada com autoridade falsa da ANVISA), testadas contra a chain real (bypassando o retriever,
injetando o "chunk malicioso" direto no contexto). Todas as 4 falharam contra `gemini-3.1-flash-lite`
com a Regra 7 — mas isso é uma amostra pequena com um modelo específico, não uma prova permanente;
o script fica no projeto pra reexecutar sempre que o modelo ou o prompt mudarem.

**Já mitigado desde etapas anteriores (revisado aqui):** `.env` no `.gitignore` desde a Etapa 1,
chave nunca hardcoded no código, postura de "consulta, não diagnóstico" no prompt desde a Etapa 8.

**Fora de escopo, documentado por quê:** antivírus/sandboxing de PDF, autenticação de usuário, rate
limiting por IP, isolamento de vector store multi-usuário — relevante só se este projeto virasse um
serviço público com vários usuários simultâneos, o que não é o caso hoje.

---

## Etapa 14 — Custos

**O que é:** entender quanto custa rodar o sistema de verdade, com números medidos — não chutados.

**Os únicos custos reais são as chamadas de LLM.** Embeddings (Etapa 4) e reranking (Etapa 11) são
modelos locais — custam tempo de CPU, nunca dinheiro. `app/observability/handler.py` já tinha a
tabela de preços (Etapa 12); expandida aqui com Sonnet 5 ($2/$10 por milhão de tokens),
Opus 5 ($5/$25) e o preço de tabela do Gemini Flash-Lite pago ($0.25/$1.50), pra comparação.

**`scripts/estimate_costs.py`** — roda 2 perguntas reais (uma isolada, uma de acompanhamento)
contra a chain, mede tokens de verdade, e calcula custo/projeção mensal pra 4 modelos diferentes.
Resultado medido: **$0.0024-0.0026 por pergunta com Haiku 4.5** — a 100 perguntas/dia, **$7.50/mês**.

**Descobertas:**
- Escolha de modelo é a alavanca de custo mais forte (Opus custa 5x Haiku pelos mesmos tokens).
- Pergunta de acompanhamento custa ~2x em chamadas de LLM, mas só ~22% a mais em tokens de
  entrada — o que pesa é pagar o overhead do system prompt duas vezes, não o histórico em si.
- **Prompt caching** (não implementado): o `SYSTEM_PROMPT`, idêntico em toda chamada, é candidato
  natural — até 90% de desconto em chamadas repetidas. Não vale a complexidade no volume atual,
  mas seria a primeira otimização num cenário de produção real.

---

## Roteiro original: completo (Etapas 1-14)

Com Custos, fecham-se as 14 etapas do plano original. A única peça declarada como "só se houver
necessidade real" — transformar isso num Agent — segue deliberadamente não implementada, porque
essa necessidade nunca apareceu: o pipeline sempre foi um caminho fixo e previsível (chain), sem
nenhum momento em que o sistema precisasse decidir dinamicamente "o que fazer a seguir".

---

## Conceitos já cobertos (da lista original)

`RAG` · `LLM` · `Prompt Engineering` · `Tokens` · `PDF Loader` · `Documents` · `Chunks` ·
`Chunking` · `Embeddings` · `Vector Database` · `Vector Search` · `Similarity Search` ·
`Retriever` · `Context` · `Prompt` · `Generation` · `Chains` · `LCEL` · `Memory` ·
`Evaluation` · `Observability` · `Tracing` · `Latency` · `Hallucination` · `Security` ·
`Prompt Injection` · `Cost Optimization` ·
`Reranking / Cross-Encoder` (novo, não estava na lista original)

**Ainda não coberto:** `Agents` — deliberadamente, por falta de necessidade real (ver seção acima).
