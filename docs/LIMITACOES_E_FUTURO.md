# Limitações, riscos e possíveis usos futuros

## Aviso importante

**Este é um projeto de estudo.** Foi construído para aprender arquitetura de RAG na prática — não
foi projetado, testado ou validado para uso em produção, e não deve ser usado como fonte real de
decisão sobre medicamentos.

O sistema é um assistente de **consulta ao texto de uma bula**. Ele não é, e não substitui:
- diagnóstico médico;
- orientação de um médico, farmacêutico ou outro profissional de saúde;
- a bula impressa original, que deve prevalecer em caso de qualquer dúvida ou divergência.

Use este projeto para aprender sobre RAG. Não use as respostas dele para decidir como tomar um
medicamento.

---

## Limitações técnicas conhecidas

- **Um usuário, uma bula por vez.** O índice vetorial é global e compartilhado — processar uma
  bula nova substitui a anterior. Não há isolamento entre usuários; não é seguro para múltiplos
  usuários simultâneos sem retrabalho de arquitetura.
- **Sem persistência de conversa entre sessões.** O histórico vive em memória (`st.session_state`)
  e se perde ao recarregar a página ou reiniciar a aplicação.
- **Sem cache de resposta.** Perguntas repetidas geram chamadas novas à LLM — sem economia de
  custo ou tempo, mesmo para a pergunta idêntica de segundos atrás.
- **Qualidade depende do PDF.** Bulas escaneadas como imagem (sem camada de texto) não são lidas —
  o sistema extrai texto, não faz OCR.
- **Chunking por tamanho, não por seção.** Um chunk pode misturar o fim de uma seção da bula com o
  início da próxima, em vez de respeitar os limites reais de "Contraindicações", "Posologia" etc.
- **Avaliação automatizada é pequena e específica de um documento.** O gabarito de teste
  (`app/evaluation/datasets/`) cobre poucas perguntas de uma única bula — não é uma prova de
  qualidade geral, só um alarme para regressões óbvias naquele documento específico.
- **Testado majoritariamente com Gemini, não com Claude.** A maior parte do desenvolvimento usou a
  camada gratuita do Gemini (por limitação de crédito da API Anthropic). Comportamentos podem
  diferir ao trocar de provedor.
- **Cota gratuita do Gemini é limitada.** Alguns modelos têm limite de poucas dezenas de
  requisições por dia — uso intenso pode esbarrar nisso.

## Riscos de segurança conhecidos

- **Prompt injection é mitigado, não eliminado.** Uma regra no prompt instrui a LLM a ignorar
  instruções escondidas no texto da bula, testada contra algumas técnicas de ofuscação — isso
  reduz o risco, não é uma garantia permanente contra qualquer técnica futura.
- **Sem autenticação nem controle de acesso.** Qualquer pessoa com acesso à URL da aplicação pode
  enviar arquivos e fazer perguntas.
- **Sem rate limiting.** Nada impede uso abusivo (custo, volume de requisições) além do limite de
  tamanho de pergunta implementado.
- **Validação de arquivo é básica.** Confere tamanho e assinatura de PDF — não faz varredura de
  antivírus nem análise de conteúdo malicioso mais sofisticado.

## Riscos de confiabilidade do conteúdo

- **Retrieval pode falhar.** Mesmo com reranking, não há garantia de que o trecho certo da bula
  seja sempre encontrado — um bug real desse tipo foi encontrado e corrigido durante o
  desenvolvimento (ver `docs/ETAPAS.md`, Etapa 11).
- **Alucinação é mitigada, não impossível.** O prompt instrui a LLM a nunca inventar informação
  fora do contexto — isso reduz drasticamente o risco, mas nenhuma instrução de prompt é uma
  garantia absoluta de comportamento de um modelo de linguagem.
- **Comportamento pode variar entre modelos e versões futuras.** LLMs são atualizadas com
  frequência; uma troca de modelo (ou até uma nova versão do mesmo modelo) pode mudar sutilmente
  como as regras do prompt são seguidas.

---

## O que este projeto poderia se tornar, com investimento adicional

A arquitetura de base — PDF → chunks → embeddings → busca → reranking → LLM — é a mesma usada em
sistemas de produção reais. O que separa este projeto de uma aplicação real não é o conceito, é o
trabalho de engenharia adicional: segurança reforçada, testes em escala, infraestrutura
multi-usuário, validação regulatória (no caso de um produto de saúde) e integração com sistemas
externos.

**Um exemplo concreto**: este projeto, bem estruturado e com as devidas validações de segurança,
poderia ser a base de um **totem de farmácia** — o cliente escaneia o código de barras do
medicamento na embalagem, o sistema identifica e carrega a bula correspondente automaticamente (em
vez de upload manual de PDF), e o cliente faz perguntas em linguagem natural sobre dosagem,
contraindicações ou interações, sem precisar ler o texto denso da bula impressa.

Para sair do estudo e chegar a algo assim, seria necessário, entre outras coisas:
- uma base de bulas oficiais já indexadas (não upload manual), atualizada junto com a ANVISA;
- integração com um banco de dados de código de barras → medicamento;
- multi-tenant de verdade (vários totens, várias sessões simultâneas, sem interferência entre si);
- autenticação/controle de acesso adequado ao ambiente (ex.: uso interno da farmácia);
- revisão de conformidade regulatória para dispositivos de informação em saúde;
- avaliação de qualidade em escala, com gabaritos para muitas bulas, não uma;
- observabilidade e monitoramento de produção (o projeto já tem as bases disso, na Etapa 12).

Outras direções possíveis, na mesma linha: um assistente para atendentes de farmácia (consulta
rápida interna, não voltada ao público final), ou uma central de dúvidas para call centers de
laboratórios farmacêuticos.
