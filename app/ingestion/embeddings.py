from langchain_huggingface import HuggingFaceEmbeddings

# Multilíngue (a bula é em pt-BR) e local — sem custo de API.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
