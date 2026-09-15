from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_COLLECTION_NAME = "bula"


def create_new_vectorstore(
    documents: list[Document],
    embedding: Embeddings,
    persist_directory: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    # Chroma.from_documents() só adiciona, nunca substitui uma coleção
    # existente — sem apagar antes, reindexações repetidas acumulam
    # cópias duplicadas dos mesmos chunks.
    Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name=collection_name,
    ).delete_collection()

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def load_existing_vectorstore(
    embedding: Embeddings,
    persist_directory: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
        collection_name=collection_name,
    )
