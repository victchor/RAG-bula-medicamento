import sys

from app.config import load_settings
from app.llm import get_chat_model

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    settings = load_settings()
    llm = get_chat_model(settings)

    resposta = llm.invoke("Em uma frase, o que é RAG (Retrieval-Augmented Generation)?")

    print(f"Provedor: {settings.llm_provider}")
    print(f"Resposta: {resposta.content}")
    print(f"Uso de tokens: {resposta.usage_metadata}")


if __name__ == "__main__":
    main()
