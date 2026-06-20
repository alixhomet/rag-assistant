"""Client LLM (Mistral) et génération de réponses ancrées dans le contexte."""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.prompts import NO_CONTEXT_ANSWER, RAG_PROMPT
from rag_assistant.core.retriever import RetrievedChunk
from rag_assistant.utils.exceptions import LLMError
from rag_assistant.utils.logger import get_logger

logger = get_logger("llm")


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Construit le modèle de chat configuré.

    Point de bascule unique du fournisseur : passer à un modèle local
    (ex. ChatOllama) ne demandera que de modifier cette fonction.
    """
    settings = settings or get_settings()
    return ChatMistralAI(
        model=settings.chat_model,
        api_key=settings.mistral_api_key.get_secret_value(),
        temperature=settings.temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout,
    )


class AnswerGenerator:
    """Génère une réponse à partir d'une question et des passages récupérés."""

    def __init__(
        self,
        settings: Settings | None = None,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = chat_model or create_chat_model(self._settings)
        # Chaîne LCEL : prompt -> modèle -> extraction du texte
        self._chain = RAG_PROMPT | self._llm | StrOutputParser()

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        """Assemble les passages en un contexte étiqueté par source."""
        blocks = [
            f"[Source : {chunk.source}, page {chunk.page}]\n{chunk.content}"
            for chunk in chunks
        ]
        return "\n\n".join(blocks)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Produit une réponse ancrée dans les passages fournis."""
        if not question or not question.strip():
            raise LLMError("La question est vide")

        # Garde anti-hallucination : pas de contexte -> pas d'appel au LLM
        if not chunks:
            logger.info("Aucun passage pertinent : réponse de repli renvoyée")
            return NO_CONTEXT_ANSWER

        context = self._format_context(chunks)
        try:
            answer = self._chain.invoke({"context": context, "question": question})
        except Exception as exc:
            raise LLMError(f"Échec de la génération de la réponse : {exc}") from exc

        logger.info(
            "Réponse générée (%d caractères) à partir de %d passage(s)",
            len(answer), len(chunks),
        )
        return answer
