"""Recherche sémantique : récupération des passages pertinents pour une question."""

from dataclasses import dataclass

from langchain_core.documents import Document

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.vector_store import VectorStore
from rag_assistant.utils.exceptions import RetrievalError
from rag_assistant.utils.logger import get_logger

logger = get_logger("retriever")


@dataclass
class RetrievedChunk:
    """Un passage récupéré, accompagné de son score de similarité."""

    content: str
    source: str
    page: int | str
    score: float          # similarité dans [0, 1] ; 1 = très pertinent
    chunk_id: str

    @classmethod
    def from_document(cls, document: Document, score: float) -> "RetrievedChunk":
        meta = document.metadata
        return cls(
            content=document.page_content,
            source=meta.get("source", "inconnu"),
            page=meta.get("page", "?"),
            score=score,
            chunk_id=meta.get("chunk_id", ""),
        )


class Retriever:
    """Transforme une question en recherche sémantique sur la base vectorielle."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vector_store = vector_store or VectorStore(self._settings)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Récupère les passages les plus pertinents pour une question.

        - top_k : nombre de passages (défaut : config).
        - score_threshold : ignore les passages sous ce seuil de similarité.
        - sources : restreint la recherche à certains documents (multi-PDF).
        """
        if not query or not query.strip():
            raise RetrievalError("La question est vide")

        k = top_k or self._settings.top_k
        where = {"source": {"$in": sources}} if sources else None

        try:
            results = self._vector_store.store.similarity_search_with_score(
                query=query, k=k, filter=where,
            )
        except Exception as exc:
            raise RetrievalError(f"Échec de la recherche sémantique : {exc}") from exc

        chunks: list[RetrievedChunk] = []
        for document, distance in results:
            similarity = self._to_similarity(distance)
            if score_threshold is not None and similarity < score_threshold:
                continue
            chunks.append(RetrievedChunk.from_document(document, similarity))

        logger.info(
            "Recherche : '%s' -> %d passage(s) retenu(s) sur %d candidat(s)",
            query[:50], len(chunks), len(results),
        )
        return chunks

    @staticmethod
    def _to_similarity(distance: float) -> float:
        """Convertit une distance cosinus en score de similarité borné [0, 1]."""
        return max(0.0, min(1.0, 1.0 - distance))
