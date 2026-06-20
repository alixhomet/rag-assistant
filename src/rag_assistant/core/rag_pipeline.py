"""Orchestration : façade unique du pipeline RAG (ingestion + réponse)."""

from dataclasses import dataclass, field

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.chunker import TextChunker
from rag_assistant.core.llm import AnswerGenerator
from rag_assistant.core.pdf_loader import PDFLoader
from rag_assistant.core.retriever import RetrievedChunk, Retriever
from rag_assistant.core.vector_store import VectorStore
from rag_assistant.utils.logger import get_logger

logger = get_logger("pipeline")


@dataclass
class IngestionResult:
    """Bilan d'une indexation de document."""

    source: str
    pages: int
    chunks_added: int


@dataclass
class AnswerResult:
    """Réponse générée, accompagnée des passages sources."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)

    @property
    def unique_sources(self) -> list[tuple[str, int | str]]:
        """Couples (document, page) dédupliqués, prêts pour l'affichage."""
        seen: list[tuple[str, int | str]] = []
        for chunk in self.sources:
            key = (chunk.source, chunk.page)
            if key not in seen:
                seen.append(key)
        return seen


class RAGPipeline:
    """Façade unique orchestrant l'ingestion et la réponse.

    L'interface ne connaît que cette classe ; les rouages internes
    (chunking, embeddings, recherche, prompt...) restent encapsulés.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        loader: PDFLoader | None = None,
        chunker: TextChunker | None = None,
        vector_store: VectorStore | None = None,
        retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._loader = loader or PDFLoader()
        self._chunker = chunker or TextChunker(self._settings)
        # Le retriever réutilise CE vector_store : même embedder à l'index
        # et à la requête, donc des vecteurs comparables (cf. étapes 4 & 6).
        self._vector_store = vector_store or VectorStore(self._settings)
        self._retriever = retriever or Retriever(self._vector_store, self._settings)
        self._generator = generator or AnswerGenerator(self._settings)

    # --- Phase 1 : ingestion ---
    def ingest_pdf(self, data: bytes, source_name: str, replace: bool = False) -> IngestionResult:
        """Indexe un PDF : extraction -> chunking -> stockage vectoriel.

        replace=True réindexe proprement (supprime l'ancienne version d'abord).
        """
        if replace:
            removed = self._vector_store.delete_by_source(source_name)
            if removed:
                logger.info("Réindexation de '%s' : %d ancien(s) chunk(s) purgé(s)",
                            source_name, removed)

        pages = self._loader.load(data, source_name=source_name)
        chunks = self._chunker.split(pages)
        self._vector_store.add_documents(chunks)

        logger.info("Ingestion de '%s' : %d page(s) -> %d chunk(s)",
                    source_name, len(pages), len(chunks))
        return IngestionResult(source=source_name, pages=len(pages), chunks_added=len(chunks))

    # --- Phase 2 : réponse ---
    def answer(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        sources: list[str] | None = None,
    ) -> AnswerResult:
        """Répond à une question à partir des documents indexés."""
        chunks = self._retriever.retrieve(
            question, top_k=top_k, score_threshold=score_threshold, sources=sources
        )
        text = self._generator.generate(question, chunks)
        return AnswerResult(question=question, answer=text, sources=chunks)

    # --- Gestion des documents (consommée par l'UI / bonus) ---
    def list_documents(self) -> list[str]:
        return self._vector_store.list_sources()

    def delete_document(self, source: str) -> int:
        return self._vector_store.delete_by_source(source)

    def clear(self) -> None:
        self._vector_store.clear()

    def document_count(self) -> int:
        return self._vector_store.count()
