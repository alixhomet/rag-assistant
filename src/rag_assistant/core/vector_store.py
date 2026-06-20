"""Stockage et gestion des vecteurs dans ChromaDB (persistant sur disque)."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_assistant.config import Settings, get_settings
from rag_assistant.core.embedder import Embedder
from rag_assistant.utils.exceptions import VectorStoreError
from rag_assistant.utils.logger import get_logger

logger = get_logger("vector_store")


class VectorStore:
    """Encapsule la base vectorielle ChromaDB persistée sur disque.

    Centraliser ici l'accès à Chroma permet de changer de base vectorielle
    (Qdrant, Pinecone, pgvector...) en ne modifiant que cette classe.
    """

    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or Embedder(self._settings)
        try:
            self._store = Chroma(
                collection_name=self._settings.collection_name,
                embedding_function=self._embedder.embeddings,
                persist_directory=str(self._settings.chroma_dir),
                collection_metadata={"hnsw:space": "cosine"},  # distance = cosinus
            )
        except Exception as exc:
            raise VectorStoreError(f"Impossible d'initialiser ChromaDB : {exc}") from exc

    @property
    def store(self) -> Chroma:
        """Expose le store LangChain (consommé par le retriever, étape 6)."""
        return self._store

    def add_documents(self, chunks: list[Document]) -> list[str]:
        """Indexe des chunks. L'usage des chunk_id rend l'opération idempotente."""
        if not chunks:
            raise VectorStoreError("Aucun chunk à indexer")
        try:
            ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        except KeyError as exc:
            raise VectorStoreError("Chunk sans 'chunk_id' : exécutez le chunker d'abord") from exc
        try:
            self._store.add_documents(documents=chunks, ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Échec de l'indexation : {exc}") from exc
        logger.info("Indexation : %d chunk(s) ajouté(s)/mis à jour", len(ids))
        return ids

    def count(self) -> int:
        """Nombre de chunks stockés."""
        try:
            return len(self._store.get(include=[])["ids"])
        except Exception as exc:
            raise VectorStoreError(f"Échec du comptage : {exc}") from exc

    def list_sources(self) -> list[str]:
        """Liste dédupliquée des documents indexés."""
        try:
            metadatas = self._store.get(include=["metadatas"])["metadatas"]
        except Exception as exc:
            raise VectorStoreError(f"Échec de la lecture des métadonnées : {exc}") from exc
        return sorted({meta.get("source", "inconnu") for meta in metadatas})

    def delete_by_source(self, source: str) -> int:
        """Supprime tous les chunks d'un document (base de la réindexation)."""
        try:
            ids = self._store.get(where={"source": source}, include=[])["ids"]
            if ids:
                self._store.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Échec de la suppression de '{source}' : {exc}") from exc
        logger.info("Suppression : %d chunk(s) du document '%s'", len(ids), source)
        return len(ids)

    def clear(self) -> None:
        """Vide entièrement la collection."""
        try:
            ids = self._store.get(include=[])["ids"]
            if ids:
                self._store.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Échec de la purge : {exc}") from exc
        logger.info("Collection vidée")
