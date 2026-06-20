"""Génération d'embeddings : transformation du texte en vecteurs sémantiques."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mistralai import MistralAIEmbeddings

from rag_assistant.config import Settings, get_settings
from rag_assistant.utils.exceptions import EmbeddingError
from rag_assistant.utils.logger import get_logger

logger = get_logger("embedder")


def create_embeddings(settings: Settings | None = None) -> Embeddings:
    """Construit l'objet d'embeddings configuré.

    Point de bascule unique du fournisseur : pour passer à un modèle local
    (ex. HuggingFaceEmbeddings) plus tard, il suffira de modifier cette
    fonction, sans toucher au reste du pipeline.
    """
    settings = settings or get_settings()
    return MistralAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.mistral_api_key.get_secret_value(),
        max_retries=settings.embedding_max_retries,
        timeout=settings.embedding_timeout,
    )


class Embedder:
    """Service d'embedding avec logging et gestion d'erreurs.

    Encapsule l'objet Embeddings de LangChain. Le vector store (étape 5)
    consommera directement la propriété `.embeddings` ; les méthodes
    explicites servent aux cas où l'on a besoin des vecteurs bruts
    (tests, calculs de similarité, débogage).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # Injection possible d'un faux objet pour les tests (sans appel réseau)
        self._embeddings = embeddings or create_embeddings(self._settings)

    @property
    def embeddings(self) -> Embeddings:
        """Expose l'objet LangChain, consommé par la base vectorielle."""
        return self._embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Vectorise une liste de textes (documents)."""
        if not texts:
            raise EmbeddingError("Aucun texte à vectoriser")
        try:
            vectors = self._embeddings.embed_documents(texts)
        except Exception as exc:
            raise EmbeddingError(f"Échec de la génération des embeddings : {exc}") from exc

        dim = len(vectors[0]) if vectors else 0
        logger.info("Embeddings : %d texte(s) -> %d vecteur(s) de dim %d",
                    len(texts), len(vectors), dim)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Vectorise une requête utilisateur unique."""
        if not query or not query.strip():
            raise EmbeddingError("La requête à vectoriser est vide")
        try:
            return self._embeddings.embed_query(query)
        except Exception as exc:
            raise EmbeddingError(f"Échec de la vectorisation de la requête : {exc}") from exc

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Commodité : vectorise directement une liste de Documents."""
        return self.embed_texts([doc.page_content for doc in documents])
