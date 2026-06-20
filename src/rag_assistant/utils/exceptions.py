"""Exceptions métier du projet, pour une gestion d'erreurs explicite."""


class RagAssistantError(Exception):
    """Exception de base de l'application."""


class ConfigurationError(RagAssistantError):
    """Configuration invalide ou manquante."""


class PDFExtractionError(RagAssistantError):
    """Échec de l'extraction de texte d'un PDF."""


class EmbeddingError(RagAssistantError):
    """Échec lors de la génération des embeddings."""


class VectorStoreError(RagAssistantError):
    """Erreur liée à la base vectorielle (ChromaDB)."""


class RetrievalError(RagAssistantError):
    """Erreur lors de la recherche sémantique."""


class LLMError(RagAssistantError):
    """Erreur lors de l'appel au modèle de génération."""
class ChunkingError(RagAssistantError): """Erreur lors du découpage 
    des documents en chunks."""

class VectorStoreError(RagAssistantError):
    """Erreur liée à la base vectorielle."""
