"""Configuration centralisée du projet, chargée depuis l'environnement / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet : src/rag_assistant/config.py -> remonte de 2 niveaux
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Paramètres typés et validés de l'application."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Mistral AI ---
    embedding_max_retries: int = Field(3, ge=0, description="Tentatives en cas d'échec API")
    embedding_timeout: int = Field(60, gt=0, description="Timeout d'un appel embeddings (s)")  
    llm_max_retries: int = Field(3, ge=0, description="Tentatives LLM en cas d'échec API")
    llm_timeout: int = Field(60, gt=0, description="Timeout d'un appel LLM (s)") 
    mistral_api_key: SecretStr = Field(..., description="Clé API Mistral (obligatoire)")
    chat_model: str = Field("mistral-small-latest", description="Modèle de génération")
    embedding_model: str = Field("mistral-embed", description="Modèle d'embeddings")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Créativité du LLM")
    
    # --- Paramètres RAG ---
    chunk_size: int = Field(1000, gt=0, description="Taille d'un chunk (caractères)")
    chunk_overlap: int = Field(150, ge=0, description="Chevauchement entre chunks")
    top_k: int = Field(4, gt=0, description="Nombre de passages récupérés")
    collection_name: str = Field("documents", description="Nom de la collection Chroma")

    # --- Chemins ---
    data_dir: Path = Field(PROJECT_ROOT / "data")
    upload_dir: Path = Field(PROJECT_ROOT / "data" / "uploads")
    chroma_dir: Path = Field(PROJECT_ROOT / "data" / "chroma")

    # --- Application ---
    app_env: str = Field("development")
    log_level: str = Field("INFO")

    @model_validator(mode="after")
    def _validate_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size")
        return self

    def ensure_directories(self) -> None:
        """Crée les dossiers de données s'ils n'existent pas."""
        for path in (self.data_dir, self.upload_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Retourne une instance unique de Settings (singleton via cache)."""
    settings = Settings()
    settings.ensure_directories()
    return settings
# --- Mistral AI (ajouts) ---
    llm_max_retries: int = Field(3, ge=0, description="Tentatives LLM en cas d'échec API")
    llm_timeout: int = Field(60, gt=0, description="Timeout d'un appel LLM (s)")
