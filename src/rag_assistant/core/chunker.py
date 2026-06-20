"""Découpage des Documents en chunks pour l'indexation vectorielle."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.config import Settings, get_settings
from rag_assistant.utils.exceptions import ChunkingError
from rag_assistant.utils.logger import get_logger

logger = get_logger("chunker")


class TextChunker:
    """Découpe des Documents en chunks tout en préservant leurs métadonnées.

    S'appuie sur le RecursiveCharacterTextSplitter, qui respecte la structure
    naturelle du texte (paragraphes, phrases, mots) plutôt que de couper
    arbitrairement tous les N caractères.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,   # ajoute l'offset du chunk dans la page d'origine
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Découpe une liste de Documents en chunks enrichis de métadonnées."""
        if not documents:
            raise ChunkingError("Aucun document à découper")

        chunks = self._splitter.split_documents(documents)

        if not chunks:
            raise ChunkingError("Le découpage n'a produit aucun chunk")

        # Enrichissement : un identifiant déterministe et unique par chunk
        for index, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "inconnu")
            page = chunk.metadata.get("page", "?")
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{source}::p{page}::c{index}"

        logger.info(
            "Découpage : %d page(s) -> %d chunk(s) (taille=%d, overlap=%d)",
            len(documents), len(chunks),
            self._settings.chunk_size, self._settings.chunk_overlap,
        )
        return chunks
