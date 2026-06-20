"""Extraction de texte depuis des fichiers PDF, page par page."""

import re
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_assistant.utils.exceptions import PDFExtractionError
from rag_assistant.utils.logger import get_logger

logger = get_logger("pdf_loader")

"""Extraction de texte depuis des fichiers PDF, page par page."""

import re
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag_assistant.utils.exceptions import PDFExtractionError
from rag_assistant.utils.logger import get_logger

logger = get_logger("pdf_loader")


class PDFLoader:
    """Charge un PDF et en extrait le texte sous forme de Documents LangChain.

    Chaque page non vide devient un Document indépendant, ce qui permet de
    tracer précisément la source (fichier + page) jusqu'à la réponse finale.
    """

    def __init__(self, min_chars_per_page: int = 1) -> None:
        # En dessous de ce seuil, une page est considérée comme "vide"
        self._min_chars = min_chars_per_page

    def load(self, data: bytes, source_name: str) -> list[Document]:
        """Extrait le texte d'un PDF fourni sous forme d'octets."""
        if not data:
            raise PDFExtractionError(f"Fichier vide : {source_name}")

        try:
            reader = PdfReader(BytesIO(data))
        except PdfReadError as exc:
            raise PDFExtractionError(f"PDF illisible ou corrompu : {source_name}") from exc

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")  # tente un mot de passe vide
            except Exception as exc:
                raise PDFExtractionError(f"PDF chiffré : {source_name}") from exc
            if not decrypted:
                raise PDFExtractionError(
                    f"PDF protégé par mot de passe : {source_name}"
                )

        total_pages = len(reader.pages)
        documents: list[Document] = []
        skipped = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = self._clean(page.extract_text() or "")
            if len(text) < self._min_chars:
                skipped += 1
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_number,
                        "total_pages": total_pages,
                    },
                )
            )

        if not documents:
            raise PDFExtractionError(
                f"Aucun texte extractible dans '{source_name}'. "
                "Le PDF est probablement scanné (image) et nécessiterait de l'OCR."
            )

        logger.info(
            "Extraction de '%s' : %d/%d page(s) avec texte (%d ignorée(s))",
            source_name, len(documents), total_pages, skipped,
        )
        return documents

    def load_from_path(self, path: str | Path) -> list[Document]:
        """Variante pratique : charge un PDF depuis le disque (tests, CLI)."""
        path = Path(path)
        if not path.exists():
            raise PDFExtractionError(f"Fichier introuvable : {path}")
        if path.suffix.lower() != ".pdf":
            raise PDFExtractionError(f"Le fichier n'est pas un PDF : {path}")
        return self.load(path.read_bytes(), source_name=path.name)

    @staticmethod
    def _clean(text: str) -> str:
        """Nettoyage léger : supprime les lignes vides et les espaces multiples."""
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        cleaned = "\n".join(lines)
        return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
class PDFLoader:
    """Charge un PDF et en extrait le texte sous forme de Documents LangChain.

    Chaque page non vide devient un Document indépendant, ce qui permet de
    tracer précisément la source (fichier + page) jusqu'à la réponse finale.
    """

    def __init__(self, min_chars_per_page: int = 1) -> None:
        # En dessous de ce seuil, une page est considérée comme "vide"
        self._min_chars = min_chars_per_page

    def load(self, data: bytes, source_name: str) -> list[Document]:
        """Extrait le texte d'un PDF fourni sous forme d'octets."""
        if not data:
            raise PDFExtractionError(f"Fichier vide : {source_name}")

        try:
            reader = PdfReader(BytesIO(data))
        except PdfReadError as exc:
            raise PDFExtractionError(f"PDF illisible ou corrompu : {source_name}") from exc

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")  # tente un mot de passe vide
            except Exception as exc:
                raise PDFExtractionError(f"PDF chiffré : {source_name}") from exc
            if not decrypted:
                raise PDFExtractionError(
                    f"PDF protégé par mot de passe : {source_name}"
                )

        total_pages = len(reader.pages)
        documents: list[Document] = []
        skipped = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = self._clean(page.extract_text() or "")
            if len(text) < self._min_chars:
                skipped += 1
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_number,
                        "total_pages": total_pages,
                    },
                )
            )

        if not documents:
            raise PDFExtractionError(
                f"Aucun texte extractible dans '{source_name}'. "
                "Le PDF est probablement scanné (image) et nécessiterait de l'OCR."
            )

        logger.info(
            "Extraction de '%s' : %d/%d page(s) avec texte (%d ignorée(s))",
            source_name, len(documents), total_pages, skipped,
        )
        return documents

    def load_from_path(self, path: str | Path) -> list[Document]:
        """Variante pratique : charge un PDF depuis le disque (tests, CLI)."""
        path = Path(path)
        if not path.exists():
            raise PDFExtractionError(f"Fichier introuvable : {path}")
        if path.suffix.lower() != ".pdf":
            raise PDFExtractionError(f"Le fichier n'est pas un PDF : {path}")
        return self.load(path.read_bytes(), source_name=path.name)

    @staticmethod
    def _clean(text: str) -> str:
        """Nettoyage léger : supprime les lignes vides et les espaces multiples."""
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        cleaned = "\n".join(lines)
        return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
