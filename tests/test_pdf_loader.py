"""Tests du chargeur de PDF."""

import pytest

from rag_assistant.core.pdf_loader import PDFLoader
from rag_assistant.utils.exceptions import PDFExtractionError


def test_load_extrait_les_pages(sample_pdf_bytes):
    docs = PDFLoader().load(sample_pdf_bytes, source_name="test.pdf")

    assert len(docs) == 2
    assert docs[0].metadata == {"source": "test.pdf", "page": 1, "total_pages": 2}
    assert docs[1].metadata["page"] == 2
    assert "RAG" in docs[0].page_content


def test_octets_vides_leve_une_erreur():
    with pytest.raises(PDFExtractionError):
        PDFLoader().load(b"", source_name="vide.pdf")


def test_octets_invalides_levent_une_erreur():
    with pytest.raises(PDFExtractionError):
        PDFLoader().load(b"ceci n'est pas un pdf", source_name="faux.pdf")


def test_nettoyage_normalise_les_espaces():
    brut = "  Bonjour    le    monde  \n\n\n  RAG  "
    assert PDFLoader._clean(brut) == "Bonjour le monde\nRAG"
