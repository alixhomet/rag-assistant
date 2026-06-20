"""Tests du découpage en chunks."""

import pytest
from langchain_core.documents import Document

from rag_assistant.config import Settings
from rag_assistant.core.chunker import TextChunker
from rag_assistant.utils.exceptions import ChunkingError


def _settings(**overrides) -> Settings:
    """Construit des Settings de test avec une clé factice."""
    base = dict(mistral_api_key="test-key", chunk_size=100, chunk_overlap=20)
    base.update(overrides)
    return Settings(**base)


def test_un_texte_long_produit_plusieurs_chunks():
    long_text = "Phrase de test sur le RAG. " * 30  # ~810 caractères
    docs = [Document(page_content=long_text, metadata={"source": "doc.pdf", "page": 1})]

    chunks = TextChunker(_settings()).split(docs)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)


def test_les_metadonnees_sont_preservees_et_enrichies():
    docs = [Document(page_content="x" * 250, metadata={"source": "doc.pdf", "page": 4})]

    chunks = TextChunker(_settings()).split(docs)

    assert chunks[0].metadata["source"] == "doc.pdf"
    assert chunks[0].metadata["page"] == 4
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_id"] == "doc.pdf::p4::c0"


def test_les_chunk_ids_sont_uniques():
    docs = [Document(page_content="mot " * 200, metadata={"source": "doc.pdf", "page": 1})]

    chunks = TextChunker(_settings()).split(docs)

    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))   # tous distincts


def test_liste_vide_leve_une_erreur():
    with pytest.raises(ChunkingError):
        TextChunker(_settings()).split([])
