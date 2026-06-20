"""Tests du stockage vectoriel (ChromaDB réel, embeddings factices)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.config import Settings
from rag_assistant.core.embedder import Embedder
from rag_assistant.core.vector_store import VectorStore
from rag_assistant.utils.exceptions import VectorStoreError


class FakeEmbeddings(Embeddings):
    """Embeddings déterministes de dimension 3, sans réseau."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


def _chunk(text: str, source: str, page: int, idx: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source, "page": page,
            "chunk_index": idx, "chunk_id": f"{source}::p{page}::c{idx}",
        },
    )


@pytest.fixture
def store(tmp_path) -> VectorStore:
    settings = Settings(
        mistral_api_key="test-key",
        chroma_dir=tmp_path,
        collection_name="test_collection",
    )
    embedder = Embedder(settings=settings, embeddings=FakeEmbeddings())
    return VectorStore(settings=settings, embedder=embedder)


def test_ajout_et_comptage(store):
    store.add_documents([_chunk("bonjour", "a.pdf", 1, 0), _chunk("monde", "a.pdf", 1, 1)])
    assert store.count() == 2


def test_ajout_idempotent(store):
    chunk = [_chunk("bonjour", "a.pdf", 1, 0)]
    store.add_documents(chunk)
    store.add_documents(chunk)          # même chunk_id -> upsert, pas de doublon
    assert store.count() == 1


def test_liste_des_sources(store):
    store.add_documents([_chunk("x", "a.pdf", 1, 0), _chunk("y", "b.pdf", 1, 0)])
    assert store.list_sources() == ["a.pdf", "b.pdf"]


def test_suppression_par_source(store):
    store.add_documents([_chunk("x", "a.pdf", 1, 0), _chunk("y", "b.pdf", 1, 0)])
    assert store.delete_by_source("a.pdf") == 1
    assert store.list_sources() == ["b.pdf"]


def test_purge(store):
    store.add_documents([_chunk("x", "a.pdf", 1, 0)])
    store.clear()
    assert store.count() == 0


def test_ajout_vide_leve_une_erreur(store):
    with pytest.raises(VectorStoreError):
        store.add_documents([])
