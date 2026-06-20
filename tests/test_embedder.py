"""Tests du service d'embedding (sans appel réseau)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.config import Settings
from rag_assistant.core.embedder import Embedder
from rag_assistant.utils.exceptions import EmbeddingError


class FakeEmbeddings(Embeddings):
    """Embeddings déterministes, sans appel réseau, pour les tests."""

    def __init__(self, dim: int = 8, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("API indisponible")
        return [[float((len(t) % 5) + 1)] * self.dim for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("API indisponible")
        return [float((len(text) % 5) + 1)] * self.dim


def _embedder(**kwargs) -> Embedder:
    settings = Settings(mistral_api_key="test-key")
    return Embedder(settings=settings, embeddings=FakeEmbeddings(**kwargs))


def test_embed_texts_un_vecteur_par_texte():
    vectors = _embedder().embed_texts(["bonjour", "le RAG"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 8


def test_embed_documents_depuis_des_documents():
    docs = [Document(page_content="a", metadata={}),
            Document(page_content="bb", metadata={})]
    assert len(_embedder().embed_documents(docs)) == 2


def test_embed_query_retourne_un_vecteur():
    assert len(_embedder().embed_query("ma question")) == 8


def test_liste_vide_leve_une_erreur():
    with pytest.raises(EmbeddingError):
        _embedder().embed_texts([])


def test_requete_vide_leve_une_erreur():
    with pytest.raises(EmbeddingError):
        _embedder().embed_query("   ")


def test_echec_api_encapsule_en_embedding_error():
    with pytest.raises(EmbeddingError):
        _embedder(fail=True).embed_texts(["x"])
