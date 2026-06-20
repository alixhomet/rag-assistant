"""Tests de la recherche sémantique (ChromaDB réel, embeddings prévisibles)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.config import Settings
from rag_assistant.core.embedder import Embedder
from rag_assistant.core.retriever import Retriever
from rag_assistant.core.vector_store import VectorStore
from rag_assistant.utils.exceptions import RetrievalError


class FakeEmbeddings(Embeddings):
    """Vecteurs déterministes : chaque mot-clé active une dimension."""

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        return [
            1.0 if "chat" in t else 0.0,
            1.0 if "chien" in t else 0.0,
            1.0 if "python" in t else 0.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _chunk(text: str, source: str, page: int, idx: int) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page,
                  "chunk_index": idx, "chunk_id": f"{source}::p{page}::c{idx}"},
    )


@pytest.fixture
def retriever(tmp_path) -> Retriever:
    settings = Settings(mistral_api_key="test-key", chroma_dir=tmp_path,
                        collection_name="ret_test", top_k=3)
    embedder = Embedder(settings=settings, embeddings=FakeEmbeddings())
    vs = VectorStore(settings=settings, embedder=embedder)
    vs.add_documents([
        _chunk("Le chat dort sur le canapé.", "animaux.pdf", 1, 0),
        _chunk("Le chien aboie fort.", "animaux.pdf", 2, 1),
        _chunk("Python est un langage de programmation.", "tech.pdf", 1, 2),
    ])
    return Retriever(vector_store=vs, settings=settings)


def test_retrouve_le_passage_pertinent(retriever):
    results = retriever.retrieve("chat", top_k=1)
    assert results[0].source == "animaux.pdf"
    assert "chat" in results[0].content.lower()
    assert results[0].score > 0.9            # quasi-identique


def test_top_k_limite_les_resultats(retriever):
    assert len(retriever.retrieve("chat", top_k=2)) == 2


def test_le_seuil_filtre_les_passages_faibles(retriever):
    results = retriever.retrieve("chat", top_k=3, score_threshold=0.5)
    assert len(results) == 1                 # seul le passage "chat" dépasse 0.5


def test_filtre_par_source(retriever):
    results = retriever.retrieve("chat", top_k=3, sources=["tech.pdf"])
    assert all(r.source == "tech.pdf" for r in results)


def test_question_vide_leve_une_erreur(retriever):
    with pytest.raises(RetrievalError):
        retriever.retrieve("   ")
