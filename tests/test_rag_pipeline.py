"""Test d'intégration du pipeline complet (sans appels réseau)."""

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_assistant.config import Settings
from rag_assistant.core.embedder import Embedder
from rag_assistant.core.llm import AnswerGenerator
from rag_assistant.core.rag_pipeline import RAGPipeline
from rag_assistant.core.vector_store import VectorStore
from rag_assistant.core.prompts import NO_CONTEXT_ANSWER


class FakeEmbeddings(Embeddings):
    """Vecteurs prévisibles ; la dimension de base évite tout vecteur nul."""

    def _vec(self, text: str) -> list[float]:
        t = text.lower()
        return [
            1.0,                                              # base
            1.0 if "rag" in t else 0.0,
            1.0 if "semantique" in t or "sémantique" in t else 0.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _pipeline(tmp_path, llm_response: str = "Réponse de test.") -> RAGPipeline:
    settings = Settings(mistral_api_key="test-key", chroma_dir=tmp_path,
                        collection_name="pipe_test")
    embedder = Embedder(settings=settings, embeddings=FakeEmbeddings())
    vector_store = VectorStore(settings=settings, embedder=embedder)
    generator = AnswerGenerator(settings=settings,
                                chat_model=FakeListChatModel(responses=[llm_response]))
    return RAGPipeline(settings=settings, vector_store=vector_store, generator=generator)


def test_ingestion_indexe_le_pdf(tmp_path, sample_pdf_bytes):
    pipe = _pipeline(tmp_path)
    result = pipe.ingest_pdf(sample_pdf_bytes, source_name="cours.pdf")

    assert result.pages == 2
    assert result.chunks_added >= 2
    assert pipe.list_documents() == ["cours.pdf"]


def test_reponse_avec_sources(tmp_path, sample_pdf_bytes):
    pipe = _pipeline(tmp_path, llm_response="Le RAG combine recherche et génération.")
    pipe.ingest_pdf(sample_pdf_bytes, source_name="cours.pdf")

    out = pipe.answer("Qu'est-ce que le RAG ?", top_k=2)

    assert out.answer == "Le RAG combine recherche et génération."
    assert len(out.sources) >= 1
    assert ("cours.pdf", 1) in out.unique_sources


def test_question_hors_sujet_renvoie_le_repli(tmp_path, sample_pdf_bytes):
    pipe = _pipeline(tmp_path)
    pipe.ingest_pdf(sample_pdf_bytes, source_name="cours.pdf")

    # Seuil très élevé : aucun passage retenu -> réponse de repli, sans LLM
    out = pipe.answer("Quelle est la recette de la tarte ?", score_threshold=0.99)
    assert out.answer == NO_CONTEXT_ANSWER


def test_reindexation_sans_doublon(tmp_path, sample_pdf_bytes):
    pipe = _pipeline(tmp_path)
    pipe.ingest_pdf(sample_pdf_bytes, source_name="cours.pdf")
    count_1 = pipe.document_count()
    pipe.ingest_pdf(sample_pdf_bytes, source_name="cours.pdf", replace=True)
    assert pipe.document_count() == count_1


def test_suppression_de_document(tmp_path, sample_pdf_bytes):
    pipe = _pipeline(tmp_path)
    pipe.ingest_pdf(sample_pdf_bytes, source_name="cours.pdf")
    pipe.delete_document("cours.pdf")
    assert pipe.document_count() == 0
    assert pipe.list_documents() == []
