"""Tests de la génération de réponses (sans appel LLM réel)."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from rag_assistant.config import Settings
from rag_assistant.core.llm import AnswerGenerator
from rag_assistant.core.prompts import NO_CONTEXT_ANSWER
from rag_assistant.core.retriever import RetrievedChunk
from rag_assistant.utils.exceptions import LLMError


def _settings() -> Settings:
    return Settings(mistral_api_key="test-key")


def _chunk(content="Le chat dort sur le canapé.", source="a.pdf", page=1) -> RetrievedChunk:
    return RetrievedChunk(content=content, source=source, page=page,
                          score=0.9, chunk_id=f"{source}::p{page}::c0")


def test_genere_une_reponse_avec_contexte():
    llm = FakeListChatModel(responses=["Le chat dort sur le canapé (a.pdf, page 1)."])
    answer = AnswerGenerator(settings=_settings(), chat_model=llm).generate(
        "Que fait le chat ?", [_chunk()]
    )
    assert "chat" in answer.lower()


def test_sans_contexte_aucun_appel_au_llm():
    def _interdit(_):
        raise AssertionError("Le LLM ne doit pas être appelé sans contexte")
    gen = AnswerGenerator(settings=_settings(), chat_model=RunnableLambda(_interdit))
    assert gen.generate("Une question ?", []) == NO_CONTEXT_ANSWER


def test_question_vide_leve_une_erreur():
    gen = AnswerGenerator(settings=_settings(), chat_model=FakeListChatModel(responses=["x"]))
    with pytest.raises(LLMError):
        gen.generate("   ", [_chunk()])


def test_le_contexte_inclut_les_sources():
    gen = AnswerGenerator(settings=_settings(), chat_model=FakeListChatModel(responses=["x"]))
    ctx = gen._format_context([_chunk(source="rapport.pdf", page=4)])
    assert "rapport.pdf" in ctx and "page 4" in ctx


def test_une_erreur_llm_est_encapsulee():
    def _boom(_):
        raise RuntimeError("LLM indisponible")
    gen = AnswerGenerator(settings=_settings(), chat_model=RunnableLambda(_boom))
    with pytest.raises(LLMError):
        gen.generate("Une question ?", [_chunk()])
