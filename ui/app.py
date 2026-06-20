"""Interface Streamlit de l'assistant documentaire RAG."""

import streamlit as st

from rag_assistant.config import get_settings
from rag_assistant.core.rag_pipeline import RAGPipeline
from rag_assistant.core.retriever import RetrievedChunk
from rag_assistant.utils.exceptions import RagAssistantError
from rag_assistant.utils.logger import get_logger

logger = get_logger("ui")

st.set_page_config(page_title="Assistant RAG", page_icon="📚", layout="wide")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Instancie le pipeline UNE seule fois, réutilisé entre tous les reruns."""
    return RAGPipeline()


def init_state() -> None:
    """Initialise l'historique de conversation (persistant sur la session)."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # liste de {role, content, sources?}


def render_sources(sources: list[RetrievedChunk]) -> None:
    """Affiche les passages sources avec leur score de similarité."""
    with st.expander(f"📌 Sources utilisées ({len(sources)})"):
        for chunk in sources:
            st.markdown(
                f"**{chunk.source}** — page {chunk.page} · "
                f"similarité **{chunk.score:.0%}**"
            )
            preview = chunk.content[:300]
            st.caption(preview + ("…" if len(chunk.content) > 300 else ""))
            st.divider()


def render_sidebar(pipeline: RAGPipeline) -> dict:
    """Barre latérale : gestion des documents et réglages. Retourne les options."""
    st.sidebar.header("📁 Documents")

    uploaded = st.sidebar.file_uploader(
        "Importer des PDF", type=["pdf"], accept_multiple_files=True
    )
    if uploaded and st.sidebar.button("Indexer", type="primary", use_container_width=True):
        with st.spinner("Indexation en cours…"):
            for file in uploaded:
                try:
                    result = pipeline.ingest_pdf(
                        file.getvalue(), source_name=file.name, replace=True
                    )
                    st.sidebar.success(
                        f"{file.name} : {result.pages} p., {result.chunks_added} chunks"
                    )
                except RagAssistantError as exc:
                    st.sidebar.error(f"{file.name} : {exc}")

    # Liste à jour (reflète les éventuels ajouts ci-dessus)
    documents = pipeline.list_documents()
    st.sidebar.divider()
    st.sidebar.subheader(f"Indexés ({len(documents)})")
    if not documents:
        st.sidebar.caption("Aucun document indexé pour l'instant.")
    for doc in documents:
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.write(f"📄 {doc}")
        if col2.button("🗑️", key=f"del_{doc}", help=f"Supprimer {doc}"):
            pipeline.delete_document(doc)
            st.rerun()

    selected = (
        st.sidebar.multiselect(
            "Restreindre la recherche à",
            options=documents,
            default=documents,
            help="Tout sélectionner = chercher dans l'ensemble des documents.",
        )
        if documents
        else []
    )

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Réglages")
    settings = get_settings()
    top_k = st.sidebar.slider("Passages récupérés (top-k)", 1, 10, settings.top_k)
    threshold = st.sidebar.slider("Seuil de similarité", 0.0, 1.0, 0.0, 0.05)

    if st.session_state.messages and st.sidebar.button(
        "🗑️ Effacer la conversation", use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    return {"sources": selected or None, "top_k": top_k, "score_threshold": threshold or None}


def main() -> None:
    init_state()

    # Initialisation du pipeline (échoue proprement si la clé API manque)
    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error(
            "Impossible d'initialiser l'assistant. Vérifiez `MISTRAL_API_KEY` "
            f"dans votre fichier `.env`.\n\nDétail : {exc}"
        )
        st.stop()

    options = render_sidebar(pipeline)

    st.title("📚 Assistant documentaire RAG")
    st.caption(
        "Posez des questions sur vos PDF. Les réponses sont fondées "
        "uniquement sur le contenu de vos documents."
    )

    # Historique existant
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    # Nouvelle question
    if question := st.chat_input("Votre question…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not pipeline.list_documents():
                warning = "Veuillez d'abord importer et indexer au moins un PDF."
                st.warning(warning)
                st.session_state.messages.append({"role": "assistant", "content": warning})
            else:
                with st.spinner("Recherche dans vos documents…"):
                    try:
                        result = pipeline.answer(
                            question,
                            top_k=options["top_k"],
                            score_threshold=options["score_threshold"],
                            sources=options["sources"],
                        )
                    except RagAssistantError as exc:
                        error = f"Une erreur est survenue : {exc}"
                        st.error(error)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error}
                        )
                        st.stop()

                st.markdown(result.answer)
                if result.sources:
                    render_sources(result.sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer, "sources": result.sources}
                )


if __name__ == "__main__": main()
