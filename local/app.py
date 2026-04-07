import streamlit as st
import tempfile
import os

from rag_pipeline import load_and_split, build_vectorstore, ChatPDFRAG

st.set_page_config(page_title="ChatPDF — RAG (Ollama local)", layout="wide")
st.title("📄 ChatPDF — RAG (Ollama en local)")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_files = st.file_uploader(
        "Charger un ou plusieurs PDF",
        type="pdf",
        accept_multiple_files=True
    )

    ollama_model = st.text_input("Modèle Ollama", value="llama3.2")
    chunk_size = st.slider("Taille des chunks", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Overlap des chunks", 0, 300, 150, step=50)
    k_retrieval = st.slider("Top-k passages", 1, 8, 4)
    timeout = st.slider("Timeout Ollama (s)", 30, 600, 180, step=30)

    process_btn = st.button("🔄 Indexer les documents")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "history" not in st.session_state:
    st.session_state.history = []
if "chunks_by_doc" not in st.session_state:
    st.session_state.chunks_by_doc = {}
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

if process_btn and uploaded_files:
    st.session_state.chunks_by_doc = {}
    st.session_state.indexed_docs = []
    all_chunks = []
    total_chunks = 0
    temp_paths = []

    with st.spinner("Lecture + chunking + indexation FAISS..."):
        for doc_idx, uploaded_file in enumerate(uploaded_files, start=1):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
                temp_paths.append(tmp_path)

            chunks = load_and_split(
                pdf_path=tmp_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_file=uploaded_file.name,
                doc_id=str(doc_idx)
            )

            st.session_state.chunks_by_doc[str(doc_idx)] = {
                "document_name": uploaded_file.name,
                "chunks": chunks,
                "n_chunks": len(chunks),
            }

            st.session_state.indexed_docs.append(uploaded_file.name)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)

        vectorstore, _ = build_vectorstore(all_chunks)

        st.session_state.rag = ChatPDFRAG(
            vectorstore=vectorstore,
            ollama_model=ollama_model,
            k=k_retrieval,
            timeout=timeout
        )

    for path in temp_paths:
        if os.path.exists(path):
            os.unlink(path)

    st.success(
        f"✅ {len(st.session_state.chunks_by_doc)} document(s) indexé(s) — {total_chunks} chunks."
    )

if st.session_state.rag:
    st.subheader("Documents indexés")
    selected_files = st.multiselect(
        "Limiter la recherche à certains PDF",
        options=st.session_state.indexed_docs,
        default=st.session_state.indexed_docs
    )

    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    if question := st.chat_input("Posez une question..."):
        st.chat_message("user").write(question)

        with st.spinner("Recherche + génération (Ollama)..."):
            answer, sources, grouped_sources = st.session_state.rag.ask(
                question,
                selected_files=selected_files
            )

        st.chat_message("assistant").write(answer)

        with st.expander("📄 Extraits sources utilisés"):
            for doc_name, docs in grouped_sources.items():
                st.markdown(f"### {doc_name}")
                for i, doc in enumerate(docs, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f"**[{i}] Page {page + 1 if isinstance(page, int) else page}**\n\n"
                        f"> {doc.page_content[:350]}..."
                    )

        st.session_state.history += [("user", question), ("assistant", answer)]
else:
    st.info("Chargez un ou plusieurs PDF, puis cliquez sur « Indexer les documents ».")