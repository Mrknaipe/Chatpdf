import streamlit as st
import tempfile
import os

from rag_pipeline import load_and_split, build_vectorstore, ChatPDFRAG
from image_analyzer import analyze_pdf_images

st.set_page_config(page_title="ChatPDF — RAG (Ollama local)", layout="wide")
st.title("📄 ChatPDF — RAG (Ollama en local)")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_files = st.file_uploader(
        "Charger un ou plusieurs PDF",
        type="pdf",
        accept_multiple_files=True
    )

    ollama_model = st.text_input("Modèle Ollama (texte)", value="llama3.2")
    vision_model = st.text_input("Modèle Ollama Vision", value="llama3.2-vision:11b")

    chunk_size = st.slider("Taille des chunks", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Overlap des chunks", 0, 300, 150, step=50)
    k_retrieval = st.slider("Top-k passages", 1, 8, 4)
    timeout = st.slider("Timeout Ollama (s)", 30, 600, 180, step=30)

    analyze_images = st.checkbox("🖼️ Analyser les images / schémas", value=False)
    process_btn = st.button("🔄 Indexer les documents")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "history" not in st.session_state:
    st.session_state.history = []
if "chunks_by_doc" not in st.session_state:
    st.session_state.chunks_by_doc = {}
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

if process_btn and uploaded_files:
    st.session_state.chunks_by_doc = {}
    st.session_state.indexed_docs = []
    st.session_state.doc_stats = {}

    all_chunks = []
    total_text_chunks = 0
    total_image_chunks = 0
    temp_paths = []

    with st.spinner("Lecture + chunking + indexation..."):
        for doc_idx, uploaded_file in enumerate(uploaded_files, start=1):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
                temp_paths.append(tmp_path)

            pdf_name = uploaded_file.name

            text_chunks = load_and_split(
                pdf_path=tmp_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_file=pdf_name,
                doc_id=str(doc_idx)
            )

            image_chunks = []
            image_stats = {
                "candidate_pages": 0,
                "analyzed_pages": 0,
                "image_chunks": 0
            }

            if analyze_images:
                image_chunks, image_stats = analyze_pdf_images(
                    pdf_path=tmp_path,
                    source_file=pdf_name,
                    doc_id=str(doc_idx),
                    vision_model=vision_model,
                    timeout=timeout
                )

            doc_chunks = text_chunks + image_chunks
            all_chunks.extend(doc_chunks)

            total_text_chunks += len(text_chunks)
            total_image_chunks += len(image_chunks)

            st.session_state.chunks_by_doc[str(doc_idx)] = {
                "document_name": pdf_name,
                "text_chunks": text_chunks,
                "image_chunks": image_chunks,
                "total_chunks": len(doc_chunks),
            }

            st.session_state.doc_stats[pdf_name] = {
                "text_chunks": len(text_chunks),
                "image_chunks": len(image_chunks),
                **image_stats
            }

            st.session_state.indexed_docs.append(pdf_name)

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
        f"✅ {len(st.session_state.indexed_docs)} document(s) indexé(s) — "
        f"{total_text_chunks} chunks texte + {total_image_chunks} chunks image."
    )

    if analyze_images:
        with st.expander("📊 Détail analyse image"):
            for doc_name, stats in st.session_state.doc_stats.items():
                st.markdown(
                    f"- **{doc_name}** : "
                    f"{stats['text_chunks']} chunks texte, "
                    f"{stats['image_chunks']} chunks image, "
                    f"{stats['candidate_pages']} page(s) candidate(s), "
                    f"{stats['analyzed_pages']} page(s) analysée(s)."
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

        with st.expander("📄 Sources utilisées"):
            for doc_name, docs in grouped_sources.items():
                st.markdown(f"### {doc_name}")
                for i, doc in enumerate(docs, 1):
                    page = doc.metadata.get("page", "?")
                    content_type = doc.metadata.get("content_type", "text")
                    label = "Image/Schéma" if content_type == "image" else "Texte"

                    st.markdown(
                        f"**[{i}] {label} — Page {page + 1 if isinstance(page, int) else page}**\n\n"
                        f"> {doc.page_content[:500]}..."
                    )

        st.session_state.history += [("user", question), ("assistant", answer)]
else:
    st.info("Chargez un ou plusieurs PDF, puis cliquez sur « Indexer les documents ».")