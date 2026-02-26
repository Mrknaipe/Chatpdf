import streamlit as st
import tempfile, os

from rag_pipeline import load_and_split, build_vectorstore, ChatPDFRAG

st.set_page_config(page_title="ChatPDF — RAG (Ollama local)", layout="wide")
st.title("📄 ChatPDF — RAG (Ollama en local)")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Charger un PDF", type="pdf")

    ollama_model = st.text_input("Modèle Ollama", value="llama3.2")
    chunk_size = st.slider("Taille des chunks", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Overlap des chunks", 0, 300, 150, step=50)
    k_retrieval = st.slider("Top-k passages", 1, 8, 4)
    timeout = st.slider("Timeout Ollama (s)", 30, 600, 180, step=30)

    process_btn = st.button("🔄 Indexer le document")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "history" not in st.session_state:
    st.session_state.history = []

if process_btn and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Lecture + chunking + indexation FAISS..."):
        chunks = load_and_split(tmp_path, chunk_size, chunk_overlap)
        vectorstore, _ = build_vectorstore(chunks)
        st.session_state.rag = ChatPDFRAG(
            vectorstore=vectorstore,
            ollama_model=ollama_model,
            k=k_retrieval,
            timeout=timeout
        )

    os.unlink(tmp_path)
    st.success(f"✅ Document indexé — {len(chunks)} chunks.")

if st.session_state.rag:
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    if question := st.chat_input("Posez une question..."):
        st.chat_message("user").write(question)

        with st.spinner("Recherche + génération (Ollama)..."):
            answer, sources = st.session_state.rag.ask(question)

        st.chat_message("assistant").write(answer)

        with st.expander("📄 Extraits sources utilisés"):
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] Page {page}**\n\n> {doc.page_content[:350]}...")

        st.session_state.history += [("user", question), ("assistant", answer)]
else:
    st.info("Chargez un PDF, puis cliquez sur « Indexer le document ».")
