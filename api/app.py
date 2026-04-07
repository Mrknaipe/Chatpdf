import streamlit as st
import tempfile, os
from rag_pipeline import load_and_split, build_vectorstore, build_rag_chain

st.set_page_config(page_title="ChatPDF — RAG", page_icon="📄", layout="wide")
st.title("📄 ChatPDF — Posez vos questions sur votre document")

# ── Sidebar : upload + configuration ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Charger un PDF", type="pdf")
    chunk_size    = st.slider("Taille des chunks", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Overlap des chunks", 0, 300, 150, step=50)
    k_retrieval   = st.slider("Nombre de chunks récupérés (k)", 1, 8, 4)
    process_btn   = st.button("🔄 Indexer le document")

# ── Initialisation session state ─────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── Indexation du PDF ────────────────────────────────────────────────────────
if process_btn and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    with st.spinner("Lecture et indexation du PDF..."):
        chunks = load_and_split(tmp_path, chunk_size, chunk_overlap)
        vectorstore, _ = build_vectorstore(chunks)
        st.session_state.chain = build_rag_chain(vectorstore)
    os.unlink(tmp_path)
    st.success(f"✅ Document indexé — {len(chunks)} chunks créés.")

# ── Interface de chat ────────────────────────────────────────────────────────
if st.session_state.chain:
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    if question := st.chat_input("Posez une question sur le document..."):
        st.chat_message("user").write(question)
        with st.spinner("Recherche et génération..."):
            result = st.session_state.chain.invoke({"query": question})
            answer  = result["result"]
            sources = result["source_documents"]

        st.chat_message("assistant").write(answer)

        # Affichage des sources (transparence RAG)
        with st.expander("📄 Extraits sources utilisés"):
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] Page {page}**\n\n> {doc.page_content[:300]}...")

        st.session_state.history += [("user", question), ("assistant", answer)]
else:
    st.info("👈 Chargez un PDF dans la barre latérale pour commencer.")
