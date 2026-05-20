import streamlit as st
import tempfile
import os
import shutil
import json
import time
import numpy as np
from rag_pipeline import load_and_split, build_vectorstore, ChatPDFRAG
from image_analyzer import analyze_pdf_images

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

if "initialized" not in st.session_state:
    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)
        print("--- faiss_index cleaned ---")
    st.session_state.initialized = True

st.set_page_config(page_title="ChatPDF — RAG (Ollama local)", layout="wide")
st.title("📄 ChatPDF — RAG (Local Ollama)")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    ollama_model = st.text_input("Ollama model (text)", value="llama3.2")
    vision_model = st.text_input("Ollama Vision model", value="llama3.2-vision:11b")

    chunk_size = st.slider("Chunk size", 300, 1500, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 150, step=50)
    k_retrieval = st.slider("Top-k passages", 1, 8, 4)
    timeout = st.slider("Timeout Ollama (s)", 30, 600, 180, step=30)

    analyze_images = st.checkbox("🖼️ Analyze images / diagrams", value=False)
    process_btn = st.button("🔄 Index documents")

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
if "parent_store" not in st.session_state:
    st.session_state.parent_store = {}

if process_btn and uploaded_files:
    st.session_state.chunks_by_doc = {}
    st.session_state.indexed_docs = []
    st.session_state.doc_stats = {}
    st.session_state.parent_store = {}

    all_chunks = []
    total_text_chunks = 0
    total_image_chunks = 0
    temp_paths = []

    with st.spinner("Reading + chunking + indexing..."):
        for doc_idx, uploaded_file in enumerate(uploaded_files, start=1):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
                temp_paths.append(tmp_path)

            pdf_name = uploaded_file.name

            text_chunks, parent_store = load_and_split(
                pdf_path=tmp_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_file=pdf_name,
                doc_id=str(doc_idx)
            )
            st.session_state.parent_store.update(parent_store)

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
        f"✅ {len(st.session_state.indexed_docs)} document(s) indexed — "
        f"{total_text_chunks} text chunks + {total_image_chunks} image chunks."
    )

    if analyze_images:
        with st.expander("📊 Image analysis details"):
            for doc_name, stats in st.session_state.doc_stats.items():
                st.markdown(
                    f"- **{doc_name}** : "
                    f"{stats['text_chunks']} text chunks, "
                    f"{stats['image_chunks']} image chunks, "
                    f"{stats['candidate_pages']} candidate page(s), "
                    f"{stats['analyzed_pages']} analyzed page(s)."
                )

tab1, tab2 = st.tabs(["💬 Chat", "📊 Model Evaluation"])
with tab1:
    if not st.session_state.rag:
        st.info("Upload one or more PDFs, then click \"Index documents\".")
    else:
        st.subheader("Indexed documents")
        selected_files = st.multiselect(
            "Limit search to specific PDFs",
            options=st.session_state.indexed_docs,
            default=st.session_state.indexed_docs
        )

        for role, msg in st.session_state.history:
            st.chat_message(role).write(msg)

        if question := st.chat_input("Ask a question..."):
            st.chat_message("user").write(question)

            with st.spinner("Retrieval + generation (Ollama)..."):
                answer, sources, grouped_sources = st.session_state.rag.ask(
                    question,
                    selected_files=selected_files,
                    parent_store=st.session_state.parent_store,
                    history=st.session_state.history
                )

            st.chat_message("assistant").write(answer)

            with st.expander("📄 Sources used"):
                for doc_name, docs in grouped_sources.items():
                    st.markdown(f"### {doc_name}")
                    for i, doc in enumerate(docs, 1):
                        page = doc.metadata.get("page", "?")
                        content_type = doc.metadata.get("content_type", "text")
                        label = "Image/Diagram" if content_type == "image" else "Text"
                        st.markdown(
                            f"**[{i}] {label} — Page {page + 1 if isinstance(page, int) else page}**\n\n"
                            f"> {doc.page_content[:500]}..."
                        )

            st.session_state.history += [("user", question), ("assistant", answer)]

with tab2:
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    st.markdown("### Model Quality Evaluation")

    EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
    EVAL_PDF = os.path.join(EVAL_DIR, "electricite.pdf")
    EVAL_JSON = os.path.join(EVAL_DIR, "test.json")

    if not os.path.exists(EVAL_PDF) or not os.path.exists(EVAL_JSON):
        st.error("❌ Missing files in the evaluation/ folder")
    else:
        st.success("✅ Evaluation files detected")

        with open(EVAL_JSON, "r", encoding="utf-8") as f:
            test = json.load(f)

        st.info(f"{len(test)} reference question(s) loaded")

        with st.expander("📋 View test questions"):
            for i, case in enumerate(test, 1):
                st.markdown(f"**Q{i}:** {case['question']}")
                st.markdown(f"**Expected answer:** {case['expected_answer']}")
                st.divider()

        n_runs = st.slider(
            label="Number of repetitions per question",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )


        if st.button("🚀 Launch evaluation"):

            # Index electricite.pdf separately if not already done
            if "eval_rag" not in st.session_state:
                with st.spinner("Indexing reference document..."):
                    eval_chunks, eval_parent_store = load_and_split(
                        pdf_path=EVAL_PDF,
                        chunk_size=800,
                        chunk_overlap=150,
                        source_file="electricite.pdf",
                        doc_id="eval"
                    )
                    eval_vectorstore, _ = build_vectorstore(eval_chunks)
                    st.session_state.eval_rag = ChatPDFRAG(
                        vectorstore=eval_vectorstore,
                        ollama_model=ollama_model,
                        k=4,
                        timeout=timeout
                    )
                    st.session_state.eval_parent_store = eval_parent_store

            embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

            results = []
            progress = st.progress(0)
            status = st.empty()
            total_steps = len(test) * n_runs

            for idx, case in enumerate(test):
                question = case["question"]
                reference = case["expected_answer"]

                status.text(f"Question {idx + 1}/{len(test)}: {question[:60]}...")

                run_results = []

                for run in range(n_runs):
                    status.text(
                        f"Question {idx + 1}/{len(test)} — "
                        f"Run {run + 1}/{n_runs}: {question[:50]}..."
                    )

                    start = time.time()
                    answer, sources, _ = st.session_state.eval_rag.ask(
                        question,
                        parent_store=st.session_state.eval_parent_store
                    )
                    elapsed = time.time() - start

                    rouge_scores = scorer.score(reference, answer)
                    vec1 = embed_model.encode([answer])
                    vec2 = embed_model.encode([reference])
                    similarity = float(cosine_similarity(vec1, vec2)[0][0])

                    run_results.append({
                        "run": run + 1,
                        "answer": answer,
                        "rouge1": round(rouge_scores['rouge1'].fmeasure, 3),
                        "rouge2": round(rouge_scores['rouge2'].fmeasure, 3),
                        "rougeL": round(rouge_scores['rougeL'].fmeasure, 3),
                        "similarity": round(similarity, 3),
                        "response_time_s": round(elapsed, 2),
                        "refused": "cannot find" in answer.lower(),
                    })

                    progress.progress((idx * n_runs + run + 1) / total_steps)

                results.append({
                    "question": question,
                    "expected": reference,
                    "runs": run_results,
                    "rouge1_mean": round(np.mean([r["rouge1"] for r in run_results]), 3),
                    "rouge1_std": round(np.std([r["rouge1"] for r in run_results]), 3),
                    "rouge2_mean": round(np.mean([r["rouge2"] for r in run_results]), 3),
                    "rougeL_mean": round(np.mean([r["rougeL"] for r in run_results]), 3),
                    "similarity_mean": round(np.mean([r["similarity"] for r in run_results]), 3),
                    "similarity_std": round(np.std([r["similarity"] for r in run_results]), 3),
                    "time_mean": round(np.mean([r["response_time_s"] for r in run_results]), 2),
                    "refused_rate": round(sum(r["refused"] for r in run_results) / n_runs, 3),
                })

            status.text("✅ Evaluation complete!")

            # Global results
            st.markdown("---")
            st.markdown("### Global Results")

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Avg ROUGE-1", round(np.mean([r["rouge1_mean"] for r in results]), 3))
            col2.metric("Avg ROUGE-2", round(np.mean([r["rouge2_mean"] for r in results]), 3))
            col3.metric("Avg ROUGE-L", round(np.mean([r["rougeL_mean"] for r in results]), 3))
            col4.metric("Avg Similarity", round(np.mean([r["similarity_mean"] for r in results]), 3))
            col5.metric("Avg Time", f"{round(np.mean([r['time_mean'] for r in results]), 1)}s")
            col6.metric("Refusal Rate", f"{round(np.mean([r['refused_rate'] for r in results]) * 100)}%")

            # Per question detail
            st.markdown("### Detail per question")

            for r in results:
                if r["similarity_mean"] >= 0.8:
                    icon = "🟢"
                elif r["similarity_mean"] >= 0.6:
                    icon = "🟡"
                else:
                    icon = "🔴"

                with st.expander(f"{icon} Q: {r['question'][:70]}..."):

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Avg ROUGE-1", r["rouge1_mean"])
                    col2.metric("Avg Similarity", r["similarity_mean"])
                    col3.metric(
                        "Stability (std)",
                        r["similarity_std"],
                        help="Closer to 0 = stable answers"
                    )
                    col4.metric("Refusal Rate", f"{round(r['refused_rate'] * 100)}%")

                    st.markdown(f"**Expected answer:**\n> {r['expected']}")

                    st.markdown("**Run details:**")
                    for run in r["runs"]:
                        st.markdown(
                            f"- **Run {run['run']}** — "
                            f"ROUGE-1: `{run['rouge1']}` | "
                            f"Similarity: `{run['similarity']}` | "
                            f"Time: `{run['response_time_s']}s` | "
                            f"{'⚠️ Refused' if run['refused'] else '✅'}"
                        )
                        with st.expander(f"View answer for run {run['run']}"):
                            st.write(run["answer"])

            # Export
            st.markdown("---")
            st.download_button(
                label="⬇️ Download results (JSON)",
                data=json.dumps(results, ensure_ascii=False, indent=2),
                file_name="evaluation_results.json",
                mime="application/json"
            )