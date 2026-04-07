from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate


# ── 1. CHARGEMENT & SEGMENTATION DU PDF ─────────────────────────────────────
def load_and_split(pdf_path: str, chunk_size=800, chunk_overlap=150):
    """Charge un PDF et le découpe en chunks avec overlap."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] {len(documents)} page(s) → {len(chunks)} chunks")
    return chunks


# ── 2. CRÉATION DES EMBEDDINGS & VECTORSTORE FAISS ──────────────────────────
def build_vectorstore(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Transforme les chunks en vecteurs et les stocke dans FAISS."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("[INFO] VectorStore FAISS créé et sauvegardé.")
    return vectorstore, embeddings


def load_vectorstore(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Charge un VectorStore FAISS existant."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )


# ── 3. PROMPT CONTRAINT : LE LLM NE RÉPOND QU'À PARTIR DU CONTEXTE ──────────
RAG_PROMPT = PromptTemplate(
    template="""Tu es un assistant expert en analyse de documents.
Réponds UNIQUEMENT à partir des extraits du document fournis ci-dessous.
Si la réponse ne se trouve pas dans les extraits, réponds :
"Je ne trouve pas cette information dans le document."

Extraits du document :
{context}

Question : {question}

Réponse :""",
    input_variables=["context", "question"]
)


# ── 4. CHAÎNE RAG COMPLÈTE ───────────────────────────────────────────────────
def build_rag_chain(vectorstore, repo_id="mistralai/Mistral-7B-Instruct-v0.2"):
    """Construit la chaîne RetrievalQA avec un LLM HuggingFace."""
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Top-4 chunks les plus pertinents
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # Injecte tous les chunks dans un seul prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True  # Retourne les sources pour transparence
    )
    return chain


# ── 5. REQUÊTE UTILISATEUR ───────────────────────────────────────────────────
def ask(chain, question: str):
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]

    print(f"\n🔍 Question : {question}")
    print(f"\n💬 Réponse :\n{answer}")
    print("\n📄 Sources utilisées :")
    for i, doc in enumerate(sources, 1):
        page = doc.metadata.get("page", "?")
        print(f"  [{i}] Page {page} — {doc.page_content[:120]}...")
    return answer
