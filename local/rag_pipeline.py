from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ollama_client import OllamaClient

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")

RAG_TEMPLATE = """You are an expert assistant for document analysis.
Answer ONLY from the excerpts provided below.
The excerpts may come either from PDF text or from an image/diagram description.
When information comes from a diagram or an image, state it clearly.
If the answer is not in the excerpts, respond exactly with:
"I cannot find this information in the document."

Excerpts:
{context}

Question: {question}

Answer:
"""

def load_and_split(pdf_path: str, chunk_size=800, chunk_overlap=150, source_file=None, doc_id=None):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Splitter PARENT — grands chunks pour le contexte du LLM
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Splitter ENFANT — petits chunks pour la recherche FAISS
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Découpe en parents
    parent_chunks = parent_splitter.split_documents(documents)

    parent_store = {}   # dictionnaire parent_id → chunk parent
    all_child_chunks = []  # liste de tous les enfants à indexer dans FAISS

    for parent_idx, parent_chunk in enumerate(parent_chunks):
        parent_id = f"{doc_id}_p{parent_idx}"

        # Métadonnées du parent
        parent_chunk.metadata.update({
            "source_file": source_file or parent_chunk.metadata.get("source", "unknown"),
            "doc_id": doc_id or "unknown",
            "parent_id": parent_id,
            "chunk_index": parent_idx,
            "chunk_type": "parent",
            "content_type": "text",
        })

        # Stocke le parent en mémoire
        parent_store[parent_id] = parent_chunk

        # Découpe le parent en enfants
        child_chunks = child_splitter.split_documents([parent_chunk])

        for child_idx, child_chunk in enumerate(child_chunks):
            child_chunk.metadata.update({
                "source_file": source_file or child_chunk.metadata.get("source", "unknown"),
                "doc_id": doc_id or "unknown",
                "parent_id": parent_id,
                "child_id": f"{parent_id}_c{child_idx}",
                "chunk_index": parent_idx,
                "chunk_type": "child",
                "content_type": "text",
            })
            all_child_chunks.append(child_chunk)

    return all_child_chunks, parent_store


def build_vectorstore(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # Seuls les enfants sont vectorisés dans FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_DIR)
    return vectorstore, embeddings


def load_vectorstore(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def format_context(docs, max_chars=7000):
    parts = []
    total = 0

    for d in docs:
        page = d.metadata.get("page", "?")
        source_file = d.metadata.get("source_file", "Unknown document")
        content_type = d.metadata.get("content_type", "text")

        prefix = "DIAGRAM/IMAGE" if content_type == "image" else "TEXT"
        chunk = f"[{prefix} | Document: {source_file} | Page: {page}] {d.page_content.strip()}"

        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)


class ChatPDFRAG:
    def __init__(self, vectorstore, ollama_model="llama3.2", k=4, timeout=180):
        self.vectorstore = vectorstore
        self.k = k
        self.timeout = timeout
        self.ollama = OllamaClient(model=ollama_model)
        self.ollama.verify_ollama()

    def ask(self, question: str, selected_files=None, parent_store=None):
        filter_dict = None
        if selected_files:
            filter_dict = {"source_file": {"$in": selected_files}}

        # Recherche les enfants les plus proches dans FAISS
        child_docs = self.vectorstore.similarity_search(
            question,
            k=max(self.k * 5, 20),
            filter=filter_dict,
            fetch_k=max(self.k * 8, 30),
        )
        child_docs = child_docs[:self.k]

        # Remonte aux parents via parent_id
        context_docs = []
        seen_parents = set()

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")

            if parent_id and parent_store and parent_id not in seen_parents:
                parent = parent_store.get(parent_id)
                if parent:
                    context_docs.append(parent)
                    seen_parents.add(parent_id)
                else:
                    # parent introuvable → utilise l'enfant directement
                    context_docs.append(child)
            else:
                # pas de parent_store → comportement classique
                context_docs.append(child)

        context = format_context(context_docs)
        prompt = RAG_TEMPLATE.format(context=context, question=question)
        answer = self.ollama.call_ollama(prompt, timeout=self.timeout)

        grouped_sources = defaultdict(list)
        for doc in context_docs:
            doc_name = doc.metadata.get("source_file", "Unknown document")
            grouped_sources[doc_name].append(doc)

        return answer, context_docs, dict(grouped_sources)