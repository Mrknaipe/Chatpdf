from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ollama_client import OllamaClient

RAG_TEMPLATE = """Tu es un assistant expert en analyse de documents.
Réponds UNIQUEMENT à partir des extraits fournis ci-dessous.
Les extraits peuvent venir soit du texte du PDF, soit d'une description d'image/schéma.
Quand l'information vient d'un schéma ou d'une image, indique-le clairement.
Si la réponse ne se trouve pas dans les extraits, réponds exactement :
"Je ne trouve pas cette information dans le document."

Extraits :
{context}

Question : {question}

Réponse :
"""

def load_and_split(pdf_path: str, chunk_size=800, chunk_overlap=150, source_file=None, doc_id=None):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = source_file if source_file else doc.metadata.get("source", "inconnu")
        doc.metadata["doc_id"] = doc_id if doc_id else "unknown"
        doc.metadata["content_type"] = "text"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source_file"] = source_file if source_file else chunk.metadata.get("source", "inconnu")
        chunk.metadata["doc_id"] = doc_id if doc_id else "unknown"
        chunk.metadata["content_type"] = "text"

    return chunks

def build_vectorstore(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore, embeddings

def load_vectorstore(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

def format_context(docs, max_chars=7000):
    parts = []
    total = 0

    for d in docs:
        page = d.metadata.get("page", "?")
        source_file = d.metadata.get("source_file", "Document inconnu")
        content_type = d.metadata.get("content_type", "text")

        prefix = "SCHÉMA/IMAGE" if content_type == "image" else "TEXTE"
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

    def ask(self, question: str, selected_files=None):
        filter_dict = None
        if selected_files:
            filter_dict = {"source_file": {"$in": selected_files}}

        docs = self.vectorstore.similarity_search(
            question,
            k=max(self.k * 5, 20),
            filter=filter_dict,
            fetch_k=max(self.k * 8, 30),
        )

        docs = docs[:self.k]

        context = format_context(docs)
        prompt = RAG_TEMPLATE.format(context=context, question=question)
        answer = self.ollama.call_ollama(prompt, timeout=self.timeout)

        grouped_sources = defaultdict(list)
        for doc in docs:
            doc_name = doc.metadata.get("source_file", "Document inconnu")
            grouped_sources[doc_name].append(doc)

        return answer, docs, dict(grouped_sources)