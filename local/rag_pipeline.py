from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ollama_client import OllamaClient


RAG_TEMPLATE = """Tu es un assistant expert en analyse de documents.
Réponds UNIQUEMENT à partir des extraits du document fournis ci-dessous.
Si la réponse ne se trouve pas dans les extraits, réponds exactement :
"Je ne trouve pas cette information dans le document."

Extraits du document :
{context}

Question : {question}

Réponse :
"""


def load_and_split(pdf_path: str, chunk_size=800, chunk_overlap=150):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
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
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )


def format_context(docs, max_chars=6000):
    parts = []
    total = 0
    for d in docs:
        page = d.metadata.get("page", "?")
        chunk = f"[Page {page}] {d.page_content.strip()}"
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
        docs = self.vectorstore.similarity_search(question, k=max(self.k * 5, 20))

        if selected_files and "Select all" not in selected_files:
            docs = [
                d for d in docs
                if d.metadata.get("source_file") in selected_files
            ]

        docs = docs[:self.k]

        context = format_context(docs)
        prompt = RAG_TEMPLATE.format(context=context, question=question)

        answer = self.ollama.call_ollama(prompt, timeout=self.timeout)
        return answer, docs
