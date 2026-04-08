# ChatPDF RAG with Ollama and Streamlit (use local folder for now)

A local **Streamlit** application for chatting with one or more PDF files using a Retrieval-Augmented Generation (RAG) pipeline powered by **LangChain**, **FAISS**, **sentence-transformers**, and **Ollama**. The project supports classic text-based PDF RAG and can be extended with image/schema analysis through a vision model served by Ollama.[web:44][web:45][web:31]

## Features

- Chat with one or multiple PDF files in a single Streamlit interface.[web:45]
- Local embeddings with `sentence-transformers` and vector search with FAISS.[web:1]
- Local LLM inference through Ollama, including support for text models and vision models.[web:31][web:49][web:55]
- Optional image/schema analysis workflow for pages that contain images or vector drawings.[web:30][web:31]
- Source-aware retrieval with document name and page metadata stored alongside chunks in FAISS.[web:1]

## Project structure

```text
.
├── app.py
├── rag_pipeline.py
├── image_analyzer.py
├── ollama_client.py
├── requirements.txt
└── README.md
```

## Prerequisites

Before running the app, install the following on your machine:

- **Python 3.10+** is recommended for Streamlit and modern package compatibility.[web:45]
- **Ollama** must be installed and running locally so the Python app can call the Ollama API.[web:49][web:55]
- Enough disk space for local models, because Ollama model files can take several gigabytes depending on the model you use.[web:49][web:55]

## 1. Install Ollama

Download and install Ollama from the official documentation for your OS.[web:49][web:55]

### Windows

Use the official `OllamaSetup.exe` installer.[web:55]

### macOS

Install the official `.dmg`, move the application to `Applications`, and launch it once so the CLI is configured.[web:49]

### Linux

Use the official installation instructions from the Ollama website for your distribution.[web:49][web:55]

## 2. Start Ollama and pull models

Once Ollama is installed, start the Ollama service or desktop app, then download the models you want to use.

### Recommended text model

```bash
ollama pull llama3.2
```

### Recommended vision model

```bash
ollama pull llama3.2-vision
```

The app expects a text model for standard RAG answers, and a vision model if you enable image or schema analysis in the UI.[web:31]

To verify Ollama is working, run:

```bash
ollama run llama3.2
```

If Ollama responds in the terminal, the local server is available and the app should be able to connect to `http://localhost:11434`.[web:49][web:55]

## 3. Create a Python virtual environment

Using a virtual environment is the recommended setup for Streamlit projects.[web:45]

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 4. Install Python dependencies

Install the project dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Updated requirements.txt

Use the following dependency list:

```txt
streamlit>=1.32
requests>=2.31
pymupdf>=1.24

# LangChain ecosystem
langchain>=0.3
langchain-community>=0.3
langchain-huggingface>=0.1
langchain-text-splitters>=0.3

# Document loaders & PDF
pypdf>=4.0

# Embeddings & Vector store
sentence-transformers>=2.7
faiss-cpu>=1.8

# Ollama client
ollama>=0.4
```

This list adds `requests`, `pymupdf`, and `langchain-text-splitters`, which are required by the multimodal and chunking workflow shown in the updated codebase.[web:30][web:31]

## 6. Run the Streamlit app

From the project folder, launch the app with:

```bash
python -m streamlit run app.py
```

This is the standard Streamlit command-line pattern for running an app from a Python file.[web:45][web:48]

Streamlit should open a local browser tab automatically. If it does not, open the local URL shown in the terminal, usually something like:

```text
http://localhost:8501
```

## 7. How to use the app

1. Open the Streamlit sidebar.
2. Upload one or more PDF files.
3. Choose the Ollama text model, such as `llama3.2`.
4. Optionally enable **Analyze images / schemas** and choose a vision model such as `llama3.2-vision`.[web:31]
5. Click **Index documents**.
6. Ask questions in the chat box.

When image analysis is enabled, the app first detects only the PDF pages that appear to contain images or vector drawings, then sends only those pages to the vision model. This reduces unnecessary rendering and inference cost compared with analyzing every page.[web:30][web:41]

## 8. How Ollama fits into the pipeline

Ollama is the local inference layer for the application. The app uses it in two different ways:

- **Text generation**: the RAG prompt is sent to a text model such as `llama3.2` to generate the final answer.
- **Vision analysis**: selected PDF pages rendered as images are sent to a vision model such as `llama3.2-vision` to generate structured image descriptions.[web:31]

A typical multimodal flow looks like this:

1. Extract text from the PDF.
2. Split text into chunks.
3. Detect candidate pages containing images or drawings.
4. Render only those candidate pages.
5. Ask the vision model to describe diagrams or schemas.
6. Store both text chunks and image descriptions in FAISS.
7. Retrieve the most relevant chunks at question time.[web:30][web:31][web:1]

## 9. Troubleshooting

### Ollama is not found

If the app cannot reach Ollama:

- Make sure Ollama is installed correctly.[web:49][web:55]
- Make sure the Ollama service or desktop app is running.[web:49][web:55]
- Test manually with `ollama run llama3.2`.
- Confirm that the local API is reachable on `http://localhost:11434`.[web:49][web:55]

### A model is missing

Pull the model before running the app:

```bash
ollama pull llama3.2
ollama pull llama3.2-vision
```

### Streamlit does not start

Check that:

- your virtual environment is activated,[web:45]
- dependencies are installed from `requirements.txt`,
- you are running the command from the project directory,
- your Python version is compatible with Streamlit.[web:45]

### PDF image analysis is slow

Vision inference is usually the slowest part of the pipeline. To improve speed:

- use a smaller vision model if available,[web:31]
- reduce rendered image DPI in the image analysis module,[web:30]
- keep the “image analysis” option disabled unless the PDFs contain useful diagrams,
- analyze only candidate pages, which is already the strategy used in the optimized implementation.[web:30][web:41]

## 10. Recommended install commands

A typical clean setup looks like this:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2
ollama pull llama3.2-vision
python -m streamlit run app.py
```

On Windows, use the corresponding activation command for your shell instead of `source .venv/bin/activate`.[web:45]

## 11. Notes for development

- `PyPDFLoader` handles text extraction from PDFs through the LangChain document-loading stack.[web:1]
- `PyMuPDF` is used for efficient PDF page inspection, rendering, and image-related heuristics.[web:30]
- FAISS stores vector embeddings locally for fast semantic retrieval.[web:1]
- The `ollama` Python package can be used for a native Python integration, while direct HTTP calls remain useful for custom structured vision requests.[web:53][web:31]

## License

Add your preferred license here before publishing the project.
