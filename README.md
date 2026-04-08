# ChatPDF RAG with Ollama and Streamlit (use local folder for now)

A local **Streamlit** application for chatting with one or more PDF files using a Retrieval-Augmented Generation (RAG) pipeline powered by **LangChain**, **FAISS**, **sentence-transformers**, and **Ollama**. The project supports classic text-based PDF RAG and can be extended with image/schema analysis through a vision model served by Ollama.

## Features

- Chat with one or multiple PDF files in a single Streamlit interface.
- Local embeddings with `sentence-transformers` and vector search with FAISS.
- Local LLM inference through Ollama, including support for text models and vision models.
- Optional image/schema analysis workflow for pages that contain images or vector drawings.
- Source-aware retrieval with document name and page metadata stored alongside chunks in FAISS.

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

- **Python 3.10+** is recommended for Streamlit and modern package compatibility.
- **Ollama** must be installed and running locally so the Python app can call the Ollama API.
- Enough disk space for local models, because Ollama model files can take several gigabytes depending on the model you use.

## 1. Install Ollama

Download and install Ollama from the official documentation for your OS.

### Windows

Use the official `OllamaSetup.exe` installer.

### macOS

Install the official `.dmg`, move the application to `Applications`, and launch it once so the CLI is configured.

### Linux

Use the official installation instructions from the Ollama website for your distribution.

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

The app expects a text model for standard RAG answers, and a vision model if you enable image or schema analysis in the UI.

To verify Ollama is working, run:

```bash
ollama run llama3.2
```

If Ollama responds in the terminal, the local server is available and the app should be able to connect to `http://localhost:11434`.

## 3. Create a Python virtual environment

Using a virtual environment is the recommended setup for Streamlit projects.

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

This list adds `requests`, `pymupdf`, and `langchain-text-splitters`, which are required by the multimodal and chunking workflow shown in the updated codebase.

## 6. Run the Streamlit app

From the project folder, launch the app with:

```bash
python -m streamlit run app.py
```

This is the standard Streamlit command-line pattern for running an app from a Python file.

Streamlit should open a local browser tab automatically. If it does not, open the local URL shown in the terminal, usually something like:

```text
http://localhost:8501
```

## 7. How to use the app

1. Open the Streamlit sidebar.
2. Upload one or more PDF files.
3. Choose the Ollama text model, such as `llama3.2`.
4. Optionally enable **Analyze images / schemas** and choose a vision model such as `llama3.2-vision:11b`.
5. Click **Index documents**.
6. Ask questions in the chat box.

When image analysis is enabled, the app first detects only the PDF pages that appear to contain images or vector drawings, then sends only those pages to the vision model. This reduces unnecessary rendering and inference cost compared with analyzing every page.

## 8. How Ollama fits into the pipeline

Ollama is the local inference layer for the application. The app uses it in two different ways:

- **Text generation**: the RAG prompt is sent to a text model such as `llama3.2` to generate the final answer.
- **Vision analysis**: selected PDF pages rendered as images are sent to a vision model such as `llama3.2-vision` to generate structured image descriptions.

A typical multimodal flow looks like this:

1. Extract text from the PDF.
2. Split text into chunks.
3. Detect candidate pages containing images or drawings.
4. Render only those candidate pages.
5. Ask the vision model to describe diagrams or schemas.
6. Store both text chunks and image descriptions in FAISS.
7. Retrieve the most relevant chunks at question time.

## 9. Troubleshooting

### Ollama is not found

If the app cannot reach Ollama:

- Make sure Ollama is installed correctly.
- Make sure the Ollama service or desktop app is running.
- Test manually with `ollama run llama3.2`.
- Confirm that the local API is reachable on `http://localhost:11434`.

### A model is missing

Pull the model before running the app:

```bash
ollama pull llama3.2
ollama pull llama3.2-vision
```

### Streamlit does not start

Check that:

- your virtual environment is activated,
- dependencies are installed from `requirements.txt`,
- you are running the command from the project directory,
- your Python version is compatible with Streamlit.

### PDF image analysis is slow

Vision inference is usually the slowest part of the pipeline. To improve speed:

- use a smaller vision model if available,
- reduce rendered image DPI in the image analysis module,
- keep the “image analysis” option disabled unless the PDFs contain useful diagrams,
- analyze only candidate pages, which is already the strategy used in the optimized implementation.

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

On Windows, use the corresponding activation command for your shell instead of `source .venv/bin/activate`.

## 11. Notes for development

- `PyPDFLoader` handles text extraction from PDFs through the LangChain document-loading stack.
- `PyMuPDF` is used for efficient PDF page inspection, rendering, and image-related heuristics.
- FAISS stores vector embeddings locally for fast semantic retrieval.
- The `ollama` Python package can be used for a native Python integration, while direct HTTP calls remain useful for custom structured vision requests.

## License

Add your preferred license here before publishing the project.
