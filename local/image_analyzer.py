import os
import json
import base64
import tempfile
import requests
import pymupdf

from langchain_core.documents import Document

OLLAMA_URL = "http://localhost:11434/api/chat"

VISION_PROMPT = """
Tu analyses une page de PDF contenant potentiellement une image, un schéma, un graphe,
un organigramme, un diagramme d'architecture, une figure technique ou une illustration.

Réponds en JSON strict avec les clés suivantes :
- is_relevant_image: bool
- image_type: string
- visible_text: string
- key_elements: array of strings
- relationships: array of strings
- summary: string

Règles :
- Si la page ne contient pas d'image informative ou pas de schéma utile, mets is_relevant_image à false.
- visible_text doit contenir le texte lisible dans l'image/schéma.
- summary doit être factuel, court et exploitable pour une recherche RAG.
- N'invente rien.
"""

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def detect_candidate_pages(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    candidates = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        image_count = len(page.get_images(full=True))
        drawing_count = len(page.get_drawings())
        text_blocks = page.get_text("dict").get("blocks", [])

        inline_image_blocks = sum(1 for b in text_blocks if b.get("type") == 1)
        text_length = len(page.get_text().strip())

        has_visual = (
            image_count > 0
            or drawing_count > 20
            or inline_image_blocks > 0
        )

        if has_visual:
            candidates.append({
                "page_index": page_idx,
                "image_count": image_count,
                "drawing_count": drawing_count,
                "inline_image_blocks": inline_image_blocks,
                "text_length": text_length,
            })

    return candidates

def render_page_to_image(pdf_path: str, page_index: int, output_dir: str, dpi: int = 140) -> str:
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    page = doc[page_index]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img_path = os.path.join(output_dir, f"page_{page_index + 1}.png")
    pix.save(img_path)
    return img_path

def call_vision_model(image_path: str, vision_model: str, timeout: int = 180):
    image_b64 = encode_image_to_base64(image_path)

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": VISION_PROMPT,
                "images": [image_b64]
            }
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "is_relevant_image": {"type": "boolean"},
                "image_type": {"type": "string"},
                "visible_text": {"type": "string"},
                "key_elements": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "relationships": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "summary": {"type": "string"}
            },
            "required": [
                "is_relevant_image",
                "image_type",
                "visible_text",
                "key_elements",
                "relationships",
                "summary"
            ]
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    content = data.get("message", {}).get("content", "{}").strip()
    return json.loads(content)

def image_analysis_to_document(result: dict, source_file: str, doc_id: str, page_index: int, image_path: str):
    visible_text = result.get("visible_text", "").strip()
    image_type = result.get("image_type", "").strip()
    summary = result.get("summary", "").strip()
    key_elements = result.get("key_elements", [])
    relationships = result.get("relationships", [])

    text = f"""
Type d'image: {image_type}
Texte visible: {visible_text}
Éléments clés: {", ".join(key_elements)}
Relations: {" | ".join(relationships)}
Résumé: {summary}
""".strip()

    return Document(
        page_content=text,
        metadata={
            "source_file": source_file,
            "doc_id": doc_id,
            "page": page_index,
            "content_type": "image",
            "image_type": image_type,
            "image_path": image_path,
        }
    )

def analyze_pdf_images(pdf_path: str, source_file: str, doc_id: str, vision_model: str, timeout: int = 180):
    candidates = detect_candidate_pages(pdf_path)
    image_docs = []

    output_dir = os.path.join(tempfile.gettempdir(), "chatpdf_image_cache", doc_id)
    os.makedirs(output_dir, exist_ok=True)

    analyzed_pages = 0

    for item in candidates:
        page_index = item["page_index"]

        try:
            image_path = render_page_to_image(
                pdf_path=pdf_path,
                page_index=page_index,
                output_dir=output_dir,
                dpi=140
            )

            result = call_vision_model(
                image_path=image_path,
                vision_model=vision_model,
                timeout=timeout
            )

            analyzed_pages += 1

            if result.get("is_relevant_image", False):
                image_doc = image_analysis_to_document(
                    result=result,
                    source_file=source_file,
                    doc_id=doc_id,
                    page_index=page_index,
                    image_path=image_path
                )
                image_docs.append(image_doc)

        except Exception:
            continue

    stats = {
        "candidate_pages": len(candidates),
        "analyzed_pages": analyzed_pages,
        "image_chunks": len(image_docs),
    }

    return image_docs, stats