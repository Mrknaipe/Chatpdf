"""Extrait tout le texte du PDF AI Music Industry."""
from pypdf import PdfReader

src = r"C:\Users\knipe\Downloads\AI Music Industry.pdf"
reader = PdfReader(src)

for i, page in enumerate(reader.pages):
    print(f"\n====== PAGE {i + 1} ======")
    txt = page.extract_text() or ""
    print(txt)
