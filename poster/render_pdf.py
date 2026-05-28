"""Rend les pages 3, 5 et 6 du PDF AI Music Industry en PNG pour matcher le style."""
import fitz

src = r"C:\Users\knipe\Downloads\AI Music Industry.pdf"
doc = fitz.open(src)

for page_num in [3, 5, 6, 8]:
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=140)
    out = rf"C:\Code\py\ChatpdfRag\poster\src-slide-{page_num}.png"
    pix.save(out)
    print(f"OK {out}  size={pix.width}x{pix.height}")
doc.close()
