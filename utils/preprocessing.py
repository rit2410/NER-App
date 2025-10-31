import re
from typing import Optional

try:
    import pdfplumber
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False
    print("⚠️ pdfplumber not installed. PDF reading disabled.")

def read_text_from_upload(uploaded_file) -> Optional[str]:
    print(f"📂 Reading uploaded file: {uploaded_file.name}")
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        print("📄 Detected TXT file.")
        data = uploaded_file.read()
        try:
            txt = data.decode("utf-8")
        except UnicodeDecodeError:
            txt = data.decode("latin-1", errors="ignore")
        print(f"✅ TXT file read ({len(txt)} chars).")
        return txt

    if name.endswith(".pdf"):
        if not _HAS_PDF:
            raise RuntimeError("❌ pdfplumber not installed. Please add 'pdfplumber' to requirements.txt.")
        print("📄 Detected PDF. Extracting text...")
        text_parts = []
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                print(f"📘 Page {i+1}: {len(page_text)} chars.")
                text_parts.append(page_text)
        joined = "\n".join(text_parts)
        print(f"✅ PDF read ({len(joined)} chars).")
        return joined

    print("⚠️ Unsupported file type.")
    return None

def clean_text(text: str) -> str:
    print("🧹 Cleaning text...")
    text = text.replace("\r", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    cleaned = text.strip()
    print(f"✅ Cleaned text length: {len(cleaned)}")
    return cleaned
