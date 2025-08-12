# file_loader.py
import io
import pdfplumber
from docx import Document

def extracting_text(file_bytes: bytes, filename: str):
    """
    Extracts text from PDF or DOCX file bytes.
    Works fully in-memory (no saving to disk).
    """
    doc_text = []

    if filename.lower().endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    doc_text.append(text)
        return doc_text

    elif filename.lower().endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():
                doc_text.append(para.text)
        return doc_text

    else:
        raise ValueError("Unsupported file format. Please upload .pdf or .docx files.")
