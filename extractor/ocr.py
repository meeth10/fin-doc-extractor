"""
OCR fallback for pages with no usable digital text layer.

Rasterizes the page with PyMuPDF itself (no poppler/pdf2image dependency
needed) then runs Tesseract via pytesseract. Local, free, offline.
"""

import io
import pymupdf as fitz
from PIL import Image
import pytesseract

OCR_DPI_ZOOM = 300 / 72  # render at ~300dpi for decent OCR accuracy


def ocr_page(fitz_page) -> str:
    mat = fitz.Matrix(OCR_DPI_ZOOM, OCR_DPI_ZOOM)
    pix = fitz_page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)
