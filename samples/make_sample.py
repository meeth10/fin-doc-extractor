"""
Builds a synthetic ~30-page "annual report" PDF to test the pipeline:
- cover page
- ~10 pages of MD&A-style prose (should NOT be flagged)
- Consolidated Balance Sheet page with a real table
- Consolidated Statement of Operations page with a real table
- Consolidated Statement of Cash Flows page with a real table
- ~5 pages of "Notes to Financial Statements" prose (flagged only if --include-notes)
- one page rendered as a PNG and re-inserted as an image-only page,
  simulating a scanned page, to exercise the OCR fallback path
"""

import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image, ImageDraw
import pymupdf as fitz

OUT_PATH = "samples/sample_annual_report.pdf"
SCAN_PAGE_PATH = "samples/_scanned_page.png"


def draw_prose_page(c, title, n_lines=35):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 740, title)
    c.setFont("Helvetica", 9)
    y = 715
    for i in range(n_lines):
        c.drawString(72, y, f"Lorem ipsum dolor sit amet, filler MD&A discussion line {i+1} "
                             f"about market conditions and outlook, not a financial statement.")
        y -= 18
        if y < 60:
            break


def draw_table_page(c, heading, rows):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 740, heading)
    c.setFont("Helvetica", 10)
    y = 700
    col_x = [72, 320, 440]
    for row in rows:
        for x, val in zip(col_x, row):
            c.drawString(x, y, str(val))
        y -= 20


def build():
    c = canvas.Canvas(OUT_PATH, pagesize=letter)

    # Cover page
    c.setFont("Helvetica-Bold", 22)
    c.drawString(72, 700, "Acme Global Holdings Inc.")
    c.setFont("Helvetica", 14)
    c.drawString(72, 670, "Annual Report — Fiscal Year 2025")
    c.showPage()

    # MD&A filler pages (10)
    for i in range(10):
        draw_prose_page(c, f"Management's Discussion and Analysis (page {i+1})")
        c.showPage()

    # Consolidated Balance Sheet
    balance_rows = [
        ["Line Item", "FY2025", "FY2024"],
        ["Cash and cash equivalents", "1,245", "980"],
        ["Accounts receivable", "742", "690"],
        ["Total current assets", "3,102", "2,850"],
        ["Total assets", "9,884", "9,010"],
        ["Total current liabilities", "1,900", "1,750"],
        ["Total liabilities", "4,220", "4,010"],
        ["Total stockholders' equity", "5,664", "5,000"],
    ]
    draw_table_page(c, "Consolidated Balance Sheets (in millions)", balance_rows)
    c.showPage()

    # Consolidated Statement of Operations
    income_rows = [
        ["Line Item", "FY2025", "FY2024"],
        ["Net sales", "12,400", "11,200"],
        ["Cost of goods sold", "7,300", "6,800"],
        ["Gross profit", "5,100", "4,400"],
        ["Operating expenses", "2,200", "2,050"],
        ["Operating income", "2,900", "2,350"],
        ["Net income", "2,105", "1,690"],
    ]
    draw_table_page(c, "Consolidated Statements of Operations (in millions)", income_rows)
    c.showPage()

    # Consolidated Statement of Cash Flows
    cf_rows = [
        ["Line Item", "FY2025", "FY2024"],
        ["Net cash from operating activities", "2,950", "2,410"],
        ["Net cash used in investing activities", "(1,100)", "(980)"],
        ["Net cash used in financing activities", "(700)", "(650)"],
        ["Net increase in cash", "1,150", "780"],
    ]
    draw_table_page(c, "Consolidated Statements of Cash Flows (in millions)", cf_rows)
    c.showPage()

    # Notes pages (5)
    for i in range(5):
        draw_prose_page(c, f"Notes to Consolidated Financial Statements (page {i+1})")
        c.showPage()

    c.save()

    # --- Now build a "scanned" page (image-only, no text layer) and merge it in,
    # to exercise the OCR path. We render a Statement of Stockholders' Equity
    # as a flat PNG image, then insert it as a page with zero text layer.
    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "Statement of Stockholders' Equity (in millions)", fill="black")
    draw.text((100, 160), "For the year ended FY2025 / FY2024", fill="black")
    lines = [
        "Equity share capital                500        500",
        "Securities premium                 1,200      1,000",
        "Retained earnings                   3,964      3,500",
        "Balance, beginning of year          5,000",
        "Net income                          2,105",
        "Dividends paid                       (441)",
        "Total comprehensive income          2,105      1,690",
        "Balance, end of year                5,664",
    ]
    y = 240
    for line in lines:
        draw.text((100, y), line, fill="black")
        y += 60
    img.save(SCAN_PAGE_PATH)

    # Merge: rebuild final doc = original pages + one image-only page
    base = fitz.open(OUT_PATH)
    final = fitz.open()
    final.insert_pdf(base)
    img_page = final.new_page(width=612, height=792)
    rect = fitz.Rect(36, 36, 576, 756)
    img_page.insert_image(rect, filename=SCAN_PAGE_PATH)
    final.save(OUT_PATH, incremental=False)
    final.close()
    base.close()

    print(f"Built {OUT_PATH}")


if __name__ == "__main__":
    build()
