"""
PDF to Excel Converter (works with scanned / image-based PDFs too)
--------------------------------------------------------------------
Run locally with:
    python -m streamlit run pdf_to_excel_app.py

Requirements (see requirements.txt):
    python -m pip install -r requirements.txt

System requirement (OCR engine, separate from the Python package):
    Windows : https://github.com/UB-Mannheim/tesseract/wiki
    macOS   : brew install tesseract
    Linux   : sudo apt-get install tesseract-ocr

Deploying on Streamlit Community Cloud:
    Upload the included packages.txt alongside this file (and requirements.txt)
    in the same GitHub repo. Streamlit Cloud reads packages.txt to install
    system-level (apt) packages such as tesseract-ocr, which pip cannot install.
    After adding it, click "Reboot app" (or push a new commit) so the app
    rebuilds with it.
"""

import io
import shutil
import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# ----------------------------------------------------------------------
# Locate the Tesseract binary automatically where possible.
# On Windows, if it's installed but not on PATH, uncomment and set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ----------------------------------------------------------------------
_TESSERACT_PATH = shutil.which("tesseract")
if _TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH

st.set_page_config(page_title="PDF to Excel Converter", layout="wide")
st.title("📄 PDF to Excel Converter")
st.caption("Upload a PDF (scanned or digital) → Submit → Download the extracted data as a single-sheet Excel file.")


# ----------------------------------------------------------------------
# Core conversion logic
# ----------------------------------------------------------------------
def pdf_page_to_image(page, zoom=3.0):
    """Render a PDF page to a PIL image at high resolution for better OCR accuracy."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def try_extract_text_layer(page):
    """Try to pull structured table rows directly from a digital (non-scanned) PDF page."""
    tables = page.find_tables()
    if tables and tables.tables:
        page_rows = []
        for t in tables.tables:
            data = t.extract()
            if data:
                page_rows.extend([[("" if c is None else str(c)) for c in row] for row in data])
        if page_rows:
            return page_rows
    return None


def ocr_page_to_rows(image, row_tolerance=12, col_gap_factor=2.2):
    """
    OCR a page image and reconstruct a table structure by clustering
    recognized words into rows (by y-position) and columns (by x-gaps).
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip("-").isdigit() else -1
        if conf < 0:
            continue
        words.append({
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
        })

    if not words:
        return []

    words.sort(key=lambda w: (w["top"], w["left"]))
    rows = []
    current_row = [words[0]]
    current_top = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - current_top) <= row_tolerance:
            current_row.append(w)
        else:
            rows.append(current_row)
            current_row = [w]
            current_top = w["top"]
    rows.append(current_row)

    structured_rows = []
    for row in rows:
        row.sort(key=lambda w: w["left"])
        avg_char_width = np.median([w["width"] / max(len(w["text"]), 1) for w in row])
        gap_threshold = max(avg_char_width * col_gap_factor, 20)

        cells = []
        current_cell_words = [row[0]]
        for prev, cur in zip(row, row[1:]):
            gap = cur["left"] - (prev["left"] + prev["width"])
            if gap > gap_threshold:
                cells.append(" ".join(w["text"] for w in current_cell_words))
                current_cell_words = [cur]
            else:
                current_cell_words.append(cur)
        cells.append(" ".join(w["text"] for w in current_cell_words))
        structured_rows.append(cells)

    return structured_rows


def normalize_cell(text):
    return "".join(ch for ch in text.strip().lower() if ch.isalnum())


def row_matches_header(row, header, threshold=0.5):
    """Detect a repeated header row (common on every page of a scanned multi-page table)."""
    if not header:
        return False
    norm_row = [normalize_cell(c) for c in row]
    norm_header = [normalize_cell(c) for c in header]
    compare_len = min(len(norm_row), len(norm_header))
    if compare_len == 0:
        return False
    matches = sum(
        1 for a, b in zip(norm_row[:compare_len], norm_header[:compare_len])
        if a and b and (a == b or a in b or b in a)
    )
    return (matches / compare_len) >= threshold


def make_unique_headers(header):
    """Ensure no two column names are identical (required by pandas/pyarrow for display)."""
    seen = {}
    unique = []
    for i, h in enumerate(header):
        name = h.strip() if h and h.strip() else f"Column {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        unique.append(name)
    return unique


def convert_pdf_to_single_table(pdf_bytes, progress_callback=None):
    """
    Main pipeline — merges every page into ONE continuous table (single sheet):
      - Tries native table extraction first (accurate for digital PDFs)
      - Falls back to OCR-based reconstruction for scanned/image-only pages
      - Uses the first row found as the master header
      - Automatically skips rows on later pages that look like a repeated header
    Returns a single pandas DataFrame.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    master_header = None
    all_data_rows = []
    max_cols = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        if progress_callback:
            progress_callback(page_index + 1, len(doc))

        page_rows = try_extract_text_layer(page)
        if not page_rows:
            image = pdf_page_to_image(page, zoom=3.0)
            page_rows = ocr_page_to_rows(image)

        if not page_rows:
            continue

        for row in page_rows:
            max_cols = max(max_cols, len(row))
            if master_header is None:
                master_header = row
                continue
            if row_matches_header(row, master_header):
                continue  # skip a repeated header row from a later page
            all_data_rows.append(row)

    doc.close()

    if master_header is None:
        return pd.DataFrame()

    def pad(r):
        return r + [""] * (max_cols - len(r))

    header = make_unique_headers(pad(master_header))
    body = [pad(r) for r in all_data_rows]
    df = pd.DataFrame(body, columns=header)
    return df


def build_excel_bytes(df: pd.DataFrame):
    """Write the single merged table to one Excel sheet."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if df.empty:
            pd.DataFrame({"Message": ["No table data could be extracted."]}).to_excel(
                writer, index=False, sheet_name="Result"
            )
        else:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
    output.seek(0)
    return output


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
if "excel_bytes" not in st.session_state:
    st.session_state.excel_bytes = None
if "preview_df" not in st.session_state:
    st.session_state.preview_df = None

if _TESSERACT_PATH is None:
    st.warning(
        "Tesseract OCR engine was not found on this system. Native-table extraction will "
        "still work for digital PDFs, but scanned/image PDFs will fail until Tesseract is "
        "installed (see the instructions at the top of this file / packages.txt for Streamlit Cloud)."
    )

# 1. Upload button
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

# 2. Submit button
if st.button("Submit", type="primary", disabled=uploaded_file is None):
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        progress_bar = st.progress(0, text="Starting conversion...")

        def update_progress(current, total):
            progress_bar.progress(current / total, text=f"Processing page {current} of {total}...")

        with st.spinner("Converting PDF to Excel — scanned pages are processed with OCR, this may take a moment..."):
            try:
                df = convert_pdf_to_single_table(pdf_bytes, progress_callback=update_progress)
                excel_io = build_excel_bytes(df)
                st.session_state.excel_bytes = excel_io.getvalue()
                st.session_state.preview_df = df
                progress_bar.empty()
                st.success(f"Conversion complete — {len(df)} rows extracted into a single sheet.")
            except pytesseract.pytesseract.TesseractNotFoundError:
                progress_bar.empty()
                st.error(
                    "Tesseract OCR engine is not installed on this server, so scanned pages "
                    "can't be read. If you're on Streamlit Community Cloud, add a packages.txt "
                    "file containing 'tesseract-ocr' to your repo and reboot the app. If running "
                    "locally, install Tesseract (see instructions at the top of the script)."
                )

# Preview
if st.session_state.preview_df is not None and not st.session_state.preview_df.empty:
    st.subheader("Preview")
    try:
        st.dataframe(st.session_state.preview_df, use_container_width=True)
    except Exception:
        st.dataframe(st.session_state.preview_df.astype(str), use_container_width=True)

# 3. Download button
if st.session_state.excel_bytes:
    st.download_button(
        label="Download Excel File",
        data=st.session_state.excel_bytes,
        file_name="converted_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
