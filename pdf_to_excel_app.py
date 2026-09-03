"""
PDF to Excel Converter (works with scanned / image-based PDFs too)
--------------------------------------------------------------------
Run with:  streamlit run pdf_to_excel_app.py

Requirements (see requirements.txt):
    pip install -r requirements.txt

System requirement:
    Tesseract OCR engine must be installed on the machine running this app
    (this is separate from the pytesseract python package).
        Windows : https://github.com/UB-Mannheim/tesseract/wiki
        macOS   : brew install tesseract
        Linux   : sudo apt-get install tesseract-ocr
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# If Tesseract is not on PATH, uncomment and set the path below:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="PDF to Excel Converter", layout="wide")
st.title("📄 PDF to Excel Converter")
st.caption("Upload a PDF (scanned or digital) → Submit → Download the extracted data as an Excel file.")


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
    """Try to pull a structured table directly from a digital (non-scanned) PDF page."""
    tables = page.find_tables()
    if tables and tables.tables:
        dfs = []
        for t in tables.tables:
            data = t.extract()
            if data and len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                dfs.append(df)
        if dfs:
            return dfs
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

    # Sort by vertical position, then group into rows using a tolerance band
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

    # Within each row, sort left-to-right and merge words that are close
    # together (same column) using a dynamic gap threshold.
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


def rows_to_dataframe(rows):
    """Pad rows to equal length and build a DataFrame, using the first row as header."""
    if not rows:
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]
    header, *body = padded
    header = [h if h else f"Column {i+1}" for i, h in enumerate(header)]
    df = pd.DataFrame(body, columns=header)
    return df


def convert_pdf_to_excel(pdf_bytes, progress_callback=None):
    """
    Main pipeline:
      - Try to extract native tables from each page first (fast, accurate for digital PDFs)
      - Fall back to OCR-based table reconstruction for scanned pages
      - Returns a dict of {sheet_name: DataFrame}
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sheets = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        sheet_name = f"Page {page_index + 1}"

        if progress_callback:
            progress_callback(page_index + 1, len(doc))

        native_tables = try_extract_text_layer(page)
        if native_tables:
            for t_idx, df in enumerate(native_tables):
                name = sheet_name if len(native_tables) == 1 else f"{sheet_name}_{t_idx + 1}"
                sheets[name[:31]] = df
            continue

        # Fall back to OCR for scanned / image-only pages
        image = pdf_page_to_image(page, zoom=3.0)
        rows = ocr_page_to_rows(image)
        df = rows_to_dataframe(rows)
        if not df.empty:
            sheets[sheet_name[:31]] = df

    doc.close()
    return sheets


def build_excel_bytes(sheets: dict):
    """Write all extracted tables to a single Excel workbook (one sheet per page/table)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if not sheets:
            pd.DataFrame({"Message": ["No table data could be extracted."]}).to_excel(
                writer, index=False, sheet_name="Result"
            )
        for name, df in sheets.items():
            safe_name = name[:31] if name else "Sheet1"
            df.to_excel(writer, index=False, sheet_name=safe_name)
    output.seek(0)
    return output


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
if "excel_bytes" not in st.session_state:
    st.session_state.excel_bytes = None
if "preview_sheets" not in st.session_state:
    st.session_state.preview_sheets = None

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
            sheets = convert_pdf_to_excel(pdf_bytes, progress_callback=update_progress)
            excel_io = build_excel_bytes(sheets)

        progress_bar.empty()
        st.session_state.excel_bytes = excel_io.getvalue()
        st.session_state.preview_sheets = sheets
        st.success(f"Conversion complete — {len(sheets)} sheet(s) extracted.")

# Preview extracted tables
if st.session_state.preview_sheets:
    st.subheader("Preview")
    for name, df in st.session_state.preview_sheets.items():
        with st.expander(f"Sheet: {name} ({len(df)} rows)"):
            st.dataframe(df, use_container_width=True)

# 3. Download button
if st.session_state.excel_bytes:
    st.download_button(
        label="Download Excel File",
        data=st.session_state.excel_bytes,
        file_name="converted_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
