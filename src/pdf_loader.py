# import pymupdf as fitz 
# import os 
# from typing import List, Dict, Any 
 
# def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]: 
#     """ 
#     Reads a PDF file and extracts text page by page alongside metadata. 
#     """ 
#     if not os.path.exists(pdf_path): 
#         raise FileNotFoundError(f"PDF document not found at: {pdf_path}") 
         
#     doc = fitz.open(pdf_path) 
#     file_name = os.path.basename(pdf_path) 
#     extracted_data = [] 
 
#     for page_num in range(len(doc)): 
#         page = doc[page_num] 
#         raw_text = page.get_text("text") 
         
#         extracted_data.append({ 
#             "source": file_name, 
#             "page_number": page_num + 1, 
#             "raw_text": raw_text, 
#             "char_count": len(raw_text) 
#         }) 
 
#     doc.close() 
#     return extracted_data

import os
import re
from typing import List, Dict, Any

import pymupdf as fitz


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_pdf_text(raw_text: str) -> str:
    """
    Cleans common PDF extraction artifacts while preserving
    meaningful medical text.
    """

    if not raw_text:
        return ""

    text = raw_text

    # --------------------------------------------------------
    # Normalize Unicode / whitespace
    # --------------------------------------------------------

    text = text.replace("\xa0", " ")
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2012", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")

    # --------------------------------------------------------
    # Rejoin words broken at line endings.
    #
    # Example:
    #
    # inflam-
    # mation
    #
    # becomes:
    #
    # inflammation
    # --------------------------------------------------------

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # --------------------------------------------------------
    # Normalize multiple spaces
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Normalize excessive blank lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # Remove standalone page numbers
        # ----------------------------------------------------

        if re.fullmatch(
            r"(page\s*)?\d+",
            line,
            re.IGNORECASE
        ):
            continue

        # ----------------------------------------------------
        # Remove DOI-only lines
        # ----------------------------------------------------

        if re.match(
            r"^doi\s*:",
            line,
            re.IGNORECASE
        ):
            continue

        # ----------------------------------------------------
        # Remove URL-only lines
        # ----------------------------------------------------

        if re.fullmatch(
            r"https?://\S+",
            line,
            re.IGNORECASE
        ):
            continue

        # ----------------------------------------------------
        # Remove email-only lines
        # ----------------------------------------------------

        if re.fullmatch(
            r"[\w.+-]+@[\w.-]+\.\w+",
            line
        ):
            continue

        # ----------------------------------------------------
        # Remove obvious journal URL fragments
        # ----------------------------------------------------

        if (
            "jcms.sagepub.com" in line.lower()
            and len(line) < 120
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(
        cleaned_lines
    )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(
    pdf_path: str
) -> List[Dict[str, Any]]:
    """
    Extracts PDF text page-by-page and applies basic
    extraction cleanup.
    """

    if not os.path.exists(pdf_path):

        raise FileNotFoundError(
            f"PDF document not found at: {pdf_path}"
        )

    doc = fitz.open(pdf_path)

    file_name = os.path.basename(
        pdf_path
    )

    extracted_data = []


    for page_num in range(
        len(doc)
    ):

        page = doc[page_num]

        raw_text = page.get_text(
            "text"
        )

        cleaned_text = clean_pdf_text(
            raw_text
        )

        extracted_data.append({

            "source": file_name,

            "page_number": page_num + 1,

            "raw_text": cleaned_text,

            "char_count": len(
                cleaned_text
            )
        })


    doc.close()

    return extracted_data