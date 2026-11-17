import re

from typing import (
    List,
    Dict,
    Any
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# ============================================================
# CONDITION + DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_metadata(
    filename: str
) -> Dict[str, str]:
    """
    Automatically detects the skin condition and document type
    from the PDF filename.

    Examples:

        wrinkles_diet_nutrients.pdf
            category = wrinkles
            type = diet

        wrinkles_home_remedies.pdf
            category = wrinkles
            type = home_remedy
    """

    filename_lower = filename.lower()

    # --------------------------------------------------------
    # CONDITION
    # --------------------------------------------------------

    if "blackhead" in filename_lower:

        category = "blackheads"

    elif (
        "dark_spot" in filename_lower
        or "dark spot" in filename_lower
        or "darkspot" in filename_lower
        or "hyperpigmentation" in filename_lower
    ):

        category = "dark_spots"

    elif "wrinkle" in filename_lower:

        category = "wrinkles"

    elif "pore" in filename_lower:

        category = "pores"

    elif "acne" in filename_lower:

        category = "acne"

    else:

        category = "general"

    # --------------------------------------------------------
    # DOCUMENT TYPE
    # --------------------------------------------------------

    if (
        "diet" in filename_lower
        or "nutrient" in filename_lower
        or "food" in filename_lower
    ):

        document_type = "diet"

    elif (
        "home" in filename_lower
        or "remed" in filename_lower
        or "ingredient" in filename_lower
        or "topical" in filename_lower
    ):

        document_type = "home_remedy"

    else:

        document_type = "medical"

    return {
        "category": category,
        "type": document_type
    }


# ============================================================
# HEADER DETECTION
# ============================================================

def is_header(
    line: str
) -> bool:

    line_str = line.strip()

    if not line_str:
        return False

    if len(line_str) > 100:
        return False

    # --------------------------------------------------------
    # Numbered sections
    # --------------------------------------------------------

    if re.match(
        r"^(Section|Subsection)\s+\d+",
        line_str,
        re.IGNORECASE
    ):
        return True

    if re.match(
        r"^\d+\.\d+(\.\d+)?\s+\S+",
        line_str
    ):
        return True

    # --------------------------------------------------------
    # Single-level numbered headings
    # --------------------------------------------------------

    if re.match(
        r"^\d+\.\s+[A-Z][A-Za-z]",
        line_str
    ):
        return True

    # --------------------------------------------------------
    # Markdown headings
    # --------------------------------------------------------

    if line_str.startswith("#"):
        return True

    # --------------------------------------------------------
    # ALL-CAPS heading
    # --------------------------------------------------------

    if (
        line_str.isupper()
        and len(line_str) >= 4
        and len(line_str.split()) <= 12
        and not line_str.endswith(".")
        and "DOI" not in line_str
        and "HTTP" not in line_str
    ):
        return True

    return False


# ============================================================
# REFERENCE HEADING
# ============================================================

def is_reference_heading(
    line: str
) -> bool:

    normalized = (
        line.strip().lower()
    )

    reference_headings = {
        "references",
        "reference",
        "bibliography",
        "literature cited",
        "cited references",
        "references and notes",
        "references and bibliography"
    }

    return normalized in reference_headings


# ============================================================
# HEADER CLEANING
# ============================================================

def clean_header_text(
    header: str
) -> str:

    header = re.sub(
        r"^[#\s]+",
        "",
        header
    )

    return header.strip()


# ============================================================
# TEXT QUALITY CLEANING
# ============================================================

def clean_chunk_text(
    text: str
) -> str:

    text = text.strip()

    # --------------------------------------------------------
    # Remove repeated whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove leading punctuation artifacts
    # --------------------------------------------------------

    text = re.sub(
        r"^[\s\.,;:]+",
        "",
        text
    )

    # --------------------------------------------------------
    # Remove DOI / URL fragments
    # --------------------------------------------------------

    text = re.sub(
        r"\bdoi:\s*\S+",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # --------------------------------------------------------
    # Normalize spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOW QUALITY CHUNK DETECTION
# ============================================================

def is_low_quality_chunk(
    text: str
) -> bool:

    normalized = text.strip().lower()

    if not normalized:
        return True

    words = normalized.split()

    # --------------------------------------------------------
    # Very short chunks
    # --------------------------------------------------------

    if len(words) < 12:
        return True

    # --------------------------------------------------------
    # Reference indicators
    # --------------------------------------------------------

    reference_patterns = [

        r"\breferences?\b",

        r"\bbibliography\b",

        r"\bet al\b",

        r"\bdoi\b",

        r"\bissn\b",

        r"\bvol\.\s*\d+",

        r"\bpmid\b",

        r"\bclinicaltrials\b",

        r"\bhttps?://"

    ]

    reference_hits = 0

    for pattern in reference_patterns:

        if re.search(
            pattern,
            normalized
        ):

            reference_hits += 1

    if (
        reference_hits >= 3
        and len(words) < 100
    ):

        return True

    # --------------------------------------------------------
    # Citation-heavy chunks
    # --------------------------------------------------------

    citation_numbers = len(
        re.findall(
            r"\b\d+\.\s+[A-Z][A-Za-z-]+",
            text
        )
    )

    if (
        citation_numbers >= 4
        and len(words) < 120
    ):

        return True

    return False


# ============================================================
# PAGE SECTION PARSER
# ============================================================

def parse_page_sections(
    raw_text: str
) -> List[Dict[str, Any]]:

    lines = raw_text.split("\n")

    sections = []

    current_main_header = "General"

    current_sub_header = ""

    current_text_lines = []

    inside_references = False

    def flush_current():

        nonlocal current_text_lines

        if not current_text_lines:
            return

        section_path = (

            f"{current_main_header} > "
            f"{current_sub_header}"

            if current_sub_header

            else current_main_header
        )

        text = " ".join(
            current_text_lines
        )

        text = clean_chunk_text(
            text
        )

        if text:

            sections.append({

                "section_path":
                    section_path,

                "text":
                    text

            })

        current_text_lines = []

    for line in lines:

        cleaned_line = line.strip()

        if not cleaned_line:
            continue

        # ====================================================
        # REFERENCES
        # ====================================================

        if is_reference_heading(
            cleaned_line
        ):

            flush_current()

            inside_references = True

            continue

        if inside_references:
            continue

        # ====================================================
        # HEADER
        # ====================================================

        if is_header(
            cleaned_line
        ):

            flush_current()

            clean_h = clean_header_text(
                cleaned_line
            )

            if (
                cleaned_line
                .lower()
                .startswith("subsection")
            ):

                current_sub_header = clean_h

            elif re.match(
                r"^\d+\.\d+",
                cleaned_line
            ):

                current_sub_header = clean_h

            else:

                current_main_header = clean_h

                current_sub_header = ""

            continue

        # ====================================================
        # NORMAL CONTENT
        # ====================================================

        current_text_lines.append(
            cleaned_line
        )

    flush_current()

    return sections


# ============================================================
# RECURSIVE CHUNKING
# ============================================================

def recursive_chunk_page(
    page_record: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 75
) -> List[Dict[str, Any]]:

    page_sections = parse_page_sections(
        page_record["raw_text"]
    )

    # --------------------------------------------------------
    # Detect metadata ONCE for this PDF
    # --------------------------------------------------------

    document_metadata = detect_document_metadata(
        page_record["source"]
    )

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap,

        length_function=len,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            ""
        ]
    )

    chunks = []

    chunk_index = 0

    for section in page_sections:

        section_path = section[
            "section_path"
        ]

        section_text = section[
            "text"
        ]

        raw_chunks = (
            text_splitter
            .split_text(
                section_text
            )
        )

        for chunk_str in raw_chunks:

            chunk_str = clean_chunk_text(
                chunk_str
            )

            # ------------------------------------------------
            # Remove garbage
            # ------------------------------------------------

            if is_low_quality_chunk(
                chunk_str
            ):

                continue

            # ------------------------------------------------
            # Add section context
            # ------------------------------------------------

            header_prefixed_text = (

                f"[{section_path}]\n"
                f"{chunk_str}"

            )

            chunk_id = (

                f"{page_record['source']}"
                f"_p{page_record['page_number']}"
                f"_c{chunk_index}"

            )

            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            metadata = {

                "document_id":
                    page_record["source"],

                "source":
                    page_record["source"],

                "page":
                    page_record["page_number"],

                "section_path":
                    section_path,

                "chunk_index":
                    chunk_index,

                "char_length":
                    len(header_prefixed_text),

                # NEW
                "category":
                    document_metadata["category"],

                # NEW
                "type":
                    document_metadata["type"]

            }

            chunks.append({

                "chunk_id":
                    chunk_id,

                "text":
                    header_prefixed_text,

                "metadata":
                    metadata

            })

            chunk_index += 1

    return chunks