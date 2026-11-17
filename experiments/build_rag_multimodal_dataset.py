import json
import re

from src.retriever import MedicalRetriever


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "dataset/multimodal_instructions.jsonl"
)

OUTPUT_FILE = (
    "dataset/rag_multimodal_instructions.jsonl"
)

TOP_K = 3

MAX_CAUSE_SENTENCES = 3


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "acne": "acne",
    "blackheads": "blackheads",
    "dark_spots": "dark spots / hyperpigmentation",
    "pores": "enlarged pores",
    "wrinkles": "facial wrinkles"
}


# ============================================================
# CAUSAL SENTENCE EXTRACTION
#
# Instead of writing a fixed placeholder like
# "Use only factors supported by the retrieved evidence.",
# we pull real sentences out of the actual retrieved evidence
# that describe causes / contributing factors. This is what
# makes the training target *grounded* -- it varies per
# example based on what was actually retrieved, and the model
# learns to summarize real evidence instead of reciting a
# fixed boilerplate string.
# ============================================================

CAUSAL_KEYWORDS = [
    "cause", "causes", "caused by", "due to", "resulting from",
    "results from", "attributed to", "exacerbated by",
    "triggered by", "associated with", "linked to",
    "contributing factor", "risk factor", "leads to",
    "induced by", "arises from"
]


def split_sentences(text: str):

    text = re.sub(r"\s+", " ", text).strip()

    # Split on ". " while keeping it simple -- good enough for
    # extracting standalone evidence sentences.
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in raw_sentences if s.strip()]


def extract_causal_sentences(evidence_texts, max_sentences=MAX_CAUSE_SENTENCES):

    candidates = []

    for text in evidence_texts:

        for sentence in split_sentences(text):

            lower = sentence.lower()

            # Skip obvious citation / reference fragments
            if re.search(r"\bdoi\b|\bpmid\b|\bissn\b|et al\.", lower):
                continue

            if re.match(r"^\d+\.\s+[A-Z]", sentence):
                continue

            if len(sentence.split()) < 6:
                continue

            if any(keyword in lower for keyword in CAUSAL_KEYWORDS):
                candidates.append(sentence)

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []

    for sentence in candidates:
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(sentence)

    return unique_candidates[:max_sentences]


# ============================================================
# RETRIEVER
# ============================================================

retriever = MedicalRetriever(
    db_directory="./vector_db"
)


# ============================================================
# BUILD
# ============================================================

count = 0
skipped_no_evidence = 0

with open(INPUT_FILE, "r", encoding="utf-8") as source, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as target_file:

    for line in source:

        record = json.loads(line)

        label = record["label"]
        user_text = record["user_text"]

        display_name = DISPLAY_NAMES.get(
            label, label.replace("_", " ")
        )

        # ====================================================
        # RETRIEVAL QUERY
        # ====================================================

        query = f"""
Skin condition:
{label.replace("_", " ")}

User symptoms or question:
{user_text}

Need medical information about possible causes,
contributing factors and general management.
"""

        results = retriever.retrieve(
            query=query,
            top_k=TOP_K,
            max_distance_threshold=0.35,
            category_filter=label
        )

        # Fallback: retry without category filter if nothing matched
        # (e.g. category naming mismatch between classifier classes
        # and PDF folder names in the vector DB).
        if not results:
            results = retriever.retrieve(
                query=query,
                top_k=TOP_K,
                max_distance_threshold=0.35,
                category_filter=None
            )

        if not results:
            skipped_no_evidence += 1

        # ====================================================
        # BUILD EVIDENCE BLOCK (for the prompt)
        # ====================================================

        evidence_parts = []

        for index, result in enumerate(results):
            evidence_parts.append(
                f"[Evidence {index + 1}]\n"
                f"Source: {result['source']}\n"
                f"Page: {result['page']}\n"
                f"{result['text']}"
            )

        evidence = "\n\n".join(evidence_parts)

        # ====================================================
        # BUILD REAL, GROUNDED TARGET
        #
        # This replaces the old static placeholder TARGETS
        # dict. Causes are real sentences pulled from the
        # retrieved evidence. If no causal sentence can be
        # found, we say so honestly instead of inventing one
        # or repeating an instruction as if it were an answer.
        # ====================================================

        evidence_texts = [r["text"] for r in results]

        causal_sentences = extract_causal_sentences(evidence_texts)

        if causal_sentences:
            causes = causal_sentences
        else:
            causes = [
                "The retrieved evidence did not contain a clearly "
                "stated cause for this case; general contributing "
                "factors for this condition should be discussed "
                "with a dermatologist."
            ]

        if results:
            medical_context_summary = (
                f"Evidence drawn from: "
                f"{', '.join(sorted(set(r['source'] for r in results)))}."
            )
        else:
            medical_context_summary = (
                "No directly matching medical evidence was retrieved "
                "for this case."
            )

        target = {
            "possible_condition": display_name,

            "visible_observation":
                f"The image classifier predicts the {label.replace('_', ' ')} class.",

            "possible_causes_or_contributing_factors": causes,

            "general_guidance": [
                "This information is general and educational, not a "
                "personalized treatment plan.",
                "A dermatologist can confirm the cause and recommend "
                "an appropriate approach."
            ],

            "medical_context": medical_context_summary,

            "disclaimer": "This is not a definitive diagnosis."
        }

        # ====================================================
        # FINAL VLM PROMPT
        # ====================================================

        instruction = f"""
Analyze the facial image and the user's message.

USER MESSAGE:
{user_text}

SPECIALIZED SKIN CLASSIFIER:
{label.replace("_", " ")}

MEDICAL EVIDENCE:
{evidence if evidence else "No relevant medical evidence retrieved."}

Provide:

Possible condition:
Visible observation:
Possible causes or contributing factors:
General medical guidance:

Important:
- This is not a definitive diagnosis.
- Use the supplied medical evidence.
- Do not invent medical facts.
- Do not introduce unrelated conditions.
"""

        output_record = {
            "image": record["image"],
            "label": label,
            "user_text": user_text,
            "instruction": instruction,
            "medical_context": evidence,
            "target": json.dumps(target, ensure_ascii=False)
        }

        target_file.write(
            json.dumps(output_record, ensure_ascii=False) + "\n"
        )

        count += 1

        if count % 500 == 0:
            print("Processed:", count)


print("\nCreated:", OUTPUT_FILE)
print("Examples:", count)
print("Examples with NO retrieved evidence:", skipped_no_evidence)

if skipped_no_evidence > 0:
    print(
        "\nWARNING: some examples had zero retrieved evidence even "
        "after removing the category filter. Check that ./vector_db "
        "actually contains chunks for every class, and that "
        "max_distance_threshold (0.35) isn't too strict."
    )