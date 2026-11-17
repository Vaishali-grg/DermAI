# ============================================================
# final_multimodal_rag_deterministic.py
#
# DETERMINISTIC MEDICAL RAG PIPELINE (no LLM generation step)
#
# USER IMAGE -> EfficientNet-B2 -> predicted class
#            -> BGE + ChromaDB -> medical evidence
#            -> Python extraction -> grounded answer (no LLM)
#
# No model call for the text answer, so no hallucination risk
# and nothing to train. Fastest and most reliable option.
# ============================================================

import os
import sys
import json
import re

from PIL import Image

from src.efficientnet import EfficientNetPredictor
from src.retriever import MedicalRetriever


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 3
MAX_CAUSE_SENTENCES = 3

DISPLAY_NAMES = {
    "acne": "acne",
    "blackheads": "blackheads",
    "dark_spots": "dark spots / hyperpigmentation",
    "pores": "enlarged pores",
    "wrinkles": "facial wrinkles"
}

CAUSAL_KEYWORDS = [
    "cause", "causes", "caused by", "due to", "resulting from",
    "results from", "attributed to", "exacerbated by",
    "triggered by", "associated with", "linked to",
    "contributing factor", "risk factor", "leads to",
    "induced by", "arises from"
]


# ============================================================
# HELPERS
# ============================================================

def split_sentences(text: str):

    text = re.sub(r"\s+", " ", text).strip()
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in raw_sentences if s.strip()]


def extract_causal_sentences(evidence_texts, max_sentences=MAX_CAUSE_SENTENCES):

    candidates = []

    for text in evidence_texts:

        for sentence in split_sentences(text):

            lower = sentence.lower()

            if re.search(r"\bdoi\b|\bpmid\b|\bissn\b|et al\.", lower):
                continue

            if re.match(r"^\d+\.\s+[A-Z]", sentence):
                continue

            if len(sentence.split()) < 6:
                continue

            if any(keyword in lower for keyword in CAUSAL_KEYWORDS):
                candidates.append(sentence)

    seen = set()
    unique_candidates = []

    for sentence in candidates:
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(sentence)

    return unique_candidates[:max_sentences]


def build_answer_text(display_name, causes, sources):

    lines = []

    lines.append(
        f"Based on the image, the classifier's top prediction is "
        f"consistent with {display_name}."
    )

    if causes:
        lines.append("Retrieved medical evidence points to the following possible contributing factors:")
        for sentence in causes:
            lines.append(f"- {sentence}")
    else:
        lines.append(
            "The retrieved medical evidence did not contain a clearly "
            "stated cause for this specific case."
        )

    if sources:
        unique_sources = sorted(set(sources))
        lines.append(
            "This information is drawn from: " + ", ".join(unique_sources) + "."
        )

    lines.append(
        "This is general information only, not a definitive diagnosis. "
        "A dermatologist can confirm the cause and recommend appropriate next steps."
    )

    return "\n".join(lines)


# ============================================================
# INPUT
# ============================================================

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    IMAGE_PATH = "test_images/testgirl.jpg"

if len(sys.argv) > 2:
    USER_TEXT = sys.argv[2]
else:
    USER_TEXT = (
        "I have dark-looking areas on my face. "
        "What could be causing this and what general "
        "information should I know?"
    )


print("====================================")
print("DETERMINISTIC MEDICAL RAG PIPELINE")
print("====================================")
print("Image:", IMAGE_PATH)
print("User text:", USER_TEXT)


if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")


# ============================================================
# 1. LOAD IMAGE
# ============================================================

print("\n====================================")
print("LOADING IMAGE")
print("====================================")

image = Image.open(IMAGE_PATH).convert("RGB")
print("Image loaded successfully!")


# ============================================================
# 2. EFFICIENTNET
# ============================================================

print("\n====================================")
print("EFFICIENTNET")
print("====================================")

efficientnet = EfficientNetPredictor()
prediction = efficientnet.predict(image)

predicted_condition = prediction["predicted_class"]
confidence = prediction["confidence"]
probabilities = prediction["probabilities"]

print("Prediction:", predicted_condition)
print("Confidence:", f"{confidence:.6f}")

print("\nClass probabilities:")
for class_name, probability in probabilities.items():
    print(f"  {class_name:12s}: {probability:.6f}")

display_name = DISPLAY_NAMES.get(
    predicted_condition, predicted_condition.replace("_", " ")
)


# ============================================================
# 3. RETRIEVE MEDICAL EVIDENCE
# ============================================================

print("\n====================================")
print("MEDICAL RETRIEVAL")
print("====================================")

retriever = MedicalRetriever(db_directory="./vector_db")

retrieval_query = f"""
Condition:
{display_name}

User question:
{USER_TEXT}

Retrieve medical information specifically relevant to:
- definition or description
- possible causes
- contributing factors
- relevant clinical context

Do not prioritize treatment procedures, references,
or unrelated medical conditions.
"""

results = retriever.retrieve(
    query=retrieval_query,
    top_k=5,
    max_distance_threshold=0.35,
    category_filter=predicted_condition
)

print(f"\n[Retrieval] category-filtered hits: {len(results)}")

if not results:

    print(
        "[Retrieval] No hits with category filter "
        f"'{predicted_condition}'. Retrying without category filter..."
    )

    results = retriever.retrieve(
        query=retrieval_query,
        top_k=5,
        max_distance_threshold=0.35,
        category_filter=None
    )

    print(f"[Retrieval] unfiltered hits: {len(results)}")

    if not results:
        print(
            "[Retrieval] WARNING: still zero hits. Check that "
            "your vector DB actually contains chunks, and that "
            "max_distance_threshold isn't too strict."
        )


# ============================================================
# 4. RANK / SELECT TOP EVIDENCE
# ============================================================

priority_terms = [
    "cause", "causes", "inflammation", "melanin", "hyperpigmentation",
    "ultraviolet", "sun exposure", "contributing", "clinical",
    "pathogenesis", "pigmentation"
]

scored_results = []

for result in results:

    text_lower = result["text"].lower()
    score = sum(term in text_lower for term in priority_terms)

    if score > 0:
        scored_results.append((score, result))

scored_results.sort(key=lambda item: (-item[0], item[1]["distance"]))

selected_results = [item[1] for item in scored_results[:TOP_K]]

if not selected_results:
    selected_results = results[:TOP_K]

print("\nSelected evidence:", len(selected_results))

for index, result in enumerate(selected_results):
    print(f"\nEvidence {index + 1}")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Distance:", result["distance"])
    print(result["text"])


# ============================================================
# 5. BUILD DETERMINISTIC ANSWER (no LLM)
# ============================================================

print("\n====================================")
print("BUILDING GROUNDED ANSWER")
print("====================================")

evidence_texts = [r["text"] for r in selected_results]
causal_sentences = extract_causal_sentences(evidence_texts)
source_names = [r["source"] for r in selected_results]

answer_text = build_answer_text(display_name, causal_sentences, source_names)


# ============================================================
# 6. FINAL STRUCTURED JSON
# ============================================================

structured_response = {

    "possible_condition": display_name,
    "confidence": round(confidence, 4),
    "user_message": USER_TEXT,

    "possible_causes_or_contributing_factors":
        causal_sentences if causal_sentences else
        ["No clearly stated cause was found in the retrieved evidence."],

    "model_response": answer_text,

    "medical_sources": [
        {"document": result["source"], "page": result["page"]}
        for result in selected_results
    ],

    "disclaimer": "This is not a definitive diagnosis."
}


print("\n====================================")
print("FINAL MEDICAL RESPONSE")
print("====================================")
print(answer_text)

print("\n====================================")
print("FINAL STRUCTURED JSON")
print("====================================")
print(json.dumps(structured_response, indent=2, ensure_ascii=False))

print("\n====================================")
print("PIPELINE SUMMARY")
print("====================================")
print("User image:", IMAGE_PATH)
print("User text:", USER_TEXT)
print("EfficientNet prediction:", predicted_condition)
print("EfficientNet confidence:", f"{confidence:.6f}")
print("RAG evidence:", len(selected_results))
print("Generation method: deterministic (no LLM, no training)")

print("\n====================================")
print("FINAL PIPELINE COMPLETED")
print("====================================")
