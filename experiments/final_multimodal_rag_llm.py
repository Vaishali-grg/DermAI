# ============================================================
# final_multimodal_rag_llm.py
#
# LLM-BASED MEDICAL RAG PIPELINE (zero-shot, no training)
#
# Uses google/paligemma2-3b-mix-224, which is Google's
# instruction-tuned PaliGemma checkpoint (as opposed to the
# raw "pt-224" pretrained-only checkpoint). Because it's
# already instruction-tuned, it can follow an evidence-grounded
# prompt reasonably well out of the box -- no LoRA fine-tuning
# and no multi-hour training run required.
#
# If you later want tighter, more consistent phrasing, you can
# still fine-tune a LoRA adapter on top of this checkpoint --
# but this script works without one.
# ============================================================

import os
import sys
import json
import re

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from PIL import Image

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration
)

from src.efficientnet import EfficientNetPredictor
from src.retriever import MedicalRetriever


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "google/paligemma2-3b-mix-224"

TOP_K = 3
MAX_NEW_TOKENS = 150

DISPLAY_NAMES = {
    "acne": "acne",
    "blackheads": "blackheads",
    "dark_spots": "dark spots / hyperpigmentation",
    "pores": "enlarged pores",
    "wrinkles": "facial wrinkles"
}


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("====================================")
print("LLM MEDICAL RAG PIPELINE (mix-224, zero-shot)")
print("====================================")
print("Device:", device)
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
# 3. MEDICAL RETRIEVER
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
# 4. SECONDARY FILTER / RANK
# ============================================================

priority_terms = [
    "cause", "causes", "inflammation", "melanin", "hyperpigmentation",
    "ultraviolet", "sun exposure", "contributing", "clinical",
    "pathogenesis", "pigmentation"
]

# Reject chunks that are reference-list / citation / publisher
# boilerplate even if they happen to contain a priority term
# (e.g. a citation title mentioning "hyperpigmentation").
CITATION_PATTERNS = [
    r"^\[?\d+\.\s+[A-Z][a-zA-Z-]+",   # "49. Silpa-Archa N, ..."
    r"\bet al\.",
    r"\bdoi\b",
    r"\bpmid\b",
    r"\bissn\b",
    r"article reuse guidelines",
    r"sagepub\.com",
    r"©\s*the author",
]


def looks_like_citation(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in CITATION_PATTERNS)


useful_results = []

for result in results:

    if looks_like_citation(result["text"]):
        continue

    text_lower = result["text"].lower()
    score = sum(term in text_lower for term in priority_terms)

    if score > 0:
        useful_results.append((score, result))

useful_results.sort(key=lambda item: (-item[0], item[1]["distance"]))

selected_results = [item[1] for item in useful_results[:TOP_K]]

if not selected_results:
    # Fall back to raw results, but still exclude citation-like chunks
    selected_results = [
        r for r in results if not looks_like_citation(r["text"])
    ][:TOP_K]

if not selected_results:
    # Truly nothing usable left -- fall back to raw results as last resort
    selected_results = results[:TOP_K]

print("\nSelected evidence:", len(selected_results))

for index, result in enumerate(selected_results):
    print(f"\nEvidence {index + 1}")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Distance:", result["distance"])
    print(result["text"])


# ============================================================
# 5. BUILD COMPACT CONTEXT
# ============================================================

context_parts = []

for index, result in enumerate(selected_results):

    text = result["text"].strip()

    if len(text) > 1000:
        text = text[:1000]

    context_parts.append(
        f"[Evidence {index + 1}]\n"
        f"Source: {result['source']}\n"
        f"Page: {result['page']}\n"
        f"{text}"
    )

if context_parts:
    medical_context = "\n\n".join(context_parts)
else:
    medical_context = "No relevant medical evidence retrieved."


# ============================================================
# 6. PROMPT
#
# mix-224 is instruction-tuned, so it can follow a longer,
# structured instruction like this reasonably well zero-shot.
# ============================================================

# NOTE: "answer en" is PaliGemma's SHORT-VQA task prefix -- it
# was trained to give brief factoid answers with that prefix,
# which is why long-form generation with it tends to degrade
# into copying fragments of the context instead of composing a
# real paragraph. Dropping it and writing this as a direct
# instruction gets closer to free-form generation behavior.

prompt = f"""Skin-condition classifier prediction: {display_name}

User question: {USER_TEXT}

Medical evidence:
{medical_context}

Write a short, plain-language paragraph (3-5 sentences) that uses only the
evidence above to explain what this predicted condition may involve and
what factors could contribute to it. Do not give a definitive diagnosis
and do not mention conditions absent from the evidence. Write in full
sentences, not a list.
"""

print("\n====================================")
print("PALIGEMMA QUESTION")
print("====================================")
print(prompt)


# ============================================================
# 7. LOAD PALIGEMMA 2 MIX-224 (zero-shot, no LoRA)
# ============================================================

print("\n====================================")
print("LOADING PALIGEMMA 2 3B (mix-224, instruction-tuned)")
print("====================================")

processor = PaliGemmaProcessor.from_pretrained(MODEL_NAME)

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    model_dtype = torch.bfloat16
elif torch.cuda.is_available():
    model_dtype = torch.float16
else:
    model_dtype = torch.float32

model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=model_dtype,
    device_map="auto"
)

model.eval()

print("PaliGemma 2 3B (mix-224) loaded successfully!")


# ============================================================
# 8. PROCESS IMAGE + QUESTION
# ============================================================

print("\n====================================")
print("PALIGEMMA INPUT PROCESSING")
print("====================================")

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
)

for key in inputs:

    if torch.is_tensor(inputs[key]):

        if inputs[key].is_floating_point():
            inputs[key] = inputs[key].to(device=model.device, dtype=model_dtype)
        else:
            inputs[key] = inputs[key].to(device=model.device)


# ============================================================
# 9. GENERATE
# ============================================================

print("\n====================================")
print("PALIGEMMA GENERATION")
print("====================================")

input_length = inputs["input_ids"].shape[-1]

with torch.inference_mode():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

generated_ids = generated_ids[0, input_length:]

response = processor.decode(generated_ids, skip_special_tokens=True).strip()

response = re.sub(r"^answer en\s*", "", response, flags=re.IGNORECASE).strip()


# ============================================================
# 10. FINAL STRUCTURED JSON
# ============================================================

structured_response = {

    "possible_condition": display_name,
    "confidence": round(confidence, 4),
    "user_message": USER_TEXT,
    "model_response": response,

    "medical_sources": [
        {"document": result["source"], "page": result["page"]}
        for result in selected_results
    ],

    "disclaimer": "This is not a definitive diagnosis."
}


print("\n====================================")
print("FINAL MEDICAL RESPONSE")
print("====================================")
print(response)

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
print("PaliGemma:", MODEL_NAME, "(zero-shot, no LoRA, no training)")

print("\n====================================")
print("FINAL PIPELINE COMPLETED")
print("====================================")
