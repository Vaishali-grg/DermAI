# ============================================================
# final_multimodal_rag_gemma3.py
#
# LLM-BASED MEDICAL RAG PIPELINE using Gemma 3 4B IT
#
# WHY THE SWITCH FROM PALIGEMMA:
# PaliGemma (both pt-224 and mix-224) is fundamentally a
# VQA/captioning specialist trained on short, structured tasks
# ("answer en", "caption en", "detect", etc.). It is not a
# genuine instruction-following chat model, which is why it
# kept degenerating into copying/fabricating fragments of the
# retrieved evidence instead of composing a real answer.
#
# Gemma 3 4B IT ("it" = instruction-tuned) is a real
# multimodal chat model, trained specifically to follow
# open-ended instructions and hold a conversation about an
# image. No LoRA, no training required -- it works zero-shot.
#
# First run downloads the model weights (~8GB) from Hugging
# Face -- a one-time cost, not training. Every run after that
# loads from local cache in seconds.
# ============================================================

import os
import sys
import json
import re

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from PIL import Image

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from src.efficientnet import EfficientNetPredictor
from src.retriever import MedicalRetriever


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "google/gemma-3-4b-it"

TOP_K = 3
MAX_NEW_TOKENS = 250

DISPLAY_NAMES = {
    "acne": "acne",
    "blackheads": "blackheads",
    "dark_spots": "dark spots / hyperpigmentation",
    "pores": "enlarged pores",
    "wrinkles": "facial wrinkles"
}


# ============================================================
# CITATION / REFERENCE FILTER
#
# Same fix as the PaliGemma script -- reference-list chunks
# were previously slipping through and poisoning generation.
# ============================================================

def looks_like_citation(text: str) -> bool:

    case_sensitive_patterns = [
        r"^\[?\d+\.\s+[A-Z][a-zA-Z-]+",
        r"©\s*[Tt]he [Aa]uthor",
    ]

    for pattern in case_sensitive_patterns:
        if re.search(pattern, text):
            return True

    lower = text.lower()

    case_insensitive_patterns = [
        r"\bet al\.",
        r"\bdoi\b",
        r"\bpmid\b",
        r"\bissn\b",
        r"article reuse guidelines",
        r"sagepub\.com",
    ]

    return any(re.search(pattern, lower) for pattern in case_insensitive_patterns)


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
print("LLM MEDICAL RAG PIPELINE (Gemma 3 4B IT)")
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
# 4. FILTER OUT CITATIONS / RANK
# ============================================================

priority_terms = [
    "cause", "causes", "inflammation", "melanin", "hyperpigmentation",
    "ultraviolet", "sun exposure", "contributing", "clinical",
    "pathogenesis", "pigmentation"
]

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
    selected_results = [
        r for r in results if not looks_like_citation(r["text"])
    ][:TOP_K]

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
# ============================================================

prompt_text = f"""A skin-condition classifier predicts: {display_name}

The user asked: {USER_TEXT}

Relevant medical evidence:
{medical_context}

Using only the evidence above, write a short, plain-language paragraph
(3-5 sentences) explaining what this predicted condition may involve and
what causes or contributing factors are supported by the evidence. Do
not give a definitive diagnosis and do not introduce anything not
present in the evidence. Write in full sentences, not a list.
"""

print("\n====================================")
print("GEMMA 3 PROMPT")
print("====================================")
print(prompt_text)


# ============================================================
# 7. LOAD GEMMA 3 4B IT
# ============================================================

print("\n====================================")
print("LOADING GEMMA 3 4B IT")
print("(first run downloads ~8GB, one-time only)")
print("====================================")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    model_dtype = torch.bfloat16
elif torch.cuda.is_available():
    model_dtype = torch.float16
else:
    model_dtype = torch.float32

if device.type == "cuda":
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        device_map="auto"
    ).eval()
else:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device).eval()

print("Gemma 3 4B IT loaded successfully!")


# ============================================================
# 8. BUILD CHAT MESSAGES + PROCESS
# ============================================================

print("\n====================================")
print("GEMMA 3 INPUT PROCESSING")
print("====================================")

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a careful medical-information assistant. "
                    "You explain possible skin conditions in plain "
                    "language using only the evidence you are given. "
                    "You never give a definitive diagnosis."
                )
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

if device.type == "cuda":
    inputs = {
        key: value.to(device=device, dtype=model_dtype)
        if torch.is_floating_point(value)
        else value.to(device)
        for key, value in inputs.items()
    }
else:
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

input_len = inputs["input_ids"].shape[-1]


# ============================================================
# 9. GENERATE
# ============================================================

print("\n====================================")
print("GEMMA 3 GENERATION")
print("====================================")

with torch.inference_mode():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    generated_ids = generated_ids[0][input_len:]

response = processor.decode(
    generated_ids,
    skip_special_tokens=True
).strip()


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
print("Model:", MODEL_NAME, "(zero-shot, no LoRA, no training)")

print("\n====================================")
print("FINAL PIPELINE COMPLETED")
print("====================================")