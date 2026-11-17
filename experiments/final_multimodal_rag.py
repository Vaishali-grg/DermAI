# ============================================================
# final_multimodal_rag.py
#
# FINAL REAL-TIME MEDICAL RAG PIPELINE
#
# USER IMAGE
#       ↓
# EfficientNet-B2
#       ↓
# Skin-condition prediction
#       ↓
# User text + prediction
#       ↓
# BGE + ChromaDB
#       ↓
# Focused medical evidence
#       ↓
# PaliGemma 2 3B + LoRA adapter
#       ↓
# Concise evidence summary
#       ↓
# Python structured JSON
#
# FIXES vs previous version:
# - Actually loads the trained LoRA adapter (models/paligemma2_medical_lora)
#   on top of the base PaliGemma checkpoint. Without this, the base
#   "pt-224" model is being asked to do open-ended instruction following
#   it was never tuned for, which is the main source of hallucination.
# - If category-filtered retrieval returns nothing (e.g. category name
#   mismatch between classifier classes and PDF folder names), it retries
#   without the category filter instead of silently sending "No relevant
#   medical evidence retrieved." to the model.
# - Prints how many chunks were retrieved at each stage so a silent
#   empty-retrieval failure is visible in the logs.
# ============================================================

import os
import sys
import json
import re

# ============================================================
# TORCH SETTINGS
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


# ============================================================
# IMPORTS
# ============================================================

import torch

from PIL import Image

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration
)

from peft import PeftModel

from src.efficientnet import EfficientNetPredictor
from src.retriever import MedicalRetriever


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = (
    "google/paligemma2-3b-pt-224"
)

LORA_ADAPTER_DIR = (
    "models/paligemma2_medical_lora"
)

TOP_K = 3

MAX_NEW_TOKENS = 150


# ============================================================
# INPUT IMAGE
# ============================================================

if len(sys.argv) > 1:

    IMAGE_PATH = sys.argv[1]

else:

    IMAGE_PATH = (
        "test_images/testgirl.jpg"
    )


# ============================================================
# USER TEXT
# ============================================================

if len(sys.argv) > 2:

    USER_TEXT = sys.argv[2]

else:

    USER_TEXT = (
        "I have dark-looking areas on my face. "
        "What could be causing this and what general "
        "information should I know?"
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("====================================")
print("FINAL MEDICAL RAG PIPELINE")
print("====================================")

print("Device:", device)
print("Image:", IMAGE_PATH)
print("User text:", USER_TEXT)


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

print("\n====================================")
print("LOADING IMAGE")
print("====================================")

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image loaded successfully!")


# ============================================================
# 1. EFFICIENTNET
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


# ============================================================
# 2. MEDICAL RETRIEVER
# ============================================================

print("\n====================================")
print("MEDICAL RETRIEVAL")
print("====================================")

retriever = MedicalRetriever(
    db_directory="./vector_db"
)


# ============================================================
# 3. BUILD FOCUSED RETRIEVAL QUERY
# ============================================================

condition_name = predicted_condition.replace("_", " ")

retrieval_query = f"""
Condition:
{condition_name}

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

print("\nRetrieval query:")
print(retrieval_query)


# ============================================================
# 4. RETRIEVE (with category filter)
# ============================================================

print("\n====================================")
print("RAG RETRIEVAL")
print("====================================")

results = retriever.retrieve(
    query=retrieval_query,
    top_k=5,
    max_distance_threshold=0.35,
    category_filter=predicted_condition
)

print(f"\n[Retrieval] category-filtered hits: {len(results)}")

# ------------------------------------------------------------
# FALLBACK: if category filter matched nothing (e.g. the
# classifier class name and the PDF folder/category name in
# ChromaDB don't line up exactly), retry without the filter
# rather than silently continuing with zero evidence.
# ------------------------------------------------------------

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
# 5. SECONDARY FILTER
# ============================================================

useful_results = []

priority_terms = [
    "cause",
    "causes",
    "inflammation",
    "melanin",
    "hyperpigmentation",
    "ultraviolet",
    "sun exposure",
    "contributing",
    "clinical",
    "pathogenesis",
    "pigmentation"
]

for result in results:

    text_lower = result["text"].lower()

    score = sum(
        term in text_lower
        for term in priority_terms
    )

    if score > 0:
        useful_results.append((score, result))


# ============================================================
# RANK
# ============================================================

useful_results.sort(
    key=lambda item: (-item[0], item[1]["distance"])
)

selected_results = [
    item[1]
    for item in useful_results[:TOP_K]
]

# Fallback: if the priority-term filter removed everything,
# just take the top-K raw results instead of nothing.
if not selected_results:
    selected_results = results[:TOP_K]

print("\nSelected evidence:", len(selected_results))


# ============================================================
# 6. DISPLAY EVIDENCE
# ============================================================

print("\n====================================")
print("SELECTED MEDICAL EVIDENCE")
print("====================================")

for index, result in enumerate(selected_results):

    print(f"\nEvidence {index + 1}")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Distance:", result["distance"])
    print(result["text"])


# ============================================================
# 7. BUILD COMPACT CONTEXT
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
# 8. PALIGEMMA QUESTION
# ============================================================

prompt = f"""answer en

A skin-condition classifier predicts:
{condition_name}

The user's question is:
{USER_TEXT}

Relevant medical evidence:
{medical_context}

Using only the supplied evidence, briefly explain:
1. what this predicted skin condition may represent;
2. which causes or contributing factors are supported;
3. what general medical context is relevant.

Do not give a definitive diagnosis.
Do not introduce conditions not present in the evidence.
Keep the answer concise.
"""

print("\n====================================")
print("PALIGEMMA QUESTION")
print("====================================")
print(prompt)


# ============================================================
# 9. LOAD PALIGEMMA 2 3B + LORA ADAPTER
# ============================================================

print("\n====================================")
print("LOADING PALIGEMMA 2 3B")
print("====================================")

processor = PaliGemmaProcessor.from_pretrained(MODEL_NAME)


# ============================================================
# DTYPE
# ============================================================

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    model_dtype = torch.bfloat16
elif torch.cuda.is_available():
    model_dtype = torch.float16
else:
    model_dtype = torch.float32


# ============================================================
# BASE MODEL
# ============================================================

model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=model_dtype,
    device_map="auto"
)

print("Base PaliGemma 2 3B loaded successfully!")


# ============================================================
# LORA ADAPTER
#
# This is the key fix: previously the fine-tuned adapter
# trained in train_multimodal_lora.py was never attached here,
# so generation was coming entirely from the base pretrained
# ("pt-224") checkpoint, which isn't tuned for open-ended
# instruction following -> hallucination.
# ============================================================

if os.path.isdir(LORA_ADAPTER_DIR):

    print(f"\nLoading LoRA adapter from '{LORA_ADAPTER_DIR}'...")

    model = PeftModel.from_pretrained(
        model,
        LORA_ADAPTER_DIR
    )

    print("LoRA adapter loaded and merged into inference model!")

else:

    print(
        f"\nWARNING: LoRA adapter directory '{LORA_ADAPTER_DIR}' "
        "not found. Falling back to the BASE pretrained model. "
        "Expect weaker, less-grounded answers until you train "
        "and save an adapter there."
    )

model.eval()


# ============================================================
# 10. PROCESS IMAGE + QUESTION
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

            inputs[key] = inputs[key].to(
                device=model.device,
                dtype=model_dtype
            )

        else:

            inputs[key] = inputs[key].to(
                device=model.device
            )


# ============================================================
# 11. GENERATE
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


# ============================================================
# 12. REMOVE INPUT TOKENS
# ============================================================

generated_ids = generated_ids[0, input_length:]


# ============================================================
# 13. DECODE
# ============================================================

response = processor.decode(
    generated_ids,
    skip_special_tokens=True
).strip()


# ============================================================
# 14. BASIC CLEANING
# ============================================================

response = re.sub(
    r"^answer en\s*",
    "",
    response,
    flags=re.IGNORECASE
).strip()


# ============================================================
# 15. FINAL APPLICATION JSON
# ============================================================

structured_response = {

    "possible_condition": condition_name,

    "confidence": round(confidence, 4),

    "user_message": USER_TEXT,

    "model_response": response,

    "medical_sources": [
        {
            "document": result["source"],
            "page": result["page"]
        }
        for result in selected_results
    ],

    "disclaimer": "This is not a definitive diagnosis."
}


# ============================================================
# 16. FINAL RESPONSE
# ============================================================

print("\n====================================")
print("FINAL MEDICAL RESPONSE")
print("====================================")
print(response)


# ============================================================
# 17. STRUCTURED JSON
# ============================================================

print("\n====================================")
print("FINAL STRUCTURED JSON")
print("====================================")

print(
    json.dumps(
        structured_response,
        indent=2,
        ensure_ascii=False
    )
)


# ============================================================
# 18. SOURCES
# ============================================================

print("\n====================================")
print("MEDICAL SOURCES")
print("====================================")

for result in selected_results:
    print(f"- {result['source']}, page {result['page']}")


# ============================================================
# 19. SUMMARY
# ============================================================

print("\n====================================")
print("PIPELINE SUMMARY")
print("====================================")

print("User image:", IMAGE_PATH)
print("User text:", USER_TEXT)
print("EfficientNet prediction:", predicted_condition)
print("EfficientNet confidence:", f"{confidence:.6f}")
print("RAG evidence:", len(selected_results))
print("PaliGemma:", MODEL_NAME)
print(
    "LoRA adapter:",
    LORA_ADAPTER_DIR if os.path.isdir(LORA_ADAPTER_DIR) else "NOT FOUND (using base model)"
)

print("\n====================================")
print("FINAL PIPELINE COMPLETED")
print("====================================")