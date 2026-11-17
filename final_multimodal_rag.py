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
# CPU + GPU COMPATIBLE
#
# GPU available:
#     CUDA is automatically used.
#
# GPU unavailable:
#     CPU is automatically used.
#
# No model architecture or RAG logic has been changed.
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
#
# Automatically selects:
#
# CUDA → when an NVIDIA GPU is available
# CPU  → when no GPU is available
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DISPLAY DEVICE INFORMATION
# ============================================================

print("====================================")
print("FINAL MEDICAL RAG PIPELINE")
print("====================================")

print("Device:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print("GPU: Not available")
    print("Running entirely on CPU.")


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

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    "Image loaded successfully!"
)


# ============================================================
# 1. EFFICIENTNET
# ============================================================

print("\n====================================")
print("EFFICIENTNET")
print("====================================")

efficientnet = EfficientNetPredictor()

prediction = efficientnet.predict(
    image
)

predicted_condition = (
    prediction["predicted_class"]
)

confidence = (
    prediction["confidence"]
)

probabilities = (
    prediction["probabilities"]
)


print(
    "Prediction:",
    predicted_condition
)

print(
    "Confidence:",
    f"{confidence:.6f}"
)


print("\nClass probabilities:")

for class_name, probability in probabilities.items():

    print(
        f"  {class_name:12s}: "
        f"{probability:.6f}"
    )


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

condition_name = (
    predicted_condition.replace(
        "_",
        " "
    )
)


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
# 4. RETRIEVE WITH CATEGORY FILTER
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


print(
    f"\n[Retrieval] "
    f"category-filtered hits: "
    f"{len(results)}"
)


# ============================================================
# FALLBACK RETRIEVAL
# ============================================================

if not results:

    print(
        "[Retrieval] No hits with category "
        f"filter '{predicted_condition}'."
    )

    print(
        "[Retrieval] Retrying without "
        "category filter..."
    )


    results = retriever.retrieve(
        query=retrieval_query,
        top_k=5,
        max_distance_threshold=0.35,
        category_filter=None
    )


    print(
        f"[Retrieval] unfiltered hits: "
        f"{len(results)}"
    )


    if not results:

        print(
            "[Retrieval] WARNING: still zero hits."
        )

        print(
            "[Retrieval] Check that your vector "
            "DB contains chunks and that "
            "max_distance_threshold is not "
            "too strict."
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

    text_lower = (
        result["text"].lower()
    )


    score = sum(
        term in text_lower
        for term in priority_terms
    )


    if score > 0:

        useful_results.append(
            (
                score,
                result
            )
        )


# ============================================================
# 6. RANK RESULTS
# ============================================================

useful_results.sort(
    key=lambda item: (
        -item[0],
        item[1]["distance"]
    )
)


selected_results = [

    item[1]

    for item in useful_results[:TOP_K]

]


# ============================================================
# FALLBACK IF SECONDARY FILTER REMOVED EVERYTHING
# ============================================================

if not selected_results:

    selected_results = (
        results[:TOP_K]
    )


print(
    "\nSelected evidence:",
    len(selected_results)
)


# ============================================================
# 7. DISPLAY EVIDENCE
# ============================================================

print("\n====================================")
print("SELECTED MEDICAL EVIDENCE")
print("====================================")


for index, result in enumerate(
    selected_results
):

    print(
        f"\nEvidence {index + 1}"
    )

    print(
        "Source:",
        result["source"]
    )

    print(
        "Page:",
        result["page"]
    )

    print(
        "Distance:",
        result["distance"]
    )

    print(
        result["text"]
    )


# ============================================================
# 8. BUILD COMPACT MEDICAL CONTEXT
# ============================================================

context_parts = []


for index, result in enumerate(
    selected_results
):

    text = (
        result["text"].strip()
    )


    # Prevent excessively large
    # context from being sent to PaliGemma

    if len(text) > 1000:

        text = text[:1000]


    context_parts.append(

        f"[Evidence {index + 1}]\n"
        f"Source: {result['source']}\n"
        f"Page: {result['page']}\n"
        f"{text}"

    )


if context_parts:

    medical_context = (
        "\n\n".join(
            context_parts
        )
    )

else:

    medical_context = (
        "No relevant medical "
        "evidence retrieved."
    )


# ============================================================
# 9. PALIGEMMA PROMPT
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
# 10. LOAD PALIGEMMA PROCESSOR
# ============================================================

print("\n====================================")
print("LOADING PALIGEMMA 2 3B")
print("====================================")


processor = (
    PaliGemmaProcessor.from_pretrained(
        MODEL_NAME
    )
)


# ============================================================
# 11. SELECT MODEL DTYPE
#
# GPU:
#     bfloat16 when supported
#     otherwise float16
#
# CPU:
#     float32
# ============================================================

if (
    device.type == "cuda"
    and torch.cuda.is_bf16_supported()
):

    model_dtype = (
        torch.bfloat16
    )

elif device.type == "cuda":

    model_dtype = (
        torch.float16
    )

else:

    model_dtype = (
        torch.float32
    )


print(
    "Model dtype:",
    model_dtype
)


# ============================================================
# 12. LOAD BASE PALIGEMMA
#
# GPU:
#     device_map="auto"
#
# CPU:
#     explicitly move the model to CPU
# ============================================================

if device.type == "cuda":

    model = (
        PaliGemmaForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=model_dtype,
            device_map="auto"
        )
    )

else:

    model = (
        PaliGemmaForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        )
        .to(device)
    )


model.eval()


print(
    "Base PaliGemma 2 3B "
    "loaded successfully!"
)


# ============================================================
# 13. LOAD LORA ADAPTER
# ============================================================

if os.path.isdir(
    LORA_ADAPTER_DIR
):

    print(
        "\nLoading LoRA adapter from:"
    )

    print(
        LORA_ADAPTER_DIR
    )


    model = (
        PeftModel.from_pretrained(
            model,
            LORA_ADAPTER_DIR
        )
    )


    print(
        "LoRA adapter loaded successfully!"
    )


else:

    print(
        "\nWARNING:"
    )

    print(
        f"LoRA adapter directory "
        f"'{LORA_ADAPTER_DIR}' "
        "was not found."
    )

    print(
        "Using the base PaliGemma model."
    )


model.eval()


# ============================================================
# 14. PROCESS IMAGE + QUESTION
# ============================================================

print("\n====================================")
print("PALIGEMMA INPUT PROCESSING")
print("====================================")


inputs = processor(

    text=prompt,

    images=image,

    return_tensors="pt"

)


# ============================================================
# MOVE INPUTS TO SELECTED DEVICE
#
# Floating tensors:
#     use model dtype
#
# Integer tensors:
#     keep their integer dtype
# ============================================================

for key in inputs:

    if not torch.is_tensor(
        inputs[key]
    ):

        continue


    if inputs[key].is_floating_point():

        inputs[key] = inputs[key].to(

            device=device,

            dtype=model_dtype

        )

    else:

        inputs[key] = inputs[key].to(
            device=device
        )


# ============================================================
# 15. GENERATION
# ============================================================

print("\n====================================")
print("PALIGEMMA GENERATION")
print("====================================")


input_length = (
    inputs["input_ids"].shape[-1]
)


with torch.inference_mode():

    generated_ids = model.generate(

        **inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=False

    )


# ============================================================
# 16. REMOVE INPUT TOKENS
# ============================================================

generated_ids = (
    generated_ids[
        0,
        input_length:
    ]
)


# ============================================================
# 17. DECODE RESPONSE
# ============================================================

response = (
    processor.decode(
        generated_ids,
        skip_special_tokens=True
    )
    .strip()
)


# ============================================================
# 18. BASIC RESPONSE CLEANING
# ============================================================

response = re.sub(

    r"^answer en\s*",

    "",

    response,

    flags=re.IGNORECASE

).strip()


# ============================================================
# 19. FINAL APPLICATION JSON
# ============================================================

structured_response = {

    "possible_condition":
        condition_name,

    "confidence":
        round(
            confidence,
            4
        ),

    "user_message":
        USER_TEXT,

    "model_response":
        response,

    "medical_sources": [

        {
            "document":
                result["source"],

            "page":
                result["page"]

        }

        for result in selected_results

    ],

    "disclaimer":
        "This is not a definitive diagnosis."

}


# ============================================================
# 20. FINAL MEDICAL RESPONSE
# ============================================================

print("\n====================================")
print("FINAL MEDICAL RESPONSE")
print("====================================")

print(response)


# ============================================================
# 21. STRUCTURED JSON
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
# 22. MEDICAL SOURCES
# ============================================================

print("\n====================================")
print("MEDICAL SOURCES")
print("====================================")


for result in selected_results:

    print(
        f"- {result['source']}, "
        f"page {result['page']}"
    )


# ============================================================
# 23. PIPELINE SUMMARY
# ============================================================

print("\n====================================")
print("PIPELINE SUMMARY")
print("====================================")

print(
    "User image:",
    IMAGE_PATH
)

print(
    "User text:",
    USER_TEXT
)

print(
    "Device:",
    device
)

print(
    "EfficientNet prediction:",
    predicted_condition
)

print(
    "EfficientNet confidence:",
    f"{confidence:.6f}"
)

print(
    "RAG evidence:",
    len(selected_results)
)

print(
    "PaliGemma:",
    MODEL_NAME
)

print(
    "LoRA adapter:",
    (
        LORA_ADAPTER_DIR
        if os.path.isdir(
            LORA_ADAPTER_DIR
        )
        else
        "NOT FOUND "
        "(using base model)"
    )
)


print("\n====================================")
print("FINAL PIPELINE COMPLETED")
print("====================================")