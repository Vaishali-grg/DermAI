import os
import re
import time
import uuid
from typing import List

import torch
from PIL import Image

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException
)
from fastapi.middleware.cors import (
    CORSMiddleware
)
from fastapi.staticfiles import (
    StaticFiles
)
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    FileResponse
)
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration
)

from src.efficientnet import (
    EfficientNetPredictor
)
from src.retriever import (
    MedicalRetriever
)


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Multimodal Skin Analysis RAG API (Gemma 3)"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

app.mount(
    "/uploads",
    StaticFiles(
        directory=UPLOAD_DIR
    ),
    name="uploads"
)

app.mount(
    "/frontend",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "acne": "Acne",
    "blackheads": "Blackheads",
    "dark_spots": "Dark Spots / Hyperpigmentation",
    "pores": "Enlarged Pores",
    "wrinkles": "Facial Wrinkles"
}


# ============================================================
# EFFICIENTNET
# ============================================================

print("\n[System] Initializing EfficientNet-B2 Model...")

predictor = EfficientNetPredictor()

print("[System] EfficientNet-B2 ready!")


# ============================================================
# MEDICAL RETRIEVER
# ============================================================

print("\n[System] Initializing Medical Retriever...")

retriever = MedicalRetriever(
    db_directory="./vector_db",
    collection_name="dermatology_kb"
)

print("[System] Medical Retriever ready!")


# ============================================================
# GEMMA 3
# ============================================================

print("\n[System] Initializing Gemma 3 4B IT...")
print("[System] This may take a moment...\n")

MODEL_NAME = "google/gemma-3-4b-it"


# ============================================================
# MODEL DTYPE
# ============================================================

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        print("[Gemma] Using bfloat16")
    else:
        model_dtype = torch.float16
        print("[Gemma] Using float16")
else:
    model_dtype = torch.float32
    print("[Gemma] CUDA unavailable - using float32")


# ============================================================
# PROCESSOR & MODEL LOADING
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("[Gemma] Loading model...")

gemma_model = (
    Gemma3ForConditionalGeneration
    .from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
        device_map="auto"
    )
    .eval()
)

print("[System] Gemma 3 loaded successfully!")


# ============================================================
# CITATION FILTER
# ============================================================

def looks_like_citation(text: str) -> bool:
    case_sensitive_patterns = [
        r"^\[?\d+\.\s+[A-Z][a-zA-Z-]+",
        r"©\s*[Tt]he [Aa]uthor"
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
        r"sagepub\.com"
    ]

    return any(
        re.search(pattern, lower)
        for pattern in case_insensitive_patterns
    )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results,
    label: str
) -> str:
    if not results:
        return f"No relevant {label.lower()} evidence retrieved."

    parts = []

    for index, result in enumerate(results):
        text = result["text"].strip()

        if len(text) > 1000:
            text = text[:1000]

        parts.append(
            f"[{label} {index + 1}]\n"
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"{text}"
        )

    return "\n\n".join(parts)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    return RedirectResponse(url="/frontend/")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": str(predictor.device),
        "gemma": MODEL_NAME
    }


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/api/analyze")
async def analyze_skin(
    image: UploadFile = File(...),
    user_text: str = Form(...)
):
    start_time = time.time()

    try:
        # ----------------------------------------------------
        # 1. SAVE IMAGE
        # ----------------------------------------------------
        file_extension = (
            os.path.splitext(image.filename)[1]
            or ".jpg"
        )

        unique_filename = (
            f"{uuid.uuid4().hex}"
            f"{file_extension}"
        )

        file_path = os.path.join(
            UPLOAD_DIR,
            unique_filename
        )

        contents = await image.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        print("\n========================================")
        print("[1] Image uploaded and saved")
        print(f"[1] File: {unique_filename}")
        print("========================================")

        # ----------------------------------------------------
        # 2. LOAD IMAGE
        # ----------------------------------------------------
        pil_image = Image.open(file_path).convert("RGB")

        # ----------------------------------------------------
        # 3. EFFICIENTNET
        # ----------------------------------------------------
        print("[2] Running EfficientNet-B2...")

        efficientnet_start = time.time()

        prediction = predictor.predict(pil_image)

        efficientnet_time = (
            time.time() - efficientnet_start
        )

        predicted_condition = prediction["predicted_class"]

        confidence = float(prediction["confidence"])

        display_name = DISPLAY_NAMES.get(
            predicted_condition,
            predicted_condition
        )

        print(f"[2] EfficientNet completed in {efficientnet_time:.2f}s")
        print(f"[2] Prediction: {display_name}")
        print(f"[2] Confidence: {confidence:.4f}")

        # ----------------------------------------------------
        # 4. RETRIEVAL QUERIES
        # ----------------------------------------------------
        medical_query = f"""
Condition:
{display_name}

User question:
{user_text}

Retrieve medical evidence about:
- what the condition involves
- possible causes
- contributing factors
- relevant clinical information
"""

        diet_query = f"""
Condition:
{display_name}

User question:
{user_text}

Retrieve ONLY dietary information relevant to this condition, including:
- foods to include
- relevant nutrients
- dietary patterns
- foods or dietary patterns to moderate
"""

        home_query = f"""
Condition:
{display_name}

User question:
{user_text}

Retrieve ONLY information about gentle home/topical ingredients relevant to this condition, including:
- ingredient names
- possible use
- possible benefits
- safety or precautions
"""

        # ----------------------------------------------------
        # 5. RAG RETRIEVAL
        # ----------------------------------------------------
        print("[3] Searching medical knowledge base...")

        retrieval_start = time.time()

        medical_results = retriever.retrieve(
            query=medical_query,
            top_k=5,
            max_distance_threshold=0.35,
            category_filter=predicted_condition,
            type_filter="medical"
        )

        diet_results = retriever.retrieve(
            query=diet_query,
            top_k=5,
            max_distance_threshold=0.35,
            category_filter=predicted_condition,
            type_filter="diet"
        )

        home_results = retriever.retrieve(
            query=home_query,
            top_k=5,
            max_distance_threshold=0.35,
            category_filter=predicted_condition,
            type_filter="home_remedy"
        )

        # Fallback handling
        if not medical_results:
            print("[3] No category/type-specific medical results. Running fallback...")
            medical_results = retriever.retrieve(
                query=medical_query,
                top_k=5,
                max_distance_threshold=0.35,
                category_filter=predicted_condition
            )

            medical_results = [
                r for r in medical_results
                if r.get("type") not in ["diet", "home_remedy"]
            ]

        if not medical_results:
            medical_results = retriever.retrieve(
                query=medical_query,
                top_k=5,
                max_distance_threshold=0.35,
                category_filter=None
            )

        # ----------------------------------------------------
        # 6. FILTER CITATIONS AND RANK
        # ----------------------------------------------------
        medical_results = [
            r for r in medical_results
            if not looks_like_citation(r["text"])
        ]

        diet_results = [
            r for r in diet_results
            if not looks_like_citation(r["text"])
        ]

        home_results = [
            r for r in home_results
            if not looks_like_citation(r["text"])
        ]

        priority_terms = [
            "cause", "causes", "inflammation", "melanin",
            "hyperpigmentation", "ultraviolet", "sun exposure", "clinical"
        ]

        useful_medical = []

        for res in medical_results:
            text_lower = res["text"].lower()

            score = sum(
                term in text_lower
                for term in priority_terms
            )

            useful_medical.append((score, res))

        useful_medical.sort(
            key=lambda item: (
                -item[0],
                item[1].get("distance", 0)
            )
        )

        if useful_medical:
            medical_results = [item[1] for item in useful_medical]

        medical_results = medical_results[:3]

        diet_results = diet_results[:5]

        home_results = home_results[:5]

        retrieval_time = time.time() - retrieval_start

        print(f"[3] RAG retrieval completed in {retrieval_time:.2f}s")

        print(f"[3] Medical results: {len(medical_results)}")

        print(f"[3] Diet results: {len(diet_results)}")

        print(f"[3] Home results: {len(home_results)}")

        # ----------------------------------------------------
        # 7. BUILD CONTEXT & SOURCES
        # ----------------------------------------------------
        medical_context = build_context(medical_results, "Medical Evidence")

        diet_context = build_context(diet_results, "Diet Evidence")

        home_context = build_context(home_results, "Home Ingredient Evidence")

        medical_sources = [
            {
                "document": r["source"],
                "page": r["page"],
                "type": r.get("type", "medical")
            }
            for r in medical_results
        ]

        diet_sources = [
            {
                "document": r["source"],
                "page": r["page"],
                "type": r.get("type", "diet")
            }
            for r in diet_results
        ]

        home_sources = [
            {
                "document": r["source"],
                "page": r["page"],
                "type": r.get("type", "home_remedy")
            }
            for r in home_results
        ]

        # ----------------------------------------------------
        # 8. GEMMA PROMPT
        # ----------------------------------------------------
        prompt_text = f"""
A skin-condition classifier predicts:
{display_name}

Classifier confidence:
{confidence:.4f}

The user asked:
{user_text}

================ MEDICAL EVIDENCE ================
{medical_context}

================ DIET & NUTRIENTS EVIDENCE ================
{diet_context}

================ HOME INGREDIENTS EVIDENCE ================
{home_context}

Using ONLY the retrieved evidence, answer the user.

Return exactly these sections:

CONDITION
Briefly explain what the predicted condition may involve.

POSSIBLE CAUSES / CONTRIBUTING FACTORS
Explain causes or contributing factors supported by the medical evidence.

DIET & NUTRIENTS
Give the relevant foods and nutrients supported by the Diet & Nutrients evidence.
Do not invent foods or nutrients.
If there is no relevant evidence, say:
"No relevant dietary evidence retrieved."

HOME INGREDIENTS
Give the relevant home/topical ingredients supported by the Home Ingredients evidence.
Mention use or precautions ONLY when supported by the evidence.
Do not invent remedies.
If there is no relevant evidence, say:
"No relevant home-ingredient evidence retrieved."

IMPORTANT
Clearly state that this is informational analysis and not a definitive diagnosis.

RULES:
- Use ONLY the supplied evidence.
- Do not invent medical facts.
- Do not invent foods.
- Do not invent ingredients.
- Do not mix diet information into Home Ingredients.
- Do not mix Home Ingredients into Diet & Nutrients.
- Do not give a definitive diagnosis.
- Keep the response concise and easy to understand.
"""

        # ----------------------------------------------------
        # 9. CHAT MESSAGES
        # ----------------------------------------------------
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a careful medical-information assistant. "
                            "Use only the retrieved evidence. Never give a "
                            "definitive diagnosis and never invent information."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]

        # ----------------------------------------------------
        # 10. GEMMA INFERENCE
        # ----------------------------------------------------
        print("[4] Preparing Gemma input...")

        gemma_input_start = time.time()

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(gemma_model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        gemma_input_time = time.time() - gemma_input_start

        print(f"[4] Gemma input prepared in {gemma_input_time:.2f}s")

        input_len = inputs["input_ids"].shape[-1]

        print("[5] Starting Gemma generation...")

        gemma_start = time.time()

        with torch.inference_mode():
            generated_ids = gemma_model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=False
            )

        gemma_time = time.time() - gemma_start

        print(f"[5] Gemma generation completed in {gemma_time:.2f}s")

        generated_ids = generated_ids[0][input_len:]

        response_text = processor.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        print("\n[6] Gemma response:")
        print(response_text)

        total_time = time.time() - start_time

        print(f"\n[7] Total request time: {total_time:.2f}s")
        print("========================================\n")

        # ----------------------------------------------------
        # 11. RESPONSE
        # ----------------------------------------------------
        return JSONResponse(
            {
                "status": "success",
                "image_url": f"/uploads/{unique_filename}",
                "prediction": {
                    "class": predicted_condition,
                    "display_name": display_name,
                    "confidence": round(confidence, 4)
                },
                "user_query": user_text,
                "model_response": response_text,
                "diet_nutrients": diet_results,
                "home_ingredients": home_results,
                "medical_sources": medical_sources,
                "diet_sources": diet_sources,
                "home_sources": home_sources,
                "timing": {
                    "efficientnet_seconds": round(efficientnet_time, 2),
                    "rag_seconds": round(retrieval_time, 2),
                    "gemma_seconds": round(gemma_time, 2),
                    "total_seconds": round(total_time, 2)
                },
                "disclaimer": (
                    "This system provides informational analysis only and "
                    "is not a substitute for clinical medical evaluation."
                )
            }
        )

    except Exception as e:
        print("\n[ERROR] Analysis failed:")

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )