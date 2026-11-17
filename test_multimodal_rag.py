import os

# ============================================================
# DISABLE TORCH COMPILE / DYNAMO
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"


# ============================================================
# IMPORTS
# ============================================================

import torch
from PIL import Image

from src.gemma import GemmaModel
from src.siglip import SigLIPEncoder
from src.vision_projection import VisionProjection
from src.retriever import MedicalRetriever


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = "test_images/testgirl.jpg"

PROJECTION_PATH = (
    "models/vision_projection/best_model.pth"
)

SIGLIP_DIM = 768
GEMMA_DIM = 2304

TOP_K_PER_QUERY = 5

MAX_FINAL_CONTEXT_CHUNKS = 4

MAX_NEW_TOKENS = 100


# ============================================================
# CURRENT EFFICIENTNET RESULT
# ============================================================

# Your verified EfficientNet result:
#
# Prediction: dark_spots
#
# We use this here while testing the complete
# multimodal-RAG pipeline.

PREDICTED_CONDITION = "dark_spots"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("====================================")
print("TARGETED MULTIMODAL RAG TEST")
print("====================================")

print(
    "Device:",
    device
)


# ============================================================
# 1. LOAD GEMMA
# ============================================================

print("\n====================================")
print("LOADING GEMMA")
print("====================================")

gemma = GemmaModel()

print(
    "Gemma loaded successfully!"
)


# ============================================================
# 2. LOAD SIGLIP
# ============================================================

print("\n====================================")
print("LOADING SIGLIP")
print("====================================")

siglip = SigLIPEncoder()

print(
    "SigLIP loaded successfully!"
)


# ============================================================
# 3. LOAD TRAINED VISION PROJECTION
# ============================================================

print("\n====================================")
print("LOADING TRAINED VISION PROJECTION")
print("====================================")

projection = VisionProjection(
    siglip_dim=SIGLIP_DIM,
    gemma_dim=GEMMA_DIM
)


projection.load_state_dict(
    torch.load(
        PROJECTION_PATH,
        map_location=device,
        weights_only=True
    )
)


# Projection was trained in float32.

projection = projection.to(
    device=device,
    dtype=torch.float32
)

projection.eval()


print(
    "Trained projection loaded successfully!"
)

print(
    "Checkpoint:",
    PROJECTION_PATH
)


# ============================================================
# 4. LOAD MEDICAL RETRIEVER
# ============================================================

print("\n====================================")
print("LOADING MEDICAL RETRIEVER")
print("====================================")

retriever = MedicalRetriever(
    db_directory="./vector_db"
)

print(
    "Medical Retriever loaded successfully!"
)


# ============================================================
# 5. LOAD IMAGE
# ============================================================

print("\n====================================")
print("LOADING IMAGE")
print("====================================")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    "Image:",
    IMAGE_PATH
)


# ============================================================
# 6. EFFICIENTNET RESULT
# ============================================================

print("\n====================================")
print("EFFICIENTNET RESULT")
print("====================================")

print(
    "Predicted condition:",
    PREDICTED_CONDITION
)


# ============================================================
# 7. TARGETED RAG QUERIES
# ============================================================

condition_name = (
    PREDICTED_CONDITION.replace(
        "_",
        " "
    )
)


rag_queries = [

    "post-inflammatory hyperpigmentation",

    "facial hyperpigmentation",

    "causes of facial dark spots"

]


print("\n====================================")
print("TARGETED RAG QUERIES")
print("====================================")


for index, query in enumerate(
    rag_queries
):

    print(
        f"{index + 1}. {query}"
    )


# ============================================================
# 8. MULTI-QUERY RETRIEVAL
# ============================================================

print("\n====================================")
print("MULTI-QUERY RAG RETRIEVAL")
print("====================================")


all_results = []


for query in rag_queries:

    print("\n------------------------------------")

    print(
        "RAG QUERY:",
        query
    )


    results = retriever.retrieve(

        query=query,

        top_k=TOP_K_PER_QUERY,

        max_distance_threshold=0.35,

        category_filter=PREDICTED_CONDITION

    )


    print(
        "Useful chunks:",
        len(results)
    )


    all_results.extend(
        results
    )


# ============================================================
# 9. REMOVE DUPLICATES
# ============================================================

unique_results = {}


for result in all_results:

    chunk_id = result["chunk_id"]

    if chunk_id not in unique_results:

        unique_results[
            chunk_id
        ] = result


unique_results = list(
    unique_results.values()
)


# ============================================================
# 10. RANK RESULTS
# ============================================================
#
# IMPORTANT:
#
# We do NOT use "combined_score" here.
#
# We use:
#
# 1. quality_score if available
# 2. retrieval distance
#
# This makes the script compatible with both versions
# of your retriever.
# ============================================================

def get_quality_score(
    result
):

    return float(
        result.get(
            "quality_score",
            0.0
        )
    )


def get_distance(
    result
):

    return float(
        result.get(
            "distance",
            999.0
        )
    )


selected_results = sorted(

    unique_results,

    key=lambda result: (

        -get_quality_score(
            result
        ),

        get_distance(
            result
        )

    )

)[
    :MAX_FINAL_CONTEXT_CHUNKS
]


# ============================================================
# 11. DISPLAY CLEAN CONTEXT
# ============================================================

print("\n====================================")
print("CLEAN MEDICAL CONTEXT")
print("====================================")


print(
    "Unique candidates:",
    len(unique_results)
)

print(
    "Selected chunks:",
    len(selected_results)
)


for index, result in enumerate(
    selected_results
):

    print("\n------------------------------------")

    print(
        f"SOURCE {index + 1}"
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
        "Category:",
        result["category"]
    )

    print(
        "Distance:",
        result["distance"]
    )

    print(
        "Quality:",
        result.get(
            "quality_score",
            "N/A"
        )
    )

    print(
        "Text:"
    )

    print(
        result["text"]
    )


# ============================================================
# 12. BUILD RAG CONTEXT
# ============================================================

if selected_results:

    context_parts = []


    for index, result in enumerate(
        selected_results
    ):

        source = result[
            "source"
        ]

        page = result[
            "page"
        ]

        text = result[
            "text"
        ]


        context_parts.append(

            f"[Evidence {index + 1}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"{text}"

        )


    rag_context = "\n\n".join(
        context_parts
    )


else:

    rag_context = (
        "No relevant medical evidence "
        "was retrieved."
    )


# ============================================================
# 13. BUILD GEMMA PROMPT
# ============================================================

prompt_text = f"""
You are a medical information assistant.

The visual classification model predicts:
{condition_name}

Use ONLY the medical evidence supplied below.

MEDICAL EVIDENCE:
{rag_context}

Provide a concise response using this format:

Possible condition:
State the predicted condition as a possibility.

Visible observation:
Only describe observations supported by the image model.
Do not invent acne, blackheads, pores, wrinkles, or other
conditions unless they are supported.

Possible causes or contributing factors:
Use only information supported by the medical evidence.

Medical context:
Summarize the relevant retrieved evidence.

Important:
- This is not a definitive diagnosis.
- Do not invent medical facts.
- Do not introduce unrelated skin conditions.
- Do not repeat sentences.
- Give one concise answer.

Answer:
"""


print("\n====================================")
print("FINAL GEMMA PROMPT")
print("====================================")

print(
    prompt_text
)


# ============================================================
# 14. IMAGE → SIGLIP PATCH TOKENS
# ============================================================

print("\n====================================")
print("SIGLIP IMAGE ENCODING")
print("====================================")


siglip_inputs = siglip.processor(
    images=image,
    return_tensors="pt"
)


siglip_inputs = {

    key: value.to(
        siglip.device
    )

    for key, value in siglip_inputs.items()

}


with torch.no_grad():

    vision_outputs = (
        siglip.model.vision_model(
            pixel_values=siglip_inputs[
                "pixel_values"
            ],
            return_dict=True
        )
    )


siglip_tokens = (
    vision_outputs.last_hidden_state
)


print(
    "SigLIP tokens:",
    siglip_tokens.shape
)


# ============================================================
# 15. SIGLIP → TRAINED PROJECTION
# ============================================================

print("\n====================================")
print("VISION PROJECTION")
print("====================================")


siglip_tokens = siglip_tokens.to(
    device=device,
    dtype=torch.float32
)


with torch.no_grad():

    image_tokens = projection(
        siglip_tokens
    )


# Gemma is using float16 on CUDA.

image_tokens = image_tokens.to(
    device=device,
    dtype=gemma.model.dtype
)


print(
    "Projected image tokens:",
    image_tokens.shape
)


# ============================================================
# 16. TOKENIZE PROMPT
# ============================================================

print("\n====================================")
print("TEXT ENCODING")
print("====================================")


text_inputs = gemma.tokenizer(

    prompt_text,

    return_tensors="pt",

    add_special_tokens=True,

    truncation=True,

    max_length=768

)


input_ids = (
    text_inputs["input_ids"]
    .to(device)
)


attention_mask = (
    text_inputs["attention_mask"]
    .to(device)
)


print(
    "Text tokens:",
    input_ids.shape
)


# ============================================================
# 17. TEXT → GEMMA EMBEDDINGS
# ============================================================

with torch.no_grad():

    text_embeddings = (
        gemma.model.get_input_embeddings()(
            input_ids
        )
    )


text_embeddings = text_embeddings.to(

    device=device,

    dtype=gemma.model.dtype

)


print(
    "Text embeddings:",
    text_embeddings.shape
)


# ============================================================
# 18. COMBINE IMAGE + TEXT
# ============================================================

print("\n====================================")
print("MULTIMODAL INPUT")
print("====================================")


combined_embeddings = torch.cat(

    [

        image_tokens,

        text_embeddings

    ],

    dim=1

)


print(
    "Combined embeddings:",
    combined_embeddings.shape
)


# ============================================================
# 19. ATTENTION MASK
# ============================================================

image_attention_mask = torch.ones(

    (

        image_tokens.shape[0],

        image_tokens.shape[1]

    ),

    dtype=attention_mask.dtype,

    device=device

)


combined_attention_mask = torch.cat(

    [

        image_attention_mask,

        attention_mask

    ],

    dim=1

)


print(
    "Combined attention mask:",
    combined_attention_mask.shape
)


# ============================================================
# 20. GEMMA GENERATION
# ============================================================

print("\n====================================")
print("GEMMA MULTIMODAL RAG GENERATION")
print("====================================")


with torch.no_grad():

    generated_ids = gemma.model.generate(

        inputs_embeds=combined_embeddings,

        attention_mask=combined_attention_mask,

        max_new_tokens=MAX_NEW_TOKENS,

        # ----------------------------------------------------
        # Conservative sampling
        # ----------------------------------------------------

        do_sample=True,

        temperature=0.55,

        top_p=0.80,

        repetition_penalty=1.30,

        no_repeat_ngram_size=4,

        pad_token_id=(
            gemma.tokenizer.eos_token_id
        ),

        use_cache=True

    )


# ============================================================
# 21. DECODE
# ============================================================

generated_text = gemma.tokenizer.decode(

    generated_ids[0],

    skip_special_tokens=True

)


# ============================================================
# 22. FINAL RESPONSE
# ============================================================

print("\n====================================")
print("FINAL MULTIMODAL RAG RESPONSE")
print("====================================")

print(
    generated_text
)


# ============================================================
# 23. PIPELINE SUMMARY
# ============================================================

print("\n====================================")
print("PIPELINE SUMMARY")
print("====================================")

print(
    "Image:",
    IMAGE_PATH
)

print(
    "Predicted condition:",
    PREDICTED_CONDITION
)

print(
    "Unique retrieved candidates:",
    len(unique_results)
)

print(
    "Final context chunks:",
    len(selected_results)
)

print(
    "SigLIP tokens:",
    siglip_tokens.shape
)

print(
    "Projected image tokens:",
    image_tokens.shape
)

print(
    "Text tokens:",
    input_ids.shape
)

print(
    "Combined embeddings:",
    combined_embeddings.shape
)

print("\n====================================")
print("TARGETED MULTIMODAL RAG TEST COMPLETED")
print("====================================")