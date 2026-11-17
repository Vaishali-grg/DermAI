import os
import json


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = (
    "dataset/Skin v2"
)

OUTPUT_FILE = (
    "dataset/multimodal_instructions.jsonl"
)


# ============================================================
# CLASS MAP
#
# Keys are the folder names on disk, values are the canonical
# class names used everywhere else in the pipeline (must match
# EfficientNetPredictor.class_names and the "category" metadata
# stored in ChromaDB).
# ============================================================

CLASS_MAP = {

    "acne": "acne",

    "blackheades": "blackheads",

    "dark spots": "dark_spots",

    "pores": "pores",

    "wrinkles": "wrinkles"
}


# ============================================================
# USER TEXT TEMPLATES
# ============================================================

USER_TEXTS = {

    "acne": [
        "I have acne on my face. What could be causing it?",
        "I am getting pimples and acne. What should I know?",
        "My facial skin has acne. Can you explain possible causes?",
        "I have acne and want some general skincare guidance.",
        "Why might I be getting acne on my face?"
    ],

    "blackheads": [
        "I have blackheads on my face. What could cause them?",
        "I have small dark-looking spots around my nose. What could they be?",
        "Why am I getting blackheads?",
        "What causes facial blackheads?",
        "I have blackheads and want general skincare information."
    ],

    "dark_spots": [
        "I have dark spots on my face. What could cause them?",
        "My face has darker patches. What could this be?",
        "I have facial hyperpigmentation. What are possible causes?",
        "Why do I have dark spots on my skin?",
        "I have dark marks after skin inflammation. What could explain them?"
    ],

    "pores": [
        "My facial pores look enlarged. What could cause this?",
        "Why do my pores look large?",
        "I have visible enlarged pores. What are possible contributing factors?",
        "My skin has very visible pores. What should I know?",
        "What can contribute to enlarged facial pores?"
    ],

    "wrinkles": [
        "I have facial wrinkles. What could contribute to them?",
        "Why am I developing wrinkles?",
        "I have visible wrinkles and want general information.",
        "What factors can contribute to facial wrinkles?",
        "My facial skin has wrinkles. What should I know?"
    ]
}


# ============================================================
# CREATE DATASET
#
# NOTE: this stage intentionally does NOT create the training
# "target" text. Real targets need actual retrieved medical
# evidence to be grounded in, so they are built in
# build_rag_multimodal_dataset.py instead. Writing a fake/
# placeholder target here was the previous bug -- it taught
# the model to output generic boilerplate instead of an answer
# actually grounded in the retrieved evidence.
# ============================================================

os.makedirs("dataset", exist_ok=True)

count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    for folder_name, class_name in CLASS_MAP.items():

        folder_path = os.path.join(DATASET_ROOT, folder_name)

        if not os.path.isdir(folder_path):
            print("Folder not found:", folder_path)
            continue

        user_templates = USER_TEXTS[class_name]

        image_index = 0

        for file_name in os.listdir(folder_path):

            if not file_name.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                continue

            image_path = os.path.join(folder_path, file_name)

            user_text = user_templates[
                image_index % len(user_templates)
            ]

            instruction = f"""
A user has uploaded a facial image.

User message:
{user_text}

A specialized skin classifier predicts:
{class_name.replace("_", " ")}

Analyze the image together with the user's message.

Give a concise medical-information response.

Do not provide a definitive diagnosis.
Use medical evidence supplied during retrieval.
"""

            record = {
                "image": image_path,
                "label": class_name,
                "user_text": user_text,
                "instruction": instruction
            }

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            count += 1
            image_index += 1


print("\nCreated:", OUTPUT_FILE)
print("Examples:", count)