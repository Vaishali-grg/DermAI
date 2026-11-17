import json
import random


INPUT_FILE = (
    "dataset/rag_multimodal_instructions.jsonl"
)

TRAIN_FILE = (
    "dataset/train_multimodal.jsonl"
)

VAL_FILE = (
    "dataset/val_multimodal.jsonl"
)

SEED = 42

VAL_RATIO = 0.10


# ============================================================
# LOAD
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as file:

    records = [
        json.loads(line)
        for line in file
    ]


print(
    "Total:",
    len(records)
)


# ============================================================
# SHUFFLE
# ============================================================

random.seed(
    SEED
)

random.shuffle(
    records
)


# ============================================================
# SPLIT
# ============================================================

val_size = int(
    len(records)
    * VAL_RATIO
)


val_records = records[
    :val_size
]

train_records = records[
    val_size:
]


# ============================================================
# SAVE
# ============================================================

with open(
    TRAIN_FILE,
    "w",
    encoding="utf-8"
) as file:

    for record in train_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


with open(
    VAL_FILE,
    "w",
    encoding="utf-8"
) as file:

    for record in val_records:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


print(
    "\nTraining:",
    len(train_records)
)

print(
    "Validation:",
    len(val_records)
)