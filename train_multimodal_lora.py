import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import random

import torch
from PIL import Image

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = (
    "google/paligemma2-3b-pt-224"
)

TRAIN_FILE = (
    "dataset/train_multimodal.jsonl"
)

OUTPUT_DIR = (
    "models/paligemma2_medical_lora"
)

MAX_TRAIN_SAMPLES = 500

NUM_EPOCHS = 1

LEARNING_RATE = 1e-4

SEED = 42


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if device.type != "cuda":

    raise RuntimeError(
        "PaliGemma QLoRA requires CUDA "
        "for this training configuration."
    )


print("====================================")
print("PALIGEMMA 2 3B QLoRA TRAINING")
print("====================================")

print(
    "Device:",
    device
)


# ============================================================
# SEED
# ============================================================

random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


# ============================================================
# LOAD DATA
# ============================================================

with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as file:

    records = [
        json.loads(line)
        for line in file
    ]


random.shuffle(
    records
)


records = records[
    :MAX_TRAIN_SAMPLES
]


print(
    "Training samples:",
    len(records)
)


# ============================================================
# PROCESSOR
# ============================================================

print("\nLoading PaliGemma processor...")

processor = (
    PaliGemmaProcessor
    .from_pretrained(
        MODEL_NAME
    )
)


# ============================================================
# QUANTIZATION
# ============================================================

compute_dtype = (

    torch.bfloat16

    if torch.cuda.is_bf16_supported()

    else torch.float16

)


quantization_config = (
    BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_use_double_quant=True,

        bnb_4bit_compute_dtype=
            compute_dtype

    )
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading PaliGemma 2 3B...")

model = (
    PaliGemmaForConditionalGeneration
    .from_pretrained(

        MODEL_NAME,

        quantization_config=
            quantization_config,

        device_map="auto",

        low_cpu_mem_usage=True

    )
)


# ============================================================
# PREPARE FOR KBIT TRAINING
# ============================================================

model = (
    prepare_model_for_kbit_training(
        model
    )
)


# ============================================================
# LoRA
# ============================================================

print("\nAdding LoRA...")

lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    bias="none",

    task_type=TaskType.CAUSAL_LM,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]

)


model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()


# ============================================================
# TRAIN MODE
# ============================================================

model.train()


# ============================================================
# OPTIMIZER
# ============================================================

trainable_parameters = [
    p
    for p in model.parameters()
    if p.requires_grad
]


optimizer = torch.optim.AdamW(

    trainable_parameters,

    lr=LEARNING_RATE,

    weight_decay=0.01

)


# ============================================================
# TRAINING
# ============================================================

best_loss = float(
    "inf"
)


for epoch in range(
    NUM_EPOCHS
):

    print("\n====================================")
    print(
        f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
    )
    print("====================================")


    random.shuffle(
        records
    )


    total_loss = 0.0


    for step, record in enumerate(
        records
    ):

        # ====================================================
        # IMAGE
        # ====================================================

        image = Image.open(
            record["image"]
        ).convert(
            "RGB"
        )


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = record[
            "instruction"
        ]


        # ====================================================
        # TARGET
        # ====================================================

        target = record[
            "target"
        ]


        # ====================================================
        # PROCESSOR
        #
        # PaliGemmaProcessor creates labels using suffix.
        # ====================================================

        inputs = processor(

            text=prompt,

            images=image,

            suffix=target,

            return_tensors="pt"

        )


        # ----------------------------------------------------
        # Move tensors
        # ----------------------------------------------------

        for key in inputs:

            if torch.is_tensor(
                inputs[key]
            ):

                inputs[key] = (
                    inputs[key].to(
                        device
                    )
                )


        # ====================================================
        # FORWARD
        # ====================================================

        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            **inputs
        )


        loss = outputs.loss


        # ====================================================
        # BACKPROP
        # ====================================================

        loss.backward()


        # ====================================================
        # CLIP
        # ====================================================

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0
        )


        optimizer.step()


        # ====================================================
        # LOSS
        # ====================================================

        loss_value = (
            loss.item()
        )


        total_loss += (
            loss_value
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            step == 0
            or (step + 1) % 25 == 0
        ):

            print(

                f"Step "
                f"{step + 1}/"
                f"{len(records)} "
                f"| Loss: "
                f"{loss_value:.4f}"

            )


        # ====================================================
        # MEMORY
        # ====================================================

        del (
            inputs,
            outputs,
            loss
        )


        torch.cuda.empty_cache()


    average_loss = (
        total_loss
        / len(records)
    )


    print(
        "\nAverage loss:",
        average_loss
    )


    # ========================================================
    # SAVE
    # ========================================================

    if average_loss < best_loss:

        best_loss = (
            average_loss
        )


        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )


        model.save_pretrained(
            OUTPUT_DIR
        )


        processor.save_pretrained(
            OUTPUT_DIR
        )


        print(
            "\n✅ PALIGEMMA LoRA CHECKPOINT SAVED"
        )


print("\n====================================")
print("PALIGEMMA QLoRA TRAINING COMPLETE")
print("====================================")

print(
    "Best loss:",
    best_loss
)