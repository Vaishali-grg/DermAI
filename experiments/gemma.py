# src/gemma.py

import torch

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig
)


class GemmaModel:

    # ========================================================
    # MODEL
    # ========================================================

    MODEL_NAME = (
        "google/paligemma2-3b-pt-224"
    )

    def __init__(
        self,
        model_name=None,
        use_4bit=False
    ):

        if model_name is None:

            model_name = self.MODEL_NAME

        # ====================================================
        # DEVICE
        # ====================================================

        self.device = torch.device(

            "cuda"
            if torch.cuda.is_available()
            else "cpu"

        )

        print(
            "PaliGemma device:",
            self.device
        )

        print(
            "PaliGemma model:",
            model_name
        )

        # ====================================================
        # PROCESSOR
        # ====================================================

        print(
            "\nLoading PaliGemma processor..."
        )

        self.processor = (
            PaliGemmaProcessor
            .from_pretrained(
                model_name
            )
        )

        # ====================================================
        # QUANTIZATION
        # ====================================================

        quantization_config = None

        if (
            use_4bit
            and self.device.type == "cuda"
        ):

            print(
                "Using 4-bit QLoRA loading..."
            )

            quantization_config = (
                BitsAndBytesConfig(

                    load_in_4bit=True,

                    bnb_4bit_quant_type="nf4",

                    bnb_4bit_use_double_quant=True,

                    bnb_4bit_compute_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )

                )
            )

        # ====================================================
        # MODEL
        # ====================================================

        print(
            "\nLoading PaliGemma model..."
        )

        if quantization_config is not None:

            self.model = (
                PaliGemmaForConditionalGeneration
                .from_pretrained(

                    model_name,

                    quantization_config=
                        quantization_config,

                    device_map="auto",

                    low_cpu_mem_usage=True

                )
            )

        else:

            dtype = (

                torch.bfloat16

                if (
                    self.device.type == "cuda"
                    and torch.cuda.is_bf16_supported()
                )

                else torch.float16

                if self.device.type == "cuda"

                else torch.float32

            )

            self.model = (
                PaliGemmaForConditionalGeneration
                .from_pretrained(

                    model_name,

                    torch_dtype=dtype,

                    device_map="auto",

                    low_cpu_mem_usage=True

                )
            )

        self.model.eval()

        print(
            "\nPaliGemma loaded successfully!"
        )