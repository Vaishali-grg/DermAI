import torch

from transformers import (
    AutoProcessor,
    AutoModel
)


class SigLIPEncoder:

    def __init__(
        self,
        model_name="google/siglip-base-patch16-224"
    ):

        # -----------------------------
        # Device
        # -----------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "SigLIP device:",
            self.device
        )

        # -----------------------------
        # Processor
        # -----------------------------

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name
            )
        )

        # -----------------------------
        # Model
        # -----------------------------

        self.model = AutoModel.from_pretrained(
            model_name
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "SigLIP loaded successfully!"
        )

    # =================================
    # IMAGE EMBEDDING
    # =================================

    def get_image_embedding(
        self,
        image
    ):

        # -----------------------------
        # Preprocess image
        # -----------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # -----------------------------
        # Move to GPU
        # -----------------------------

        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }

        # -----------------------------
        # Get image features
        # -----------------------------

        with torch.no_grad():

            image_features = (
                self.model.get_image_features(
                    **inputs
                )
            )

        return image_features

# =================================
    # PATCH / VISUAL TOKENS
    # =================================

    def get_patch_tokens(self, image):

        # -----------------------------
        # Preprocess image
        # -----------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        # -----------------------------
        # Move to device
        # -----------------------------

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # -----------------------------
        # Vision Transformer
        # -----------------------------

        with torch.no_grad():

            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                return_dict=True
            )

        # -----------------------------
        # Patch-level visual tokens
        # -----------------------------

        patch_tokens = (
            vision_outputs.last_hidden_state
        )

        return patch_tokens