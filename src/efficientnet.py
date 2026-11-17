import torch
import timm
from PIL import Image
from torchvision import transforms
import os


class EfficientNetPredictor:

    def __init__(self, model_path=None):

        # -----------------------------
        # Model path
        # -----------------------------

        if model_path is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

            model_path = os.path.join(
                project_root,
                "models",
                "efficientnet_b2_skin_issues.pth"
            )

        # -----------------------------
        # Device
        # -----------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print("EfficientNet device:", self.device)

        # -----------------------------
        # Classes
        # -----------------------------

        self.class_names = [
            "acne",
            "blackheads",
            "dark_spots",
            "pores",
            "wrinkles"
        ]

        # -----------------------------
        # Create model
        # -----------------------------

        self.model = timm.create_model(
            "efficientnet_b2",
            pretrained=True,
            num_classes=5
        )

        # -----------------------------
        # Load trained weights
        # -----------------------------

        state_dict = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)

        self.model.eval()

        # -----------------------------
        # Preprocessing
        # -----------------------------

        self.transform = transforms.Compose([

            transforms.Resize((256, 256)),

            transforms.CenterCrop((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        print("EfficientNet-B2 loaded successfully!")

    # =================================
    # PREDICT
    # =================================

    def predict(self, image):

        # -----------------------------
        # Convert image to RGB
        # -----------------------------

        if isinstance(image, str):

            image = Image.open(image).convert("RGB")

        else:

            image = image.convert("RGB")

        # -----------------------------
        # Transform image
        # -----------------------------

        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self.device)

        # -----------------------------
        # Prediction
        # -----------------------------

        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        # -----------------------------
        # Predicted class
        # -----------------------------

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        predicted_class = self.class_names[predicted_index]

        confidence = probabilities[
            0,
            predicted_index
        ].item()

        # -----------------------------
        # All probabilities
        # -----------------------------

        all_probabilities = {
            class_name: probability.item()
            for class_name, probability in zip(
                self.class_names,
                probabilities[0]
            )
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": all_probabilities
        }