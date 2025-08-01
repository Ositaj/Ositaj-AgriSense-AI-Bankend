import torch
from torchvision import transforms
from PIL import Image
import io

# This function now accepts raw image bytes from file upload
def predict_disease(image_bytes: bytes) -> str:
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess image (resize + normalize)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Simulate model prediction (replace this with actual model logic later)
        predicted_label = "Leaf Spot"

        return predicted_label

    except Exception as e:
        return f"Error: {str(e)}"