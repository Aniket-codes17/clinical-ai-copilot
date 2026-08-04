import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoModelForImageClassification


class HuggingFaceModelWrapper(torch.nn.Module):

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        return self.hf_model(x).logits


def classify_image(
    image_path: str, model_path: str = "./final_pneumonia_cnn_model"
):
    """Analyzes a chest X-ray image and returns diagnosis, confidence, and Grad-CAM heatmap path."""
    if not os.path.exists(image_path):
        return {"error": f"Image file not found at path: {image_path}"}

    os.makedirs("./results", exist_ok=True)
    heatmap_output_path = "./results/marked_abnormality_heatmap.png"

    # 1. Load fine-tuned model
    hf_model = AutoModelForImageClassification.from_pretrained(model_path)
    wrapped_model = HuggingFaceModelWrapper(hf_model)
    wrapped_model.eval()

    # 2. Preprocess input image
    rgb_img = Image.open(image_path).convert("RGB").resize((224, 224))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = transform(rgb_img).unsqueeze(0)

    # 3. Model Inference
    with torch.no_grad():
        logits = wrapped_model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        pred_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][pred_class].item()

    # Label mapping (0: Normal, 1: Pneumonia)
    labels = {0: "Normal", 1: "Pneumonia"}
    prediction_label = labels.get(pred_class, "Unknown")

    # 4. Generate Grad-CAM Heatmap
    target_layers = [hf_model.resnet.encoder.stages[-1].layers[-1]]
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    img_array = np.float32(rgb_img) / 255.0
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    cv2.imwrite(
        heatmap_output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    )

    return {
        "prediction": prediction_label,
        "confidence": round(confidence * 100, 2),
        "heatmap_path": heatmap_output_path,
    }


if __name__ == "__main__":
    test_image = (
        "./results/xray.png"
        if os.path.exists("./results/xray.png")
        else "xray.png"
    )
    result = classify_image(test_image)
    print("\n--- Imaging Tool Analysis Output ---")
    print(result)