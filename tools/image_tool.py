"""
tools/image_tool.py
--------------------
Chest-radiograph classification with Grad-CAM explainability.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoModelForImageClassification

logger = logging.getLogger("clinical_agent.image_tool")

RESULTS_DIR = Path("./results")
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# Fallback only -- real labels are read from the model's own config
# (id2label) whenever it's available, since that's the source of truth
# for whatever the model was actually fine-tuned on.
FALLBACK_LABELS = {0: "Normal", 1: "Pneumonia"}


class HuggingFaceModelWrapper(torch.nn.Module):
    """Adapts a Hugging Face classifier to the plain-tensor-in/-out
    interface pytorch-grad-cam expects."""

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hf_model(x).logits


# --------------------------------------------------------------------------- #
# Model loading -- cached so it happens once per model_path, not once per
# classify_image() call. Reloading a HF model from disk on every request
# is the single most expensive thing in this file; with caching, every
# call after the first reuses the already-loaded weights.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _load_model(model_path: str) -> tuple[torch.nn.Module, HuggingFaceModelWrapper, torch.device, list]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at '{model_path}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model from %s onto %s...", model_path, device)

    hf_model = AutoModelForImageClassification.from_pretrained(model_path)
    hf_model.to(device).eval()

    wrapped_model = HuggingFaceModelWrapper(hf_model).to(device).eval()
    target_layers = _resolve_target_layers(hf_model)

    return hf_model, wrapped_model, device, target_layers


def _resolve_target_layers(hf_model: torch.nn.Module) -> list:
    try:
        return [hf_model.resnet.encoder.stages[-1].layers[-1]]
    except AttributeError as exc:
        raise AttributeError(
            "Could not locate a ResNet-style final layer at "
            "'hf_model.resnet.encoder.stages[-1].layers[-1]'. If this model "
            "uses a different architecture, update _resolve_target_layers() "
            "to point at its last convolutional block."
        ) from exc


def _id_to_label(hf_model: torch.nn.Module, class_idx: int) -> str:
    id2label = getattr(hf_model.config, "id2label", None)
    if id2label:
        return str(id2label.get(class_idx, id2label.get(str(class_idx), FALLBACK_LABELS.get(class_idx, "Unknown"))))
    return FALLBACK_LABELS.get(class_idx, "Unknown")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def classify_image(
    image_path: str, model_path: str = "./final_pneumonia_cnn_model"
) -> dict[str, Any]:
    """Analyzes a chest X-ray image and returns diagnosis, confidence, and
    Grad-CAM heatmap path. Returns {"error": "..."} on any failure rather
    than raising, since this is called directly from the Streamlit UI.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image file not found at path: {image_path}"}

    try:
        hf_model, wrapped_model, device, target_layers = _load_model(model_path)
    except (FileNotFoundError, AttributeError) as exc:
        logger.exception("Model could not be loaded from %s.", model_path)
        return {"error": str(exc)}

    try:
        rgb_img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    except UnidentifiedImageError:
        return {"error": f"'{image_path}' is not a readable image file."}

    try:
        input_tensor = _TRANSFORM(rgb_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = wrapped_model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            pred_class = int(torch.argmax(probabilities, dim=-1).item())
            confidence = float(probabilities[0][pred_class].item())

        prediction_label = _id_to_label(hf_model, pred_class)

        # Grad-CAM needs its own forward+backward pass with gradients
        # enabled, so it's kept separate from the no_grad prediction above.
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

        img_array = np.float32(rgb_img) / 255.0
        visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # Named per input image rather than one fixed filename, so
        # concurrent requests (different patients/uploads) can't overwrite
        # each other's heatmap between save and display.
        heatmap_output_path = RESULTS_DIR / f"{Path(image_path).stem}_heatmap.png"
        cv2.imwrite(str(heatmap_output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        logger.info(
            "Classified %s as '%s' (%.1f%% confidence). Heatmap: %s",
            image_path, prediction_label, confidence * 100, heatmap_output_path,
        )

        return {
            "prediction": prediction_label,
            "confidence": round(confidence * 100, 2),
            "heatmap_path": str(heatmap_output_path),
        }

    except Exception as exc:  # noqa: BLE001 -- surface any failure to the caller
        logger.exception("Classification failed for %s", image_path)
        return {"error": f"Classification failed: {exc}"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    test_image = (
        "./results/xray.png" if os.path.exists("./results/xray.png") else "xray.png"
    )
    result = classify_image(test_image)
    print("\n--- Imaging Tool Analysis Output ---")
    print(result)