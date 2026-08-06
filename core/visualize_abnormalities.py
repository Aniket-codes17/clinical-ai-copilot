"""
tools/image_tool.py
--------------------
Chest-radiograph classification with Grad-CAM explainability.

This module wraps a fine-tuned Hugging Face image-classification model
(expected to be a ResNet-style architecture) so it can be called once per
uploaded image via `classify_image(path)`, rather than run as a one-shot
script against a hardcoded file. It returns both the model's prediction
and a saved Grad-CAM heatmap highlighting the regions that drove it.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoModelForImageClassification

logger = logging.getLogger("clinical_agent.image_tool")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_PATH = Path("./final_pneumonia_cnn_model")
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


class HuggingFaceModelWrapper(torch.nn.Module):
    """Adapts a Hugging Face classifier to the plain-tensor-in/-out
    interface pytorch-grad-cam expects (it calls the model directly and
    expects a tensor of logits back, not a HF ModelOutput object)."""

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hf_model(x).logits


# --------------------------------------------------------------------------- #
# Model loading (cached so it only happens once per process, not per request)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_model() -> tuple[torch.nn.Module, HuggingFaceModelWrapper, torch.device, list]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model directory not found at '{MODEL_PATH}'. "
            "Make sure the fine-tuned model has been downloaded/exported there."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading fine-tuned model from %s onto %s...", MODEL_PATH, device)

    hf_model = AutoModelForImageClassification.from_pretrained(str(MODEL_PATH))
    hf_model.to(device).eval()

    wrapped_model = HuggingFaceModelWrapper(hf_model).to(device).eval()

    target_layers = _resolve_target_layers(hf_model)

    return hf_model, wrapped_model, device, target_layers


def _resolve_target_layers(hf_model: torch.nn.Module) -> list:
    """Locate the last conv block for Grad-CAM. Raises a clear error rather
    than a cryptic AttributeError if the architecture doesn't match the
    expected HF ResNet layout."""
    try:
        return [hf_model.resnet.encoder.stages[-1].layers[-1]]
    except AttributeError as exc:
        raise AttributeError(
            "Could not locate a ResNet-style final layer at "
            "'hf_model.resnet.encoder.stages[-1].layers[-1]'. If this model "
            "uses a different architecture, update _resolve_target_layers() "
            "to point at its last convolutional block."
        ) from exc


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _load_and_preprocess(image_path: Path) -> tuple[Image.Image, torch.Tensor]:
    rgb_image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    input_tensor = _TRANSFORM(rgb_image).unsqueeze(0)
    return rgb_image, input_tensor


def _id_to_label(hf_model: torch.nn.Module, class_idx: int) -> str:
    id2label = getattr(hf_model.config, "id2label", None) or {}
    # HF configs sometimes key id2label by int, sometimes by str -- handle both.
    return str(id2label.get(class_idx, id2label.get(str(class_idx), f"CLASS_{class_idx}")))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def classify_image(image_path: str) -> dict[str, Any]:
    """Classify a chest radiograph and generate a Grad-CAM heatmap.

    Returns a dict with keys `prediction`, `confidence` (0-100), and
    `heatmap_path` on success, or `{"error": "..."}` on failure -- callers
    (e.g. the Streamlit app) should check for the `error` key before
    reading the other fields.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return {"error": f"Image not found at '{image_path}'."}

    try:
        hf_model, wrapped_model, device, target_layers = _load_model()
    except (FileNotFoundError, AttributeError) as exc:
        logger.exception("Model could not be loaded.")
        return {"error": str(exc)}

    try:
        rgb_image, input_tensor = _load_and_preprocess(image_path)
        input_tensor = input_tensor.to(device)

        # Prediction: a plain forward pass, no gradients needed.
        with torch.no_grad():
            logits = hf_model(input_tensor).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            class_idx = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[class_idx].item()) * 100

        prediction = _id_to_label(hf_model, class_idx)

        # Grad-CAM: needs its own forward+backward pass with gradients
        # enabled, so it's kept separate from the no_grad prediction above.
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

        img_array = np.float32(rgb_image) / 255.0
        visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        heatmap_path = RESULTS_DIR / f"{image_path.stem}_heatmap.png"
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        logger.info(
            "Classified %s as '%s' (%.1f%% confidence). Heatmap saved to %s.",
            image_path.name, prediction, confidence, heatmap_path,
        )

        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "heatmap_path": str(heatmap_path),
        }

    except Exception as exc:  # noqa: BLE001 -- surface any failure to the caller
        logger.exception("Classification failed for %s", image_path)
        return {"error": f"Classification failed: {exc}"}


if __name__ == "__main__":
    # Quick manual check: classify whatever sample image is available.
    sample = "./results/xray.png" if Path("./results/xray.png").exists() else "xray.png"
    print(classify_image(sample))