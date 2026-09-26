"""
tools/image_tool.py
--------------------
Chest radiograph analysis using a pretrained multi-label model, with
Grad-CAM explainability for the top finding.

Uses torchxrayvision's DenseNet121 ("all" weights), trained across
several large public chest X-ray datasets (NIH ChestX-ray14, CheXpert,
PadChest, MIMIC-CXR). No fine-tuning or local model directory is
required -- weights download once and are cached by the library.

Unlike a single normal/abnormal classifier, this model scores 18 findings
simultaneously, covering both regions in scope for this project:

  Heart:  Cardiomegaly, Enlarged Cardiomediastinum
  Lungs:  Pneumonia, Effusion, Atelectasis, Pneumothorax, Consolidation,
          Edema, Emphysema, Fibrosis, Nodule, Mass, Infiltration,
          Lung Opacity, Lung Lesion, Pleural_Thickening, Fracture, Hernia

`classify_image()` returns every finding, not just one, since a chest
X-ray can show more than one abnormality (or none) at a time.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image, UnidentifiedImageError
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger("clinical_agent.image_tool")

RESULTS_DIR = Path("./results")
IMAGE_SIZE = 224
DEFAULT_WEIGHTS = "densenet121-res224-all"
DEFAULT_THRESHOLD = 0.5  # fallback only -- the model's own per-pathology
                          # calibrated thresholds (op_threshs) are used
                          # when available; see _load_model().

HEART_FINDINGS = {"Cardiomegaly", "Enlarged Cardiomediastinum"}


# --------------------------------------------------------------------------- #
# Model loading -- cached so weights load once per process, not per call.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _load_model(weights: str) -> tuple[torch.nn.Module, torch.device, list, np.ndarray | None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading torchxrayvision model '%s' onto %s...", weights, device)

    model = xrv.models.DenseNet(weights=weights)
    model.to(device).eval()

    target_layers = _resolve_target_layers(model)

    # Some torchxrayvision weight sets ship calibrated per-pathology
    # decision thresholds; use them instead of one flat cutoff for every
    # finding when they're available.
    thresholds = getattr(model, "op_threshs", None)
    if thresholds is not None:
        thresholds = thresholds.detach().cpu().numpy()

    return model, device, target_layers, thresholds


def _resolve_target_layers(model: torch.nn.Module) -> list:
    """Grad-CAM needs the last convolutional block. Try the specific
    DenseNet121 layer first; fall back to the last child of `.features`
    (structurally guaranteed to be the final feature-extraction layer)
    so this keeps working even if the exact internal naming differs."""
    try:
        return [model.features.denseblock4.denselayer16.conv2]
    except AttributeError:
        pass
    try:
        return [list(model.features.children())[-1]]
    except AttributeError as exc:
        raise AttributeError(
            "Could not locate a convolutional layer under 'model.features' "
            "for Grad-CAM. Update _resolve_target_layers() for this model."
        ) from exc


# --------------------------------------------------------------------------- #
# Preprocessing -- torchxrayvision models expect a specific normalization
# (rescaled to roughly [-1024, 1024], single channel), NOT the standard
# ImageNet mean/std used for typical torchvision/HF models.
# --------------------------------------------------------------------------- #
def _preprocess(image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Returns (model_input_tensor, display_image_rgb_0_1) -- the display
    image is a separate, human-viewable copy used only for the Grad-CAM
    overlay, since the model's own normalized tensor isn't renderable."""
    pil_image = Image.open(image_path).convert("L")  # grayscale, as X-rays are single-channel
    display_image = np.array(pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    display_image = np.float32(display_image) / 255.0

    img_array = np.array(pil_image)
    img_array = xrv.datasets.normalize(img_array, 255)  # 8-bit range -> xrv's expected range
    img_array = img_array[None, ...]  # add channel dim: (1, H, W)

    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(IMAGE_SIZE)]
    )
    img_array = transform(img_array)
    input_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, 1, H, W)

    return input_tensor, display_image


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def classify_image(image_path: str, weights: str = DEFAULT_WEIGHTS) -> dict[str, Any]:
    """Analyze a chest X-ray for lung and heart findings.

    Returns a dict with:
      - "findings": every pathology the model scores, sorted by
        probability descending, each as
        {"label": str, "probability": float 0-100, "region": "heart"|"lung", "flagged": bool}
      - "top_finding" / "top_confidence": the single highest-probability
        finding, for a compact summary view.
      - "heatmap_path": Grad-CAM overlay for the top finding.
    Returns {"error": "..."} on any failure rather than raising, since
    this is called directly from the Streamlit UI and an agent tool.
    """
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        return {"error": f"Image file not found at path: {image_path}"}

    try:
        model, device, target_layers, thresholds = _load_model(weights)
    except Exception as exc:  # noqa: BLE001 -- e.g. weight download failure
        logger.exception("Model could not be loaded (weights='%s').", weights)
        return {"error": f"Could not load chest X-ray model: {exc}"}

    try:
        input_tensor, display_image = _preprocess(image_path_obj)
    except UnidentifiedImageError:
        return {"error": f"'{image_path}' is not a readable image file."}

    try:
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            raw_output = model(input_tensor)[0].detach().cpu().numpy()

        pathologies = model.pathologies
        findings = []
        for idx, (label, probability) in enumerate(zip(pathologies, raw_output)):
            if not label:  # torchxrayvision pads unused label slots with ""
                continue
            cutoff = float(thresholds[idx]) if thresholds is not None else DEFAULT_THRESHOLD
            findings.append(
                {
                    "label": label,
                    "probability": round(float(probability) * 100, 1),
                    "region": "heart" if label in HEART_FINDINGS else "lung",
                    "flagged": bool(probability >= cutoff),
                }
            )
        findings.sort(key=lambda f: f["probability"], reverse=True)

        top = findings[0]
        top_idx = pathologies.index(top["label"])

        # Grad-CAM for the top finding specifically -- explains *why* that
        # particular pathology was scored highest, not a generic saliency map.
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(top_idx)],
        )[0, :]
        visualization = show_cam_on_image(display_image, grayscale_cam, use_rgb=True)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        heatmap_path = RESULTS_DIR / f"{image_path_obj.stem}_heatmap.png"
        cv2.imwrite(str(heatmap_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        logger.info(
            "Top finding for %s: %s (%.1f%%). %d finding(s) flagged.",
            image_path_obj.name, top["label"], top["probability"],
            sum(1 for f in findings if f["flagged"]),
        )

        return {
            "findings": findings,
            "top_finding": top["label"],
            "top_confidence": top["probability"],
            # Kept for compatibility with any caller still expecting the
            # old single-label shape.
            "prediction": top["label"] if top["flagged"] else "No significant findings",
            "confidence": top["probability"],
            "heatmap_path": str(heatmap_path),
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Classification failed for %s", image_path)
        return {"error": f"Classification failed: {exc}"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    import os
    test_image = "./results/xray.png" if os.path.exists("./results/xray.png") else "xray.png"
    result = classify_image(test_image)
    print("\n--- Imaging Tool Analysis Output ---")
    print(result)