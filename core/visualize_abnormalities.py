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
        output = self.hf_model(x)
        return output.logits



os.makedirs("./results", exist_ok=True)

model_path = "./final_pneumonia_cnn_model"
print(f"Loading custom fine-tuned model from {model_path}...")

hf_model = AutoModelForImageClassification.from_pretrained(model_path)
wrapped_model = HuggingFaceModelWrapper(hf_model)
wrapped_model.eval()


target_layers = [hf_model.resnet.encoder.stages[-1].layers[-1]]

image_path = (
    "./results/xray.png" if os.path.exists("./results/xray.png") else "xray.png"
)

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


cam = GradCAM(model=wrapped_model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]


img_array = np.float32(rgb_img) / 255.0
visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

output_path = "./results/marked_abnormality_heatmap.png"
cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

print(f"\nSuccessfully generated clinical heatmap! Saved to: {output_path}")