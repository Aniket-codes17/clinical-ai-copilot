import numpy as np
import torch
from transformers import AutoModelForImageClassification
from PIL import Image
import torchvision.transforms as transforms
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Create a wrapper to output a raw tensor instead of a Hugging Face object
class HuggingFaceModelWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        
    def forward(self, x):
        # Extract raw prediction logits from the Hugging Face structure
        output = self.hf_model(x)
        return output.logits

# 2. Load your custom fine-tuned model weights
model_path = "./final_pneumonia_cnn_model"
print(f"Loading custom fine-tuned model from {model_path}...")

hf_model = AutoModelForImageClassification.from_pretrained(model_path)
wrapped_model = HuggingFaceModelWrapper(hf_model)
wrapped_model.eval()

# 3. Select the target layer inside the ResNet backbone for Grad-CAM tracking
# In Hugging Face's ResNet architecture, this points to the final convolutional layer group
target_layers = [hf_model.resnet.encoder.stages[-1].layers[-1]]

# 4. Read and pre-process your local target X-Ray image
image_path = "xray.png"
rgb_img = Image.open(image_path).convert('RGB').resize((224, 224))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(rgb_img).unsqueeze(0)

# 5. Initialize the GradCAM engine with our wrapped model structure
cam = GradCAM(model=wrapped_model, target_layers=target_layers)

# 6. Compute the diagnostic activation heatmap
# target category: 0 = Normal, 1 = Pneumonia (Passing None defaults to highest prediction score)
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

# 7. Overlay the visual heatmap map back onto the base chest X-Ray graphics
img_array = np.float32(rgb_img) / 255.0
visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

# 8. Export the completed image locally
output_path = "./results/marked_abnormality_heatmap.png"
cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

print(f"\nSuccessfully generated clinical heatmap! Saved to: {output_path}")
