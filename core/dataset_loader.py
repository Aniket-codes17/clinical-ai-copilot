from datasets import load_dataset
import numpy as np
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import torch

print("Downloading dataset from Hugging Face...")
dataset = load_dataset("hf-vision/chest-xray-pneumonia")

sample_image = dataset["test"][0]["image"]
sample_image.save("./results/xray1.png")

# print("Saved sample chest image successfully as 'xray.png'.")


# clf = pipeline("image-classification")

# predictions = clf("xray1.png")
# # print(predictions)



# model = models.resnet50(pretrained=True)
# model.eval()

# target_layers = [model.layer4[-1]]

# image_path = "xray.png"
# rgb_img = Image.open(image_path).convert('RGB').resize((224, 224))
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# input_tensor = transform(rgb_img).unsqueeze(0)


# cam = GradCAM(model=model, target_layers=target_layers)


# grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]


# img_array = np.float32(rgb_img) / 255.0
# visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

# cv2.imwrite("grad_cam_output.png", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
# # print("Saved Grad-CAM heatmap visualization to 'grad_cam_output.png'.")




# dataset = load_dataset("hf-vision/chest-xray-pneumonia")
# processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# def transform_images(examples):
#     inputs = processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
#     inputs["labels"] = examples["label"]
#     return inputs


# prepared_ds = dataset.with_transform(transform_images)


# model = AutoModelForImageClassification.from_pretrained(
#     "microsoft/resnet-50", 
#     num_labels=2, 
#     ignore_mismatched_sizes=True
# )


# training_args = TrainingArguments(
#     output_dir="./medical_model_checkpoints",
#     per_device_train_batch_size=8,
#     eval_strategy="epoch",        # <-- FIXED: changed from evaluation_strategy
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     num_train_epochs=2,
#     logging_steps=50,
#     remove_unused_columns=False
# )


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=prepared_ds["train"],
#     eval_dataset=prepared_ds["validation"]
# )


# print("Starting training process...")
# trainer.train()


# model.save_pretrained("./final_pneumonia_cnn_model")
# processor.save_pretrained("./final_pneumonia_cnn_model")
# print("Fine-tuning completed. Custom model saved to './final_pneumonia_cnn_model'.")
