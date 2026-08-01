print("Downloading dataset from Hugging Face...")
dataset = load_dataset("hf-vision/chest-xray-pneumonia")

sample_image = dataset["test"][0]["image"]

# print("Saved sample chest image successfully as 'xray.png'.")
from transformers import pipeline

clf = pipeline("image-classification")

predictions = clf("./results/xray.png")
# print(predictions)