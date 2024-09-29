import os
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the YOLOv8 model
yolo_model_path = 'F:/src/runs/classify/train8/weights/best.pt' # train7 -> ct, train8 -> pet
model = YOLO(yolo_model_path)

# Function to preprocess an image object
def preprocess_image(image, size=(640, 640)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Function for Feature Ablation
def feature_ablation(model, image, mask_size=60):
    original_tensor = preprocess_image(image)
    num_rows = (image.height + mask_size - 1) // mask_size
    num_cols = (image.width + mask_size - 1) // mask_size
    heatmap = np.zeros((num_rows, num_cols))

    # Perform inference on the original image
    with torch.no_grad():
        original_pred = model(original_tensor)
    original_confidence = original_pred[0].probs.top1conf.item()

    for i in range(0, image.width, mask_size):
        for j in range(0, image.height, mask_size):
            ablated_image = image.copy()
            draw = ImageDraw.Draw(ablated_image)
            draw.rectangle([i, j, i + mask_size, j + mask_size], fill="black")
            
            ablated_tensor = preprocess_image(ablated_image)
            with torch.no_grad():
                ablated_pred = model(ablated_tensor)
            ablated_confidence = ablated_pred[0].probs.top1conf.item()

            diff = np.abs(original_confidence - ablated_confidence)
            heatmap[j // mask_size, i // mask_size] = diff

    return heatmap

# Function to overlay the heatmap on the image
def overlay_heatmap(image, heatmap, colormap=plt.cm.autumn):
    heatmap_resized = np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 1))
    heatmap_resized = colormap(heatmap_resized)
    heatmap_resized = Image.fromarray((heatmap_resized[:, :, :3] * 255).astype(np.uint8))
    heatmap_resized = heatmap_resized.resize(image.size, Image.LANCZOS)
    overlayed_image = Image.blend(image, heatmap_resized, alpha=0.5)
    return overlayed_image

# Function to process an image and save XAI result
def process_image(image_path, model, save_dir, mask_size=60):
    original_image = Image.open(image_path).convert('RGB')
    heatmap = feature_ablation(model, original_image, mask_size)
    overlayed_image = overlay_heatmap(original_image, heatmap, colormap=plt.cm.BuPu)

    # Save the XAI result
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, image_name.replace('.png', '_xai.png'))
    overlayed_image.save(save_path)

# Directory containing images
input_dir = 'F:/nsclc/Test3.v1/shap/'
# Directory to save XAI results
output_dir = 'F:/nsclc/Test3.v1/shap_xai/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each image in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
for image_file in tqdm(image_files, desc="Processing Images"):
    image_path = os.path.join(input_dir, image_file)
    process_image(image_path, model, output_dir)
