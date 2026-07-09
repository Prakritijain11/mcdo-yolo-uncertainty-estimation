import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torchvision import transforms
import torch
from imagecorruptions import get_corruption_names
from glob import glob

# Define the base directory and the specific subset (data type)
coco_data_dir = '/csghome/vy305/Project/Train/imagecorruptions-master/corrupted_testing/'

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define a transform to resize images to the expected input size of the YOLO model
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

# Define corruption types and severities, including severity level 0 for uncorrupted images
corruption_types = get_corruption_names('all')
severities = range(0, 6)  # Including severity level 0

# Initialize a dictionary to store confidence scores
confidence_scores = {corruption: {severity: [] for severity in severities} for corruption in corruption_types}
confidence_scores['original'] = {0: []}  # For uncorrupted images

# List to store all confidence scores for plotting
all_confidences = []

# Iterate over each corruption type and severity level, including uncorrupted images
for corruption in corruption_types:
    for severity in severities:
        if severity == 0:
            # Process uncorrupted images
            base_path = f"{coco_data_dir}original/"
            image_files = glob(base_path + "*.png")
        else:
            base_path = f"{coco_data_dir}{corruption}/severity_{severity}/"
            image_files = glob(base_path + "*.png")
      
        # Iterate over each image file in the directory
        for idx, image_file in enumerate(image_files):
                       
            # Perform object detection with YOLOv5
            results = model(image_file)
            
            # Access the detections
            detections = results.xyxy[0]

            # Process YOLO detections
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection.tolist()  # Use tolist() to convert to Python list
                all_confidences.append(conf)
                if severity == 0:
                    confidence_scores['original'][severity].append(conf)
                else:
                    confidence_scores[corruption][severity].append(conf)
            
            print(f'Processed {idx + 1}/{len(image_files)} images for {corruption} severity {severity}.')

# Generate a matrix of average confidence scores for each corruption and severity level
avg_confidence_scores = {corruption: [] for corruption in corruption_types}
avg_confidence_scores['original'] = []

for corruption in corruption_types:
    for severity in severities:
        avg_conf = np.mean(confidence_scores[corruption][severity]) if confidence_scores[corruption][severity] else 0
        avg_confidence_scores[corruption].append(avg_conf)

# Calculate average confidence score for original (uncorrupted) images
avg_conf = np.mean(confidence_scores['original'][0]) if confidence_scores['original'][0] else 0
avg_confidence_scores['original'].append(avg_conf)

# Sort corruption types based on their overall average confidence score
corruption_types_sorted = sorted(corruption_types + ['original'], key=lambda x: np.mean(avg_confidence_scores[x]), reverse=True)

# Define the colormap for the plot
cmap = plt.get_cmap('tab20')

# Enhanced plotting for a classier look
plt.figure(figsize=(14, 8))

# Use an available style
plt.style.use('ggplot')

# Plot each corruption type's confidence score across severity levels with unique colors
for i, corruption in enumerate(corruption_types_sorted):
    scores = avg_confidence_scores[corruption]  # No sorting
    color = cmap(i % 20)
    plt.plot(severities, scores, marker='o', color=color, label=corruption, linewidth=2, markersize=6)

# Add plot details with classy fonts and styles
plt.xlabel('Severity Level', fontsize=14, fontweight='bold')
plt.ylabel('Average Confidence Score', fontsize=14, fontweight='bold')
plt.title('Average Confidence Score vs Severity for Different Corruptions', fontsize=16, fontweight='bold')
plt.xticks(severities, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Corruption Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Adding a custom colormap background
plt.gca().set_facecolor('#f0f0f0')

# Save the plot with higher DPI for quality
plt.tight_layout()
plt.savefig('confidence_score_curves_with_severity_0.png', dpi=300)
plt.show()
