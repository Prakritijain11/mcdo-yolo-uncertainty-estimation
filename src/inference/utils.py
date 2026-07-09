# utils.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found at {image_path}. Please check the path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_uncertainty(uncertainty, save_path='uncertainty.json'):
    """Save uncertainty data to a JSON file."""
    with open(save_path, 'w') as f:
        json.dump(uncertainty, f, indent=4)
    print(f"Uncertainty data saved to {save_path}")

def visualize_results(image_path, predictions, uncertainty):
    img = load_image(image_path)

    # Draw bounding boxes and labels on the image
    for i, prediction in enumerate(predictions[0].boxes):
        xyxy = prediction.xyxy.cpu().numpy().astype(int)
        confidence = prediction.conf.cpu().numpy()
        cls = prediction.cls.cpu().numpy()
        
        # Draw the bounding box
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        
        # Add class and confidence label
        label = f'Class: {cls}, Conf: {confidence:.2f}'
        cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 2)

    print(f"Bounding box variance (uncertainty): {uncertainty}")
    
    # Save the image with bounding boxes
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('mcdo.jpg', bbox_inches='tight')
    plt.close()

    print("Image saved as 'mcdo.jpg'.")
    save_uncertainty(uncertainty, save_path='uncertainty.json')
