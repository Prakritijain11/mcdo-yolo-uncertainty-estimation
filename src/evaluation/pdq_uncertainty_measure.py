from ultralytics import YOLO
import os
import yaml
import numpy as np
import torch
from sklearn.metrics import jaccard_score

model = YOLO('mcdo-50.pt')  
model.model.train()
image_base_path = 'DATA_DIR/images/val2017'
label_base_path = 'DATA_DIR/labels/val2017/'


yaml_path = 'coco-mod.yaml'
output_file = "uncertainty_output_mcdo50.txt"

def update_yaml_file(corruption, severity):
    with open(yaml_path, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)
    data['val'] = f'images/val2017/{severity}'
    with open(yaml_path, 'w') as yamlfile:
        yaml.safe_dump(data, yamlfile)

def calculate_uncertainty(predictions):
    if not predictions or len(predictions[0][0].boxes) == 0:
        return 0.0
    
    boxes = [pred[0].boxes.xyxy.cpu().numpy() for pred in predictions]  
    confidences = [pred[0].boxes.conf.cpu().numpy() for pred in predictions]
    
    box_variance = np.var(boxes, axis=0) if len(boxes) > 1 else 0
    confidence_variance = np.var(confidences, axis=0) if len(confidences) > 1 else 0

    total_uncertainty = np.mean(box_variance) + np.mean(confidence_variance)
    return total_uncertainty


def calculate_pdq(ground_truth_boxes, predicted_boxes, predicted_confidences, box_variance):
    if len(predicted_boxes) == 0 or len(ground_truth_boxes) == 0:
        return 0.0
    
    iou = jaccard_score(ground_truth_boxes.flatten(), predicted_boxes.flatten(), average='macro')
    
    pdq = iou * np.exp(-np.mean(box_variance) / 2) / (1 + np.mean(predicted_confidences))
    return pdq

mcdo_samples = 30 

with open(output_file, mode='w') as file:

    for corruption_type in os.listdir(image_base_path):
        corruption_path = os.path.join(image_base_path, corruption_type)
        if os.path.isdir(corruption_path):
            for severity in os.listdir(corruption_path):
                severity_path = os.path.join(corruption_path, severity)
                if os.path.isdir(severity_path):
                    update_yaml_file(corruption_type, severity)
                    
                    total_bboxes = 0
                    all_confidences = []
                    all_uncertainties = []  
                    all_pdq_scores = []    

                    results = model.val(data=yaml_path, save_json=True)

                    for img_path in os.listdir(severity_path):
                        img_full_path = os.path.join(severity_path, img_path)
                        if img_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                            mcdo_predictions = [model(img_full_path) for _ in range(mcdo_samples)]
                            label_file = os.path.join(label_base_path, corruption_type, severity, img_path.replace('.jpg', '.txt'))
                            ground_truth_boxes = np.loadtxt(label_file, usecols=(1, 2, 3, 4)) if os.path.exists(label_file) else np.array([])

                            total_bboxes += len(mcdo_predictions[0][0].boxes)
                            
                            confidences = mcdo_predictions[0][0].boxes.conf.cpu().numpy()
                            all_confidences.extend(confidences)
                            
                            uncertainty = calculate_uncertainty(mcdo_predictions)
                            all_uncertainties.append(uncertainty)

                            predicted_boxes = mcdo_predictions[0][0].boxes.xyxy.cpu().numpy()
                            box_variance = np.var([pred[0].boxes.xyxy.cpu().numpy() for pred in mcdo_predictions], axis=0)
                            
                            pdq_score = calculate_pdq(ground_truth_boxes, predicted_boxes, confidences, box_variance)
                            all_pdq_scores.append(pdq_score)

                    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
                    avg_uncertainty = sum(all_uncertainties) / len(all_uncertainties) if all_uncertainties else 0
                    avg_pdq = sum(all_pdq_scores) / len(all_pdq_scores) if all_pdq_scores else 0
                    print(f"Intermediate Results for {corruption_type} with severity {severity}:")
                    print(f"Total bounding boxes detected: {total_bboxes}")
                    print(f"Average confidence score: {avg_confidence:.4f}")
                    print(f"Average uncertainty score: {avg_uncertainty:.4f}")
                    print(f"Average PDQ score: {avg_pdq:.4f}")
                    
                    file.write(f"Results for {corruption_type} with severity {severity}:\n")
                    file.write(f"Total bounding boxes detected: {total_bboxes}\n")
                    file.write(f"Average confidence score: {avg_confidence:.4f}\n")
                    file.write(f"Average uncertainty score: {avg_uncertainty:.4f}\n")
                    file.write(f"Average PDQ score: {avg_pdq:.4f}\n")
                    file.write(f"mAP@0.5:0.95: {results.box.map}\n")
                    file.write(f"mAP@0.5: {results.box.map50}\n")   
                    file.write(f"mAP@0.75: {results.box.map75}\n")     
                    file.write(f"Per-Class mAPs: {results.box.maps}\n")  
                    file.write("\n")

print(f"Validation results saved to {output_file}")
