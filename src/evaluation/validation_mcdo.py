from ultralytics import YOLO
import os
import yaml
import numpy as np
import torch

# Load a pretrained YOLO model with MCDO
model = YOLO('mcdo-50.pt')  
model.model.train()  # Enable dropout for MCDO

# Path to images and labels
image_base_path = 'DATA_DIR/images/val2017'
label_base_path = 'DATA_DIR/labels/val2017/'

# YAML file
yaml_path = 'coco-mod.yaml'

# Output file to save results
output_file = "uncertainty_output_mcdo50.txt"

# Function to update YAML file for specific severity level
def update_yaml_file(severity):
    with open(yaml_path, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)
    # Update the 'val' path dynamically based on severity level
    data['val'] = f'images/val2017/{severity}'
    with open(yaml_path, 'w') as yamlfile:
        yaml.safe_dump(data, yamlfile)
    print(f"Updated YAML file for severity: {severity}")

# Function to calculate epistemic uncertainty from MCDO predictions
def calculate_epistemic_uncertainty(predictions):
    if not predictions or len(predictions[0][0].boxes) == 0:
        return 0.0
    
    # Collect confidence scores for all predictions
    confidences = [pred[0].boxes.conf.cpu().numpy() for pred in predictions]
    confidences = np.array(confidences)
    
    # Calculate the mean of the confidences
    mean_conf = np.mean(confidences, axis=0)
    
    # Calculate the entropy of the mean prediction (average over predictions)
    entropy_mean = -np.sum(mean_conf * np.log(mean_conf + 1e-10), axis=-1)
    
    # Calculate the entropy for each individual prediction
    entropy_individual = -np.sum(confidences * np.log(confidences + 1e-10), axis=-1)
    
    # Calculate mutual information (epistemic uncertainty)
    epistemic_uncertainty = entropy_mean - np.mean(entropy_individual)
    return epistemic_uncertainty

# Define MCDO (number of forward passes)
mcdo_samples = 30

# Open output file for writing
with open(output_file, mode='w') as file:

    # Iterate over each severity level in the validation folder
    for severity in os.listdir(image_base_path):
        severity_path = os.path.join(image_base_path, severity)
        print(f"Processing severity: {severity}")
        
        if os.path.isdir(severity_path):
            # Update YAML for the current severity level
            update_yaml_file(severity)
            
            total_bboxes = 0
            all_confidences = []
            all_epistemic_uncertainties = []

            # Validate the model and get predictions
            try:
                results = model.val(data=yaml_path, save_json=True)
                print(f"  Validation results for severity {severity}: mAP@0.5: {results.box.map50}")
            except Exception as e:
                print(f"  Error during model validation: {e}")
                continue

            # Run inference on each image in the severity set
            for img_path in os.listdir(severity_path):
                img_full_path = os.path.join(severity_path, img_path)
                if img_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    print(f"    Inference on image: {img_full_path}")
                    try:
                        # Perform MCDO inference (multiple forward passes)
                        mcdo_predictions = [model(img_full_path) for _ in range(mcdo_samples)]
                        
                        if len(mcdo_predictions[0][0].boxes) == 0:
                            print(f"    No bounding boxes detected in image: {img_full_path}")
                            continue
                        
                        # Count bounding boxes
                        total_bboxes += len(mcdo_predictions[0][0].boxes)
                        
                        # Collect confidence scores for each bounding box from the first pass
                        confidences = mcdo_predictions[0][0].boxes.conf.cpu().numpy()
                        all_confidences.extend(confidences)
                        
                        # Calculate epistemic uncertainty
                        epistemic_uncertainty = calculate_epistemic_uncertainty(mcdo_predictions)
                        all_epistemic_uncertainties.append(epistemic_uncertainty)
                    except Exception as e:
                        print(f"    Error during MCDO inference on image {img_full_path}: {e}")

            # Print intermediate results for the current severity level
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            avg_epistemic_uncertainty = sum(all_epistemic_uncertainties) / len(all_epistemic_uncertainties) if all_epistemic_uncertainties else 0
            print(f"  Intermediate Results for severity {severity}:")
            print(f"    Total bounding boxes detected: {total_bboxes}")
            print(f"    Average confidence score: {avg_confidence:.4f}")
            print(f"    Average epistemic uncertainty: {avg_epistemic_uncertainty:.4f}")
            
            # Write results to text file
            file.write(f"Results for severity {severity}:\n")
            file.write(f"Total bounding boxes detected: {total_bboxes}\n")
            file.write(f"Average confidence score: {avg_confidence:.4f}\n")
            file.write(f"Average epistemic uncertainty: {avg_epistemic_uncertainty:.4f}\n")
            file.write(f"mAP@0.5:0.95: {results.box.map}\n")
            file.write(f"mAP@0.5: {results.box.map50}\n")
            file.write(f"mAP@0.75: {results.box.map75}\n")
            file.write(f"Per-Class mAPs: {results.box.maps}\n")
            file.write("\n")
            file.flush()  # Ensure data is written to file

print(f"Validation results saved to {output_file}")
