from ultralytics import YOLO
import os
import yaml

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')  # Specify the correct path to your model's weights

# Define the base path to images and labels
image_base_path = 'DATA_DIR/images/val2017'
label_base_path = 'DATA_DIR/labels/val2017'

# Path to base YAML file
yaml_path = 'coco-mod.yaml'

# Path to save terminal output
output_file = "validation_terminal_output.txt"

# Open output file for writing
with open(output_file, mode='w') as file:

    # Function to update YAML file
    def update_yaml_file(corruption, severity):
        with open(yaml_path, 'r') as yamlfile:
            data = yaml.safe_load(yamlfile)
        
        # Update the 'val' path dynamically based on corruption type and severity level
        data['val'] = f'images/val2017/{corruption}/{severity}'
        
        # Write the updated data back to the YAML file
        with open(yaml_path, 'w') as yamlfile:
            yaml.safe_dump(data, yamlfile)

    # Iterate over each corruption type and severity level
    for corruption_type in os.listdir(image_base_path):
        corruption_path = os.path.join(image_base_path, corruption_type)
        if os.path.isdir(corruption_path):
            for severity in os.listdir(corruption_path):
                severity_path = os.path.join(corruption_path, severity)
                if os.path.isdir(severity_path):
                    # Update the YAML file for the current corruption and severity
                    update_yaml_file(corruption_type, severity)
                                   
                    # Initialize bbox count and confidence scores list
                    total_bboxes = 0
                    all_confidences = []

                    # Validate the model and get predictions
                    results = model.val(data=yaml_path, save_json=True)  # Pass the path to the YAML file

                    # Run inference on each image in the validation set
                    for img_path in os.listdir(severity_path):
                        img_full_path = os.path.join(severity_path, img_path)
                        if img_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                            # Perform inference
                            prediction = model(img_full_path)
                            # Count the number of bounding boxes in the image
                            total_bboxes += len(prediction[0].boxes)
                            
                            # Collect confidence scores for each bounding box
                            confidences = prediction[0].boxes.conf.cpu().numpy()
                            all_confidences.extend(confidences)
                    # Print intermediate results for the current severity level
                    print(f"Intermediate Results for {corruption_type} with severity {severity}:")
                    print(f"Total bounding boxes detected: {total_bboxes}")
                    print(f"Average confidence score: {sum(all_confidences) / len(all_confidences) if all_confidences else 0:.4f}")
                      
                    # Write results to text file
                    file.write(f"Results for {corruption_type} with severity {severity}:\n")
                    file.write(f"Total bounding boxes detected: {total_bboxes}\n")
                    file.write(f"Average confidence score: {sum(all_confidences) / len(all_confidences) if all_confidences else 0:.4f}\n")
                    file.write(f"mAP@0.5:0.95: {results.box.map}\n")   # mAP@0.5:0.95
                    file.write(f"mAP@0.5: {results.box.map50}\n")      # mAP@0.5
                    file.write(f"mAP@0.75: {results.box.map75}\n")     # mAP@0.75
                    file.write(f"Per-Class mAPs: {results.box.maps}\n")  # mAP per class
                    file.write("\n")

print(f"Validation results saved to {output_file}")