from ultralytics import YOLO
import os
import yaml

# Load a pretrained YOLO model
model = YOLO('mcdo-50.pt')  # Specify the correct path to your model's weights

# Define the base path to images and labels
image_base_path = 'DATA_DIR/images/val2017'
label_base_path = 'DATA_DIR/labels/val2017'

# Path to base YAML file
yaml_path = 'coco-mod.yaml'

# Path to save terminal output
output_file = "validation_terminal_output_snow.txt"

# Open output file for writing
with open(output_file, mode='w') as file:

    # Function to update YAML file
    def update_yaml_file(severity):
        with open(yaml_path, 'r') as yamlfile:
            data = yaml.safe_load(yamlfile)
        
        # Update the 'val' path dynamically based on severity level
        data['val'] = f'images/val2017/{severity}/'
        
        # Write the updated data back to the YAML file
        with open(yaml_path, 'w') as yamlfile:
            yaml.safe_dump(data, yamlfile)
    
    print("Starting validation process...")  # Debugging log

    # Iterate over each severity level inside the validation folder
    for severity in os.listdir(image_base_path):
        severity_path = os.path.join(image_base_path, severity)
        print(f"Processing severity level: {severity}")  # Debugging log

        if os.path.isdir(severity_path):
            # Update the YAML file for the current severity
            update_yaml_file(severity)
                           
            # Initialize bbox count and confidence scores list
            total_bboxes = 0
            all_confidences = []

            # Validate the model and get predictions
            results = model.val(data=yaml_path, save_json=True)  # Pass the path to the YAML file
            print(f"  Results for severity {severity}: {results}")  # Debugging log

            # Run inference on each image in the severity level folder
            for img_path in os.listdir(severity_path):
                img_full_path = os.path.join(severity_path, img_path)
                if img_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    # Perform inference
                    prediction = model(img_full_path)
                    if prediction:
                        print(f"    Inference on image {img_full_path} returned predictions.")  # Debugging log
                    else:
                        print(f"    No predictions for image {img_full_path}.")  # Debugging log

                    # Count the number of bounding boxes in the image
                    total_bboxes += len(prediction[0].boxes)
                    
                    # Collect confidence scores for each bounding box
                    confidences = prediction[0].boxes.conf.cpu().numpy()
                    all_confidences.extend(confidences)

            # Print intermediate results for the current severity level
            print(f"Intermediate Results for severity {severity}:")
            print(f"Total bounding boxes detected: {total_bboxes}")
            print(f"Average confidence score: {sum(all_confidences) / len(all_confidences) if all_confidences else 0:.4f}")
                  
            # Write results to text file
            file.write(f"Results for severity {severity}:\n")
            file.write(f"Total bounding boxes detected: {total_bboxes}\n")
            file.write(f"Average confidence score: {sum(all_confidences) / len(all_confidences) if all_confidences else 0:.4f}\n")
            file.write(f"mAP@0.5:0.95: {results.box.map}\n")   # mAP@0.5:0.95
            file.write(f"mAP@0.5: {results.box.map50}\n")      # mAP@0.5
            file.write(f"mAP@0.75: {results.box.map75}\n")     # mAP@0.75
            file.write(f"Per-Class mAPs: {results.box.maps}\n")  # mAP per class
            file.write("\n")
            file.flush()  # Ensure data is written to file

print(f"Validation results saved to {output_file}")
