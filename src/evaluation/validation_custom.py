from ultralytics import YOLO
import os
import yaml

model = YOLO('yolov8n.pt')

# Define the base path to images and labels
image_base_path = 'DATA_DIR/images/val2017'
label_base_path = 'DATA_DIR/labels/val2017'

# Path to base YAML file
yaml_path = 'coco-mod.yaml'
output_file = "validation_terminal_output.txt"
with open(output_file, mode='w') as file:

    # Function to update YAML file
    def update_yaml_file(corruption, severity):
        with open(yaml_path, 'r') as yamlfile:
            data = yaml.safe_load(yamlfile)
        data['val'] = f'images/val2017/{corruption}/{severity}'
        with open(yaml_path, 'w') as yamlfile:
            yaml.safe_dump(data, yamlfile)

    # Iterate over each corruption type and severity level
    for corruption_type in os.listdir(image_base_path):
        corruption_path = os.path.join(image_base_path, corruption_type)
        if os.path.isdir(corruption_path):
            for severity in os.listdir(corruption_path):
                severity_path = os.path.join(corruption_path, severity)
                if os.path.isdir(severity_path):
                    update_yaml_file(corruption_type, severity)
                                   
                    total_bboxes = 0
                    results = model.val(data=yaml_path, save_json=True)

                    # Run inference on each image in the validation set
                    for img_path in os.listdir(severity_path):
                        img_full_path = os.path.join(severity_path, img_path)
                        if img_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                            prediction = model(img_full_path)
                            total_bboxes += len(prediction[0].boxes)
                            
#                    print(f"Intermediate Results for {corruption_type} with severity {severity}:")
#                    print(f"Total bounding boxes detected: {total_bboxes}")
                      
                    # Write results to text file
                    file.write(f"Results for {corruption_type} with severity {severity}:\n")
                    file.write(f"Total bounding boxes detected: {total_bboxes}\n")
                    file.write(f"mAP@0.5:0.95: {results.box.map}\n")
                    file.write(f"mAP@0.5: {results.box.map50}\n")
                    file.write(f"mAP@0.75: {results.box.map75}\n")
                    file.write(f"Per-Class mAPs: {results.box.maps}\n")
                    file.write("\n")

print(f"Validation results saved to {output_file}")