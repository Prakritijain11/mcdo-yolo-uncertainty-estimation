from ultralytics import YOLO
import os

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')  # Specify the correct path to your model's weights

# Path to save the terminal output
output_file = "validation_zero_severity_new.txt"

# Directory containing images
image_dir = 'DATA_DIR/images/val2017/'  # Replace this with the path to your images directory

# Initialize variables for cumulative calculations
total_confidence = 0
total_bounding_boxes = 0  # New variable to track the number of bounding boxes
image_count = 0

# Open the output file in write mode
with open(output_file, 'w') as file:
    # Run the validation and save the metrics
    metrics = model.val(save_json=True)  # Assuming the dataset path is configured in the model or in a YAML file

    # Write the general metrics to the file
    file.write(f"Validation Results:\n")
    file.write(f"mAP@0.5:0.95: {metrics.box.map}\n")  # mAP@0.5:0.95
    file.write(f"mAP@0.5: {metrics.box.map50}\n")  # mAP@0.5
    file.write(f"mAP@0.75: {metrics.box.map75}\n")  # mAP@0.75
    file.write(f"Per-Class mAPs: {metrics.box.maps}\n")  # mAP per class
    file.write("\n")  # Add a newline for readability

    # Perform inference on multiple images to get predictions
    results = model.predict(image_dir, save=True, save_txt=True)  # Predict on all images in the directory

    # Log confidence scores and bounding boxes for each result (per image)
    for result in results:
        image_count += 1
        file.write(f"Image: {result.path}\n")
        for box in result.boxes:
            # Write Class ID, Confidence Score, Bounding Box
            file.write(f"Class: {box.cls}\n")
            file.write(f"Confidence: {box.conf.item():.2f}\n")  # Convert tensor to scalar with .item()
            file.write(f"Bounding Box: {box.xyxy.tolist()}\n")
            
            # Accumulate confidence scores and count bounding boxes
            total_confidence += box.conf.item()  # Convert tensor to scalar
            total_bounding_boxes += 1  # Increment total bounding boxes for each detection
        
        file.write("\n")  # Newline for better formatting between images

    # After processing all images, compute cumulative statistics
    avg_confidence = total_confidence / total_bounding_boxes if total_bounding_boxes > 0 else 0

    # Write cumulative data to the file
    file.write("Cumulative Report:\n")
    file.write(f"Total images processed: {image_count}\n")
    file.write(f"Total bounding boxes: {total_bounding_boxes}\n")  # Output total bounding boxes
    file.write(f"Average confidence score: {avg_confidence:.2f}\n")  # Output average confidence score
    file.write(f"mAP@0.5:0.95 (Cumulative): {metrics.box.map}\n")
    file.write(f"mAP@0.5 (Cumulative): {metrics.box.map50}\n")
    file.write(f"mAP@0.75 (Cumulative): {metrics.box.map75}\n")

print(f"Validation results with cumulative report saved to {output_file}")
