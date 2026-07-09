print("--- Booted up Python")

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

print("--- Started training")

# Train the model
results = model.train(data='coco.yaml', epochs=100, imgsz=640)
