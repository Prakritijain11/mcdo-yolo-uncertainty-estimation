
import numpy as np
from custom_yolov8 import CustomYOLO
from utils import load_image, visualize_results

class MCDOWrapper:
    def __init__(self, model, num_samples=30):
        self.model = model
        self.num_samples = num_samples

    def __call__(self, image):
        self.model.train()
        predictions = [self.model.predict(source=image, save=False) for _ in range(self.num_samples)]
        return predictions

def calculate_uncertainty(predictions):
    boxes = [pred.boxes.xyxy.cpu().numpy() for pred in predictions]

    box_variance = np.var(boxes, axis=0)
    
    return box_variance

def mcdo_inference(model, image_path, num_samples=30):
    image = load_image(image_path)
    mcdo_model = MCDOWrapper(model, num_samples)
    predictions = mcdo_model(image)
    uncertainty = calculate_uncertainty(predictions)
    return predictions, uncertainty

if __name__ == "__main__":
    model = CustomYOLO('yolov8n.pt', dropout_rate=0.7) 

    image_path = '/csghome/vy305/Project/Train/YOLOv8/datasets/coco/images/val2017/000000206218.jpg'

    predictions, uncertainty = mcdo_inference(model, image_path, num_samples=30)

    visualize_results(image_path, predictions, uncertainty)
