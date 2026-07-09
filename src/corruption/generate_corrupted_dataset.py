import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from imagecorruptions import get_corruption_names
from torchvision import transforms
from pycocotools.coco import COCO
import yaml
import re 

# Load the YAML configuration file
yaml_file = '/csghome/vy305/Project/Train/imagecorruptions-master/coco.yaml'
with open(yaml_file, 'r') as file:
    yaml_config = yaml.safe_load(file)

# Extract the number of classes
nc = len(yaml_config['names']) # This assumes 'nc' is the key for the number of classes in your YAML file

# Initialize COCO API and YOLOv5 model
coco_data_dir = '/csghome/vy305/Project/Train/imagecorruptions-master/corrupted_images/'
coco_ann_file = 'DATA_DIR/annotations/instances_val2017.json'
coco = COCO(coco_ann_file)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define a transform to resize images to the expected input size of the YOLO model
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

# Define corruption types and severities
corruption_types = get_corruption_names('all')
severities = range(1, 6)

# Initialize a dictionary to store confidence scores and detections
confidence_scores = {corruption: {severity: [] for severity in severities} for corruption in corruption_types}
detections_data = {corruption: {severity: [] for severity in severities} for corruption in corruption_types}

# Function to load ground truth annotations for an image
def load_ground_truth(image_file):
    match = re.search(r'(\d+)', image_file)
    if match:
        image_id = int(match.group(1))
    else:
        raise ValueError(f"Cannot extract image ID from {image_file}")
    
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)

    ground_truth_boxes = []
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        ground_truth_boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], category_id])
    
    return ground_truth_boxes

# Iterate over each corruption type and severity level
for corruption in corruption_types:
    for severity in severities:
        image_files = [f"{coco_data_dir}{corruption}_severity_{severity}.png"]
#         image_files = [f for f in os.listdir(severity_dir) if os.path.isfile(os.path.join(severity_dir, f))]
        for idx, image_file in enumerate(image_files):
            results = model(image_file)
            detections = results.xyxy[0]

            detections_data[corruption][severity].append((image_file, detections))
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection.tolist()
                confidence_scores[corruption][severity].append(conf)
            
            print(f'Processed {idx + 1}/{len(image_files)} images for {corruption} severity {severity}.')

# Generate a matrix of average confidence scores for each corruption and severity level
avg_confidence_scores = {corruption: [] for corruption in corruption_types}
for corruption in corruption_types:
    for severity in severities:
        avg_conf = np.mean(confidence_scores[corruption][severity]) if confidence_scores[corruption][severity] else 0
        avg_confidence_scores[corruption].append(avg_conf)
        
# Function to compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)
    
    if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
        inter_area = 0.0
    else:
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area
    return iou

# Function to calculate Average Precision (AP) from precision-recall curve
def calculate_ap(recalls, precisions):
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

# Function to compute mAP
def compute_map(detections_data):
    all_aps = []
    
    for corruption in corruption_types:
        for severity in severities:
            ground_truths = []
            detections = []

            for i, (image_file, pred_boxes) in enumerate(detections_data[corruption][severity]):
                gt_boxes = load_ground_truth(image_file)
                ground_truths.append(gt_boxes)
                detections.append(pred_boxes)

            aps = []
            for class_id in range(nc):
                class_gt = [gt for gts in ground_truths for gt in gts if gt[-1] == class_id]
                class_detections = [det for dets in detections for det in dets if det[-1] == class_id]

                if len(class_gt) == 0 and len(class_detections) == 0:
                    continue

                class_detections.sort(key=lambda x: x[4], reverse=True)
                tp = np.zeros(len(class_detections))
                fp = np.zeros(len(class_detections))
                gt_detected = np.zeros(len(class_gt))

                for i, det in enumerate(class_detections):
                    iou_max = 0.0
                    for j, gt in enumerate(class_gt):
                        iou = compute_iou(det, gt)
                        if iou > iou_max:
                            iou_max = iou
                            best_gt_idx = j
                    
                    if iou_max > 0.5 and not gt_detected[best_gt_idx]:
                        tp[i] = 1
                        gt_detected[best_gt_idx] = 1
                    else:
                        fp[i] = 1

                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                recalls = tp_cumsum / len(class_gt)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
                ap = calculate_ap(recalls, precisions)
                aps.append(ap)

            mean_ap = np.mean(aps)
            all_aps.append(mean_ap)
            print(f'mAP for {corruption} severity {severity}: {mean_ap:.4f}')

    overall_map = np.mean(all_aps)
    return overall_map

overall_map = compute_map(detections_data)
print(f'Overall mAP: {overall_map:.4f}')