from ultralytics import YOLO


################ For Images#######################
# Define path to the image file
source = "pedestrians-crosswalk.jpg"

# Load a pretrained YOLOv8n model
model = YOLO("best.pt")

# Run inference on the source
results = model(source)  # list of Results objects

##################################################







############ For video###########################

# Define path to video file
#source = "test-video.mp4"

# Load a pretrained YOLOv8n model
#model = YOLO('yolov8n.pt')

# Run inference on the source
#results = model(source, stream=True)  # generator of Results objects