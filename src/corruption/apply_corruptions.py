import os
import time
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from torchvision import transforms
import torchvision.datasets as dset
import torchvision.transforms as trn

# Define the base directory and the specific subset (data type)
dataDir = 'DATA_DIR'
dataType = 'val2017'
annFile = os.path.join(dataDir, 'annotations', f'instances_{dataType}.json')

# Create a CocoDetection object to load the dataset
coco_data = dset.CocoDetection(root=os.path.join(dataDir, dataType), annFile=annFile, transform=trn.ToTensor())

# Define the output path for the corrupted images
output_path = 'corrupted_images111'

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Define corruption types
corruption_types = get_corruption_names('all')

# Initialize a dictionary to store execution times
execution_times = {corruption: [] for corruption in corruption_types}

total_images = len(coco_data)
start_time = time.time()

# Iterate over each image in the COCO dataset
for idx, (img, target) in enumerate(coco_data):
    image = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # Convert tensor image to numpy array
    
    # Iterate over each corruption type
    for corruption in corruption_types:
        # Create a directory for each corruption type
        corruption_dir = os.path.join(output_path, corruption)
        if not os.path.exists(corruption_dir):
            os.makedirs(corruption_dir)
        
        for severity in range(1, 6):
            # Corrupt the image
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            tic = time.time()
            corrupted = corrupt(image, corruption_name=corruption, severity=severity)
            toc = time.time()
            
            # Save the corrupted image
            severity_dir = os.path.join(corruption_dir, f'severity_{severity}')
            if not os.path.exists(severity_dir):
                os.makedirs(severity_dir)
            corrupted_image_path = os.path.join(severity_dir, f'image_{idx:05d}.png')
            Image.fromarray(corrupted).save(corrupted_image_path)
            
            # Record execution time
            execution_times[corruption].append(toc - tic)
    
    # Print progress and estimated time remaining
    elapsed_time = time.time() - start_time
    avg_time_per_image = elapsed_time / (idx + 1)
    remaining_images = total_images - (idx + 1)
    estimated_remaining_time = remaining_images * avg_time_per_image
    print(f'Processed {idx + 1}/{total_images} images. Estimated remaining time: {estimated_remaining_time / 60:.2f} minutes.')

# Plot the execution times for each corruption and severity level
plt.figure(figsize=(14, 10))

# Define a list of line styles and colors to use for the plot
line_styles = ['-', '--', '-.', ':']
colors = plt.cm.tab20(np.linspace(0, 1, len(corruption_types)))

# Plot each corruption's execution times
for i, (corruption, times) in enumerate(execution_times.items()):
    avg_times = [sum(times[j::5]) / len(times[j::5]) for j in range(5)]
    plt.plot(range(1, 6), avg_times, marker='o', linestyle=line_styles[i % len(line_styles)], color=colors[i], label=corruption)

# Add labels, title, legend, and grid
plt.xlabel('Severity Level')
plt.ylabel('Average Execution Time (s)')
plt.title('Average Execution Time for Each Severity Level of Corruptions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(min(min(times) for times in execution_times.values()) * 0.9, max(max(times) for times in execution_times.values()) * 1.1)
plt.yscale('log')
plt.grid(True)
plt.xticks(range(1, 6))

# Save and show the plot
plt.tight_layout()
plt.savefig('execution_times_plot_COCO.png')
plt.close()
