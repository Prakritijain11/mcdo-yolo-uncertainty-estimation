from PIL import Image
import numpy as np
import os
import time
from imagecorruptions import corrupt, get_corruption_names

def apply_corruption(input_image_path, output_base_folder, corruption_list):
    # Read the image and convert it to a NumPy array
    image = Image.open(input_image_path)
    image_array = np.array(image)
    image_name = os.path.basename(input_image_path)

    # Apply each corruption in the corruption list with varying severity levels
    for corruption_name in corruption_list:
        corruption_times = []
        for severity in range(1, 6):  # Severity levels from 1 to 5
            start_time = time.time()
            corrupted_image_array = corrupt(image_array, corruption_name=corruption_name, severity=severity)
            end_time = time.time()
            
            # Save corrupted images in the appropriate folder structure
            corruption_folder = os.path.join(output_base_folder, corruption_name)
            severity_folder = os.path.join(corruption_folder, f"severity_{severity}")
            os.makedirs(severity_folder, exist_ok=True)
            
            output_image_path = os.path.join(severity_folder, image_name)
            corrupted_image = Image.fromarray(corrupted_image_array)
            corrupted_image.save(output_image_path)
            
            corruption_times.append(end_time - start_time)

        print(f"Corruption '{corruption_name}' applied with severity 1-5 took {corruption_times} seconds each for {image_name}.")

def apply_corruption_to_folder(input_folder, output_base_folder):
    corruption_list = get_corruption_names('all')
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            apply_corruption(input_image_path, output_base_folder, corruption_list)

if __name__ == '__main__':
    input_folder = "DATA_DIR/val2017/"
    output_base_folder = "/csghome/vy305/Project/Train/Images/"
    apply_corruption_to_folder(input_folder, output_base_folder)
