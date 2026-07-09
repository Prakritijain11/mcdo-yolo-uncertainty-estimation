from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import os
image = np.asarray(Image.open('img.jpg'))
#image = np.ones((427, 640, 3), dtype=np.uint8)

# corrupted_image = corrupt(img, corruption_name='gaussian_blur', severity=1)
output_dir = 'Result/Fig'
os.makedirs(output_dir, exist_ok=True)

# List to store corruption names and their total compute times
corruption_times = []

# Corruption loop
corruption_names = get_corruption_names('blur')
if not corruption_names:
    print("No corruption names found for 'blur'")
else:
    for corruption in corruption_names:
        total_time = 0
        print(f"Processing corruption: {corruption}")
        try:
            for severity in range(5):
                tic = time.time()
                corrupted = corrupt(image, corruption_name=corruption, severity=severity+1)
                # Save corrupted image
                plt.imsave(os.path.join(output_dir, f'{corruption}_severity_{severity+1}.png'), corrupted)
                total_time += time.time() - tic
            corruption_times.append((corruption, total_time))
            print(f"{corruption}: {total_time} seconds")
        except Exception as e:
            print(f"An error occurred while processing {corruption}: {e}")

if not corruption_times:
    print("No corruption times recorded.")
else:
    # Plot the compute times
    corruptions, times = zip(*corruption_times)
    plt.figure(figsize=(10, 5))
    plt.bar(corruptions, times, color='blue')
    plt.xlabel('Corruption Types')
    plt.ylabel('Total Compute Time (seconds)')
    plt.title('Total Compute Time for Each Corruption Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compute_times.png'))
    plt.show()