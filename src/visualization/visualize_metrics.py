import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

corruption_types = []
severities = []
map_50_95 = []

# Read the output file
with open("validation_terminal_output.txt", "r") as file:
    for line in file:
        if "Results for" in line:
            parts = line.split()
            corruption_types.append(parts[2])
            severities.append(parts[5].strip(":"))
        elif "mAP@0.5:0.95:" in line:
            map_50_95.append(float(line.split(": ")[1]))

# Group the data by corruption type and calculate the average mAP for each group
corruption_map = defaultdict(list)
for c_type, mAP in zip(corruption_types, map_50_95):
    corruption_map[c_type].append(mAP)
average_map = {c_type: np.mean(mAPs) for c_type, mAPs in corruption_map.items()}
sorted_corruption_map = sorted(average_map.items(), key=lambda x: x[1])
sorted_corruption_types, sorted_average_map = zip(*sorted_corruption_map)

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(range(len(sorted_average_map)), sorted_average_map, marker='o', linestyle='-', color='#1f77b4', markersize=8, linewidth=2)
plt.xticks(range(len(sorted_corruption_types)), sorted_corruption_types, rotation=45, ha="right", fontsize=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.xlabel("Corruption Type", fontsize=12)
plt.ylabel("Average mAP@0.5:0.95", fontsize=12)
plt.title("Average Model Performance by Corruption Type (Sorted by mAP@0.5:0.95)", fontsize=14, fontweight='bold')
for i, txt in enumerate(sorted_average_map):
    plt.annotate(f"{txt:.2f}", (i, sorted_average_map[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, color='black')
plt.tight_layout()
plt.savefig("average_model_performance_sorted_beautified-1.png", dpi=300)


'''
import matplotlib.pyplot as plt

# Initialize lists to store data
corruption_types = []
severities = []
map_50_95 = []

# Read the output file
with open("validation_terminal_output.txt", "r") as file:
    for line in file:
        if "Results for" in line:
            parts = line.split()
            corruption_types.append(parts[2])
            severities.append(parts[5].strip(":"))
        elif "mAP@0.5:0.95:" in line:
            map_50_95.append(float(line.split(": ")[1]))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(len(map_50_95)), map_50_95, marker='o')
plt.xticks(range(len(map_50_95)), [f"{c}-{s}" for c, s in zip(corruption_types)], rotation=45)
plt.xlabel("Corruption Type - Severity")
plt.ylabel("mAP@0.5:0.95")
plt.title("Model Performance Across Corruption Types and Severities")
plt.tight_layout()
plt.savefig("Plot Performance Across Corruption Types and Severities")
plt.show()
'''