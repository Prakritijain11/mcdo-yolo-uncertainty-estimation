import matplotlib.pyplot as plt

corruption_types = []
severities = []
map_50_95 = []

# Read the output file for corrupted data
with open("validation_corruption_severity.txt", "r") as file:
    for line in file:
        if "Results for" in line:
            parts = line.split()
            corruption_types.append(parts[2])
            severities.append(parts[5].strip(":"))
        elif "mAP@0.5:0.95:" in line:
            map_50_95.append(float(line.split(": ")[1]))

# Read the zero severity data
with open("validation_zero_severity.txt", "r") as file:
    for line in file:
        if "mAP@0.5:0.95:" in line:
            zero_severity_map_50_95 = float(line.split(": ")[1])
            break

corruption_types.insert(0, 'Original')
severities.insert(0, '0')
map_50_95.insert(0, zero_severity_map_50_95)

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(range(len(map_50_95)), map_50_95, marker='o', linestyle='-')
plt.xticks(range(len(map_50_95)), [f"{c}-{s}" for c, s in zip(corruption_types, severities)], rotation=45)
plt.xlabel("Corruption Type - Severity")
plt.ylabel("mAP@0.5:0.95")
plt.title("Model Performance Across Corruption Types and Severities")
plt.tight_layout()
plt.grid(True)
plt.show()