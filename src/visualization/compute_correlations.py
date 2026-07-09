import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('Corruption_Data-0.csv')

# Calculate correlations for each corruption type
correlations = df.groupby('corruption').apply(lambda x: x[['confidence', 'map', 'boxes']].corr())

# Extract the specific correlations we need
conf_map = correlations.xs('confidence', level=1)['map']
map_boxes = correlations.xs('map', level=1)['boxes']
boxes_conf = correlations.xs('boxes', level=1)['confidence']

# Create a DataFrame to store the correlation coefficients
correlation_results = pd.DataFrame({
    'Corruption': conf_map.index,
    'Confidence_mAP': conf_map.values,
    'mAP_Boxes': map_boxes.values,
    'Boxes_Confidence': boxes_conf.values
})

# Save the correlation coefficients to a CSV file
correlation_results.to_csv('correlation_coefficients.csv', index=False)

print("Correlation coefficients have been saved to 'correlation_coefficients.csv'")

# Create the three graphs with trendlines
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Confidence Score vs mAP
sns.scatterplot(data=df, x='confidence', y='map', ax=ax1)
ax1.set_title('Confidence Score vs mAP')
ax1.set_xlabel('Confidence')
ax1.set_ylabel('mAP')
z = np.polyfit(df['confidence'], df['map'], 1)
p = np.poly1d(z)
ax1.plot(df['confidence'], p(df['confidence']), "r--", alpha=0.8)

# Plot 2: mAP vs Boxes
sns.scatterplot(data=df, x='map', y='boxes', ax=ax2)
ax2.set_title('mAP vs Boxes')
ax2.set_xlabel('mAP')
ax2.set_ylabel('Boxes')
z = np.polyfit(df['map'], df['boxes'], 1)
p = np.poly1d(z)
ax2.plot(df['map'], p(df['map']), "r--", alpha=0.8)

# Plot 3: Boxes vs Confidence Score
sns.scatterplot(data=df, x='confidence', y='boxes', ax=ax3)
ax3.set_title('Boxes vs Confidence Score')
ax3.set_xlabel('Confidence')
ax3.set_ylabel('Boxes')
z = np.polyfit(df['confidence'], df['boxes'], 1)
p = np.poly1d(z)
ax3.plot(df['confidence'], p(df['confidence']), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig('person-coefficient-new5.png')
plt.show()