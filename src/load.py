import json
import numpy as np
# Open the JSON file for reading
with open('../data/myData.json', 'r') as file:
    #   Load the JSON content
    #   d(ii).v4counts %spike counts from the v4 array
    #   d(ii).v4ecounts %spike counts from the v4e array
    #   d(ii).dayNum  %days post surgery for this entry
    #   d(ii).session %the session date YYYYMMDD
    #   d(ii).drug %a flag that specifies which days the monkey received ritalin. we can talk about this later, if it matters
    data = json.load(file)
    first_v4counts = data[0]['v4counts']
    #print(first_v4counts)
    matrix = np.array(first_v4counts)
    print(matrix)
    
    print(data[0]['session'])
    print(data[0]['dayNum'])
    
# Compute the correlation matrix
corr_matrix = np.corrcoef(matrix)

# Since the correlation matrix is symmetric and has 1s on the diagonal (self-correlation),
# we remove the diagonal and redundant values by masking them
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
corr_values = corr_matrix[mask]

# Compute the mean of the correlation coefficients (excluding self-correlations)
mean_spike_count_correlation = np.mean(corr_values)

print("Mean Spike Count Correlation:", mean_spike_count_correlation)

import json
import numpy as np
import matplotlib.pyplot as plt

# Load your data
with open('../data/myData.json', 'r') as file:
    data = json.load(file)

# Initialize lists to hold day numbers and mean correlations
day_nums = []
mean_correlations = []

# Loop over each entry in your dataset
for entry in data:
    # Convert the spike counts to a numpy array, filtering out any entries that contain 'None'
    matrix_data = entry['v4ecounts']  # or 'v4ecounts', depending on your requirement
    # Check if there's 'None' in matrix_data, and skip if found
    if any(None in row for row in matrix_data):
        continue
    
    if len(matrix_data) <= 30:
        continue

    # Convert to a numpy array temporarily to check column sizes
    temp_matrix = np.array(matrix_data, dtype=object)  # Use dtype=object to avoid errors with None values
    
    # Ensure all columns have more than 30 entries
    # This check assumes that all rows are of equal length; adjust as needed for your data
    if temp_matrix.shape[1] <= 50:  # Checking the number of columns
        continue
    
    
    matrix = np.array(matrix_data)  # Ensure the data is in the correct type for correlation calculation
    print(matrix)
    # Compute the correlation matrix
    corr_matrix = np.corrcoef(matrix)
    
    # Remove the diagonal and redundant values
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_values = corr_matrix[mask]
    
    # Compute the mean of the correlation coefficients
    mean_corr = np.mean(corr_values)
    
    # Append the day number and the mean correlation to their respective lists
    day_nums.append(entry['dayNum'])
    mean_correlations.append(mean_corr)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(day_nums, mean_correlations, alpha=0.6)  # alpha for better visualization of overlapping points
plt.title('Mean Spike Count Correlation vs. Day Number')
plt.xlabel('Day Number')
plt.ylabel('Mean Spike Count Correlation')
plt.grid(True)
plt.show()
