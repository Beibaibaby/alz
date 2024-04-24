import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON data
with open('../data/myData.json', 'r') as file:
    data = json.load(file)

# Initialize lists to hold firing rates, channel numbers, and eigenvalues
firing_rates = []
channel_numbers = []
all_eigenvalues = []

# Loop over each entry in the dataset
i=0
for entry in data:
    if i ==0:
      
        
    i+=1
    
    matrix_data = entry['v4ecounts']  # Adjust this key as needed

    # Skip entries with 'None' or insufficient data
    if any(None in sublist for sublist in matrix_data) or len(matrix_data) < 30:
        continue

    matrix = np.array(matrix_data)

    # Convert spike counts to firing rates (Hz) and filter out rates over 200 Hz
    print(matrix.shape)
    rates = [count * 5 for sublist in matrix_data for count in sublist if count is not None and (count * 5) <= 1000]
    firing_rates.extend(rates)

        # Store the number of channels
    channel_numbers.append(matrix.shape[0])

        # Calculate the covariance matrix and its eigenvalues, filter out eigenvalues over 500
    cov_matrix = np.cov(matrix, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    filtered_eigenvalues = [eig for eig in eigenvalues if eig >= 500]
    all_eigenvalues.extend(filtered_eigenvalues)

# Plot histogram of firing rates
plt.figure(figsize=(10, 6))
plt.hist(firing_rates, bins=50, color='blue', edgecolor='black', alpha=0.75)
plt.title('Histogram of Firing Rates')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('../firing_rate_histogram.png')
plt.show()

# Plot histogram of channel numbers
plt.figure(figsize=(10, 6))
plt.hist(channel_numbers, bins=30, color='green', edgecolor='black', alpha=0.75)
plt.title('Histogram of Channel Numbers')
plt.xlabel('Number of Channels')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('../channel_number_histogram.png')
plt.show()

# Plot histogram of covariance matrix eigenvalues
plt.figure(figsize=(10, 6))
plt.hist(all_eigenvalues, bins=100, color='red', edgecolor='black', alpha=0.75)
plt.title('Histogram of Eigenvalues of Covariance Matrix')
plt.xlabel('Eigenvalue')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('../covariance_eigenvalues_histogram.png')
plt.show()
