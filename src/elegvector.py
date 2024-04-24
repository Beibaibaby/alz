import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON data
with open('../data/myData.json', 'r') as file:
    data = json.load(file)

# Function to check if data entry is valid
def is_valid_entry(entry):
    matrix_data = entry['v4ecounts']
    return None not in [item for sublist in matrix_data for item in sublist] and len(matrix_data) >= 30

# Filter out invalid data entries first
valid_data = [entry for entry in data if is_valid_entry(entry)]

# Ensure there are enough valid entries
if len(valid_data) < 16:
    raise ValueError("Not enough valid data entries to select 16 unique datasets.")
selected_entries = np.random.choice(valid_data, 16, replace=False)

# Create a figure with a grid of subplots
fig, axs = plt.subplots(4, 4, figsize=(18, 18))
axs = axs.flatten()  # Flatten array of axes for easier iteration

for i, entry in enumerate(selected_entries):
    matrix = np.array(entry['v4ecounts'])
    cov_matrix = np.cov(matrix, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Get the principal eigenvector (associated with the largest eigenvalue)
    principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    # Plot the principal eigenvector
    axs[i].plot(principal_eigenvector, marker='o', linestyle='-', color='royalblue')
    axs[i].set_title(f'Dataset {i+1} - Principal Eigenvector')
    axs[i].set_xlabel('Component Index')
    axs[i].set_ylabel('Value')

plt.tight_layout()
plt.savefig('../eigenvectors.png')
plt.show()
