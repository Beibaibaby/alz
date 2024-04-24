import json
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate dimensionality using eigenvalues of the covariance matrix
def calculate_dimensionality(eigenvalues):
    D_C = ((np.sum(eigenvalues) / np.size(eigenvalues)) ** 2) / (np.sum(eigenvalues ** 2) / np.size(eigenvalues))
    return D_C

# Normalization of D(C) according to provided formulas
def normalize_dimensionality(D_C, N, M):
    alpha = N / M  # Ratio of number of neurons to time samples
    D_C_hat = D_C  / (1 + alpha * D_C)
    return D_C_hat

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
fig = plt.figure(figsize=(20, 20))

for i, entry in enumerate(selected_entries):
    matrix = np.array(entry['v4counts'])
    N = matrix.shape[0]  # Number of neurons
    M = matrix.shape[1]  # Number of time samples
  
    cov_matrix = np.cov(matrix, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Calculate the dimensionality and normalize it
    D_C = calculate_dimensionality(eigenvalues)
    D_C_hat = normalize_dimensionality(D_C, N, M)

    # Get the principal eigenvector (associated with the largest eigenvalue)
    principal_index = np.argmax(eigenvalues)
    principal_eigenvector = eigenvectors[:, principal_index]

    # Histogram of eigenvalues
    ax1 = plt.subplot(8, 4, 2*i + 1)
    ax1.hist(eigenvalues, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title(f'Dataset {i+1} - Normalized D(C)={D_C_hat:.2f}, Num Eigen={len(eigenvalues)},Day {entry['dayNum']}')
    ax1.set_xlabel('Eigenvalue')
    ax1.set_ylabel('Frequency')

    # Line plot of the principal eigenvector
    ax2 = plt.subplot(8, 4, 2*i + 2)
    ax2.plot(principal_eigenvector, color='royalblue', marker='o', linestyle='-')
    ax2.set_title(f'Principal Eigenvector - V4 ')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Eigenvector Value')

plt.tight_layout()
plt.savefig('../plots/eigenvalue+vectors_v4.png')
#plt.show()
