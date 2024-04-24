import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Function to calculate dimensionality using eigenvalues of the covariance matrix
def calculate_dimensionality(eigenvalues):
    return ((np.sum(eigenvalues)/np.size(eigenvalues)) ** 2) / (np.sum(eigenvalues ** 2)/np.size(eigenvalues))

# Normalization of D(C) according to provided formulas
def normalize_dimensionality(D_C, N, M):
    alpha = N / M  # Ratio of number of neurons to time samples
    D_C_hat = D_C  / (1 + alpha * D_C)
    return D_C_hat


# Function to calculate sparsity of the mean response
def calculate_sparsity(spike_counts):
    mean_spike_count = np.mean(spike_counts)
    return (mean_spike_count ** 2) / (np.mean(spike_counts ** 2))

# Load the JSON data
with open('../data/myData.json', 'r') as file:
    data = json.load(file)

# Initialize lists to hold day numbers, mean correlations, dimensionalities, sparsity measures, and trial counts
day_nums = []
mean_correlations = []
dimensionalities = []
sparsities = []
trial_counts = []

# Loop over each entry in the dataset
for entry in data:
    matrix_data = entry['v4ecounts']

    # Skip entries with 'None' or insufficient data
    if any(None in sublist for sublist in matrix_data) or len(matrix_data) < 30:
        continue

    matrix = np.array(matrix_data)
    N = matrix.shape[0]  # Number of neurons
    M = matrix.shape[1]  # Number of time samples
 
    # Compute the correlation matrix and mean correlation
    corr_matrix = np.corrcoef(matrix)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix[mask]
    mean_corr = np.mean(corr_values)
    
    # Calculate the covariance matrix, eigenvalues, and dimensionality
    cov_matrix = np.cov(matrix, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]   # Indices for sorting eigenvalues in descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Exclude the largest eigenvalue and its corresponding eigenvector
    eigenvalues = eigenvalues[1:]  # All except the first (largest)
    eigenvectors = eigenvectors[:, 1:]  # All columns except the first

    
    D_C= calculate_dimensionality(eigenvalues)
    dim = normalize_dimensionality(D_C, N, M)
    sparsity = calculate_sparsity(matrix)
    
    # Append calculated metrics to their respective lists
    day_nums.append(entry['dayNum'])
    mean_correlations.append(mean_corr)
    dimensionalities.append(dim)
    sparsities.append(sparsity)
    trial_counts.append(matrix.shape[1])  # Number of trials is the number of columns

# Normalize trial counts to use in color mapping for scatter plots
trial_norm = mcolors.Normalize(vmin=min(trial_counts), vmax=max(trial_counts))
norm = mcolors.Normalize(vmin=min(day_nums), vmax=max(day_nums))
# Create a figure with subplots for the time series plots
fig, axs = plt.subplots(3, 1, figsize=(18, 18), sharex=False)

# Create scatter plots and color the dots based on trial counts
for i, (y, title, ylabel) in enumerate(zip(
    [mean_correlations, dimensionalities, sparsities],
    ['Mean Spike Count Correlation vs. Day Number (V4-Premoved)', 'Dimensionality vs. Day Number (V4-Premoved)', 'Sparsity vs. Day Number (V4-Premoved)'],
    ['Mean Spike Count Correlation', 'Dimensionality-D(C)', 'D(R)'])):
    
    sc = axs[i].scatter(day_nums, y, c=trial_counts, cmap='viridis', norm=trial_norm, alpha=0.6)
    axs[i].set_title(title)
    axs[i].set_ylabel(ylabel)
    axs[i].grid(True)
    
    # Add a colorbar to each subplot to show trial count corresponding to the color
    cbar = fig.colorbar(sc, ax=axs[i])
    cbar.set_label('Trial Count')

plt.tight_layout()
plt.savefig('../plots/time_v4_premoved.png')
#plt.show()

# Separate scatter plot for sparsity (x) vs dimensionality (y) with day as color
fig2, ax4 = plt.subplots(figsize=(8, 6))
sc2 = ax4.scatter(sparsities, dimensionalities, c=day_nums, cmap='viridis', norm=norm, alpha=0.6)
ax4.set_title('Sparsity vs. Dimensionality (V4-Premoved)')
ax4.set_xlabel('D(R)')
ax4.set_ylabel('Dimensionality-D(C)')
ax4.grid(True)
# Add a colorbar to show the day number corresponding to the color
cbar2 = fig2.colorbar(sc2, ax=ax4)
cbar2.set_label('Day Number')

plt.tight_layout()
plt.savefig('../plots/D(C)vsD(R)_v4_premoved.png')
#plt.show()





import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the JSON data
with open('../data/myData.json', 'r') as file:
    data = json.load(file)

# Function to check if data entry is valid
def is_valid_entry(entry):
    matrix_data = entry['v4counts']
    return None not in [item for sublist in matrix_data for item in sublist] and len(matrix_data) >= 30

# Filter out invalid data entries first
valid_data = [entry for entry in data if is_valid_entry(entry)]

# Arrays to hold the largest eigenvalues and the day numbers
largest_eigenvalues = []
day_nums = []
trial_counts = []

# Loop over each valid entry to extract the largest eigenvalue, day number, and trial count
for entry in valid_data:
    matrix = np.array(entry['v4counts'])
    cov_matrix = np.cov(matrix, rowvar=True)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    largest_eigenvalue = np.max(eigenvalues)
    
    if largest_eigenvalue > 2500:
        continue
    
    largest_eigenvalues.append(largest_eigenvalue)
    day_nums.append(entry['dayNum'])
    trial_counts.append(len(matrix[0]))  # Assuming the number of trials is the length of the time dimension

# Normalize trial counts for color mapping
trial_norm = mcolors.Normalize(vmin=min(trial_counts), vmax=max(trial_counts))

# Create a scatter plot
fig, ax = plt.subplots(figsize=(15, 5))
sc = ax.scatter(day_nums, largest_eigenvalues, c=trial_counts, cmap='viridis', norm=trial_norm, alpha=0.6)
ax.set_title('Greatest Eigenvalue over Days (V4, <2500)')
ax.set_xlabel('Day Number')
ax.set_ylabel('Greatest Eigenvalue')
ax.grid(True)
# Add a color bar representing the number of trials
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Number of Trials')

plt.tight_layout()
plt.savefig('../plots/greatest_eigenvalue_over_day_v4.png')

