import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON data
with open('../data/myData.json', 'r') as file:
    data = json.load(file)

# Check if data entry is valid
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
fig, axs = plt.subplots(4, 4, figsize=(30, 20))  # 4x4 grid for 16 subplots
axs = axs.flatten()  # Flatten the array of axes for easier iteration

for i, entry in enumerate(selected_entries):
    matrix = np.array(entry['v4ecounts'])
    
    # Compute the firing rates: mean across neurons for each trial
    firing_rates = np.mean(matrix, axis=0)*5  # Multiply by 5 if you're converting to Hz and your time bin is 200ms
    
    # Plot firing rate vs trials
    axs[i].plot(firing_rates, color='blue', marker='o', linestyle='-')
    axs[i].set_title(f'Day {entry["dayNum"]} - Firing Rate vs Trials- V4E')
    axs[i].set_xlabel('Trial Number')
    axs[i].set_ylabel('Firing Rate (Hz)')

plt.tight_layout()
plt.savefig('../plots/firing_rate_vs_trials_v4e.png')
#plt.show()
