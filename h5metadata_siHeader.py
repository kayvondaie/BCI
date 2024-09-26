import json

file = 'C:/Users/kayvon.daie/Downloads/metadata (1).json'

# Open and load the JSON file
with open(file, 'r') as f:
    data = json.load(f)

# Now `data` contains the contents of the JSON file as a Python dictionary
print(data)

#%%
# The content of data['neuron17'] (as shown in your previous example)
data_neuron17_str = data['neuron17']

# Split the string by newlines and process each line
lines = data_neuron17_str.split('\n')

# Initialize an empty dictionary
neuron17_data = {}

# Loop through each line and extract key-value pairs
for line in lines:
    # Split each line by the colon to get key-value pairs
    if ':' in line:
        key, value = line.split(':', 1)
        # Strip any extra spaces and clean the key/value
        key = key.strip()
        value = value.strip().strip("'")  # Removes surrounding quotes if present
        # Add to the dictionary
        neuron17_data[key] = value

# Now you have a dictionary
print(neuron17_data)

#%%
import data_dict_create_module as ddc
folder = r'//allen/aind/scratch/BCI/2p-raw/BCI87/081324/'
siHeader = ddc.siHeader_get(folder)