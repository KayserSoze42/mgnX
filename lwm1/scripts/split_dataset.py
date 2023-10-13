import pandas as pd
from sklearn.model.selection import train_test_split

# L MD CSV

metadata = pd.read_csv('metadata.csv')

# S MD t v ts

train_data, temp_data = train_test_split(metadata, stratify=metadata['class'], test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(temp_data, stratify=metadata['class'], test_soze=0.33, random_state=42)

# S S CSV

train_data.to_csv('train_metadata.csv', index=False)
valid_data.to_csv('valid_metadata.csv', index=False)
test_data.to_csv('test_metadata.csv', index=False)
