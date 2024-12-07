import pandas as pd
import random
from itertools import permutations

# Define column names for the datasets
columns = ['label','sentence']

# Read the TSV files with specified column names
df_train = pd.read_csv("train.tsv", sep="\t", names=columns)
df_test = pd.read_csv("test.tsv", sep="\t", names=columns)
df_dev = pd.read_csv("dev.tsv", sep="\t", names=columns)

print(df_train.head())
# Separate positive and negative samples from training data
positive_samples = df_train[df_train['label'] == 1].values.tolist()
negative_samples = df_train[df_train['label'] == 0].values.tolist()
print(len(positive_samples))
print("Positive samples: ", positive_samples[0],positive_samples[1],positive_samples[2],positive_samples[3])

# Function to create a single prompt with given samples
def create_prompt(train_samples, test_sentence):
    prompt_parts = [f"Review: {sample[1]} Sentiment: {sample[0]}." for sample in train_samples]
    prompt_parts.append(f"Review: {test_sentence} Sentiment:<mask>.")
    return " ".join(prompt_parts)

# Function to get all permutations of 4 samples (2 positive, 2 negative)
def get_sample_permutations():
    pos_samples = random.sample(positive_samples, 2)
    neg_samples = random.sample(negative_samples, 2)
    all_samples = pos_samples + neg_samples
    return list(permutations(all_samples))

# Create the new dataframe structure
new_data = []
test_samples = random.sample(df_test.values.tolist(), 5)  # Get 5 random test samples

for test_sample in test_samples:
    perms = get_sample_permutations()
    test_label, test_sentence = test_sample
    
    for perm in perms:
        new_data.append({
            'sentence': test_sentence,
            'prompt': create_prompt(perm, test_sentence),
            'label': test_label,
            'predicted_label': ''
        })

# Create the new dataframe
new_df = pd.DataFrame(new_data)

# Save to CSV (optional)
new_df.to_csv('prompts.csv', index=False)

# Print sample to verify
print(f"Total rows in new dataset: {len(new_df)}")
# print("\nSample prompt:")
# print(new_df['prompt'].iloc[0])
# print("\nDataset head:")
#print(new_df.head())
print("Sample prompt:")
print(new_df['sentence'][0])
print("Prompt:")
print(new_df['prompt'][0])
print("Label:")
print(new_df['label'][0])

