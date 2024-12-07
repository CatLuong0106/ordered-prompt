from matplotlib import pyplot as plt
from pathlib import Path
from collections import defaultdict
import json 

# 1 Graph for how the accuracy varies
# 3 different permutations for 2 different models

def generate_line_graph(dataset="mpqa"): 
    folder_path = Path(dataset)
    accuracy = defaultdict(list)
    accuracy['1B'] = [0.8027166666666666, 0.7962166666666667, 0.79233]
    accuracy['3B'] = [0.8418, 0.8330083333333332, 0.8370875]
    permutations = [6, 12, 24]
    
    fig, ax = plt.subplots()
    ax.plot(permutations, accuracy['1B'], label='1B')
    ax.plot(permutations, accuracy['3B'], label='3B')
    ax.scatter(permutations, accuracy['1B'])
    ax.scatter(permutations, accuracy['3B'])
    ax.set_title(f"The {dataset} results for Llama 3.2 1B vs. 3B")
    ax.set_xticks(permutations)  # Set x-ticks to exactly match permutation values
    ax.set_xlabel('Permutations')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.savefig(f"{folder_path}/{dataset}_accuracy_line_graph.png")

def generate_entropy_line_graph(dataset="mpqa"):
    folder_path = Path(dataset)
    entropy = defaultdict(dict)
    for file in folder_path.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            if '1B' in file.name:
                key_name = '1B'
            elif '3B' in file.name: 
                key_name = '3B'
                
            if len(data['entropys']) == 6:
                entropy[key_name][6] = data['entropys']
            if len(data['entropys']) == 12:
                entropy[key_name][12] = data['entropys']
            if len(data['entropys']) == 24:
                entropy[key_name][24] = data['entropys']
    
    fig, ax = plt.subplots()
    label_key = '1B'
    ax.plot([i + 1 for i in range(6)], entropy[label_key][6], label=label_key)
    ax.plot([i + 1 for i in range(12)], entropy[label_key][12], label=label_key)
    ax.plot([i + 1 for i in range(24)], entropy[label_key][24], label=label_key)
    ax.scatter([i + 1 for i in range(6)], entropy[label_key][6])
    ax.scatter([i + 1 for i in range(12)], entropy[label_key][12])
    ax.scatter([i + 1 for i in range(24)], entropy[label_key][24])
    ax.set_title(f"The {dataset} results for entropy vs. permutation with Llama 3.2-{label_key}")
    ax.set_xticks([i + 1 for i in range(24)])  # Set x-ticks to exactly match permutation values
    ax.set_xlabel('Permutations')
    ax.set_ylabel('Accuracy')
    plt.savefig(f"{folder_path}/{dataset}_{label_key}_entropy_line_graph.png")
    
if __name__ == "__main__":
    dataset = ["mpqa","trec"]
    # generate_line_graph(dataset=dataset[0])
    generate_entropy_line_graph(dataset=dataset[1])