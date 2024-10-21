import os
import json
import random
import torch
import torch.nn.functional as F
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.stats import entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]
        else:
            label = ''  # Fallback if no labels are available

        text = clip.tokenize([label], truncate=True) 

        return image, text.squeeze(0)  # Remove the extra dimension

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
#_, preprocess = clip.load("ViT-L/14", device=device)
#model = torch.load(path/to/finetune.pt)
#model = model.to(device)

# Images: Center-Crop of https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco
val_dataset = ImageTextDataset(
    image_folder="path/to/COCO/data-square",
    annotations_file="attn-test-coco-spright-val.json",
    transform=preprocess
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

image_attention_weights = []
text_attention_weights = []

def get_attention_weights(module_type):
    def hook(module, input, output):
        if isinstance(output, tuple):
            attn_logits = output[0]
            attn_weights = F.softmax(attn_logits, dim=-1)
            if module_type == 'image':
                image_attention_weights.append(attn_weights)
            elif module_type == 'text':
                text_attention_weights.append(attn_weights)
    return hook

# Register hooks
for idx, block in enumerate(model.visual.transformer.resblocks):
    block.attn.register_forward_hook(get_attention_weights('image'))

for idx, block in enumerate(model.transformer.resblocks):
    block.attn.register_forward_hook(get_attention_weights('text'))

def calculate_attention_entropy(attn_weights):
    entropies = []
    for layer_attn in attn_weights:
        if layer_attn is None:
            continue  # Skip if no attention weights were captured
        batch_size, num_heads, seq_len = layer_attn.shape
        layer_entropies = []
        for head in range(num_heads):
            head_attn = layer_attn[:, head, :].cpu().numpy()
            if np.all(head_attn == 0):
                continue  # Skip if all attention values are zero
            head_entropy = entropy(head_attn, axis=-1)
            head_entropy = np.nan_to_num(head_entropy)  # Replace NaNs with 0 or a small value
            layer_entropies.append(head_entropy.mean())
        if layer_entropies:
            entropies.append(np.mean(layer_entropies))
    if entropies:
        return entropies
    return [0.0]  # Fallback if no valid entropies were computed

results = []

for i, (image, text) in enumerate(val_loader):
    image, text = image.to(device), text.to(device)
    
    image_attention_weights.clear()
    text_attention_weights.clear()
    
    with torch.no_grad():
        model.encode_image(image)
        model.encode_text(text)
    
    image_entropies = calculate_attention_entropy(image_attention_weights)
    avg_image_entropy = np.mean(image_entropies)

    text_entropies = calculate_attention_entropy(text_attention_weights)
    avg_text_entropy = np.mean(text_entropies)

    results.append({
        "image_index": i,
        "image_entropies_per_layer": image_entropies,
        "avg_image_entropy": avg_image_entropy,
        "text_entropies_per_layer": text_entropies,
        "avg_text_entropy": avg_text_entropy,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("attention_entropy_results_with_text.csv", index=False)

avg_image_entropy_per_layer = results_df["image_entropies_per_layer"].apply(lambda x: np.array(x)).mean(axis=0)
avg_text_entropy_per_layer = results_df["text_entropies_per_layer"].apply(lambda x: np.array(x)).mean(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(range(len(avg_image_entropy_per_layer)), avg_image_entropy_per_layer, marker='o', label="Image Entropy")
plt.title("Average Image Attention Entropy per Layer Across Validation Dataset")
plt.xlabel("Layer Index")
plt.ylabel("Average Entropy")
plt.grid(True)
plt.savefig("avg_image_entropy_per_layer.png")

plt.figure(figsize=(10, 6))
plt.plot(range(len(avg_text_entropy_per_layer)), avg_text_entropy_per_layer, marker='s', label="Text Entropy")
plt.title("Average Text Attention Entropy per Layer Across Validation Dataset")
plt.xlabel("Layer Index")
plt.ylabel("Average Entropy")
plt.grid(True)
plt.savefig("avg_text_entropy_per_layer.png")

plt.figure(figsize=(10, 6))
plt.hist(results_df["avg_image_entropy"], bins=30, edgecolor='black')
plt.title("Distribution of Average Entropy Across Images")
plt.xlabel("Average Entropy")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("image_entropy_distribution.png")

plt.figure(figsize=(10, 6))
plt.hist(results_df["avg_text_entropy"], bins=30, edgecolor='black')
plt.title("Distribution of Average Entropy Across Texts")
plt.xlabel("Average Entropy")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("text_entropy_distribution.png")


print("\nEntropy calculation complete.")
