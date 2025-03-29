"""Direct comparison"""
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split, ConcatDataset
import re
import os
import random

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# DualViewDataset class
def random_dropout_tokens(token_ids, dropout_prob=0.1):
    return [tok for tok in token_ids if random.random() > dropout_prob or tok in [101, 102, 0]]

class DualViewDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset, dropout_prob=0.1):
        if isinstance(tensor_dataset, torch.utils.data.Subset):
            self.input_ids = tensor_dataset.dataset.tensors[0][tensor_dataset.indices]
            self.attention_masks = tensor_dataset.dataset.tensors[1][tensor_dataset.indices]
            self.labels = tensor_dataset.dataset.tensors[2][tensor_dataset.indices]
        else:
            self.input_ids = tensor_dataset.tensors[0]
            self.attention_masks = tensor_dataset.tensors[1]
            self.labels = tensor_dataset.tensors[2]
        self.dropout_prob = dropout_prob

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        original = self.input_ids[idx].tolist()
        view1 = random_dropout_tokens(original, self.dropout_prob)
        view2 = random_dropout_tokens(original, self.dropout_prob)

        max_len = len(original)
        view1 = view1 + [0] * (max_len - len(view1))
        view2 = view2 + [0] * (max_len - len(view2))

        view1 = torch.tensor(view1)
        view2 = torch.tensor(view2)
        label = self.labels[idx]

        return view1, view2, label

# Cleaning text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Explicit consistent label mapping
label_mapping = {'anger': 0, 'sadness': 1}

# WASSA loading
def load_wassa(paths, emotion, cf_label, threshold=0.75, max_length=128):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep='\t')
        df = df[df['score'] >= threshold]
        df['label'] = cf_label
        df['content'] = df['tweet'].apply(clean_text)
        dfs.append(df[['content', 'label']])

    combined_df = pd.concat(dfs).reset_index(drop=True)

    encodings = tokenizer(combined_df['content'].tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(combined_df['label'].tolist()))
    print(f"WASSA-{emotion} dataset loaded: {len(dataset)} samples")
    return dataset

# Main execution
if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)

    # WASSA paths
    wassa_paths = {
        'anger': ["../data/WASSA2017/training/anger-ratings-0to1.train.txt",
                  "../data/WASSA2017/validation/anger-ratings-0to1.dev.gold.txt"],
        'sadness': ["../data/WASSA2017/training/sadness-ratings-0to1.train.txt",
                    "../data/WASSA2017/validation/sadness-ratings-0to1.dev.gold.txt"]
    }

    wassa_ds_list = []
    for emotion, paths in wassa_paths.items():
        ds = load_wassa(paths, emotion, label_mapping[emotion], threshold=0.75)
        wassa_ds_list.append(ds)

    wassa_combined_ds = ConcatDataset(wassa_ds_list)

    # Split into train/test
    train_size = int(0.8 * len(wassa_combined_ds))
    test_size = len(wassa_combined_ds) - train_size
    wassa_train, wassa_test = random_split(wassa_combined_ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    torch.save(wassa_train, '../data/train.pt')
    torch.save(wassa_test, '../data/test.pt')

    print("Datasets prepared and saved (WASSA only):")
    print(f"- WASSA train: {len(wassa_train)} samples")
    print(f"- WASSA test: {len(wassa_test)} samples")
