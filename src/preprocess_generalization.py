import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split, ConcatDataset
import re
import os
import random

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Cleaning text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# CrowdFlower loading
def load_crowdflower(path, allowed_labels, max_length=128):
    df = pd.read_csv(path)
    df = df[df['sentiment'].isin(allowed_labels)].reset_index(drop=True)

    label_map = {label: idx for idx, label in enumerate(sorted(allowed_labels))}
    df['label'] = df['sentiment'].map(label_map)

    texts = df['content'].apply(clean_text).tolist()
    labels = df['label'].tolist()

    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    print(f"CrowdFlower dataset loaded: {len(dataset)} samples")
    return dataset

# WASSA loading
def load_wassa(paths, emotion, label, threshold=0.75, max_length=128):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep='\t')
        df = df[df['score'] >= threshold]
        df['label'] = label
        df['content'] = df['tweet'].apply(clean_text)
        dfs.append(df[['content', 'label']])

    combined_df = pd.concat(dfs).reset_index(drop=True)

    encodings = tokenizer(combined_df['content'].tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(combined_df['label'].tolist()))
    print(f"WASSA-{emotion} dataset loaded: {len(dataset)} samples")
    return dataset

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


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)

    allowed_labels = ['anger', 'worry', 'happiness', 'sadness']
    label_mapping = {'anger':0, 'happiness':1, 'sadness':2, 'worry':3}


    # Load CrowdFlower dataset
    crowdflower_ds = load_crowdflower('../data/CrowdFlower/text_emotion.csv', allowed_labels)

    # Split CrowdFlower into train/val
    train_size = int(0.8 * len(crowdflower_ds))
    val_size = len(crowdflower_ds) - train_size
    train_ds, val_ds = random_split(crowdflower_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    torch.save(train_ds, '../data/train.pt')
    torch.save(val_ds, '../data/val.pt')

    # Load WASSA datasets
    wassa_paths = [
        '../data/WASSA2017/training/anger-ratings-0to1.train.txt',
        '../data/WASSA2017/validation/anger-ratings-0to1.dev.gold.txt',
        '../data/WASSA2017/training/fear-ratings-0to1.train.txt',
        '../data/WASSA2017/validation/fear-ratings-0to1.dev.gold.txt',
        '../data/WASSA2017/training/joy-ratings-0to1.train.txt',
        '../data/WASSA2017/validation/joy-ratings-0to1.dev.gold.txt',
        '../data/WASSA2017/training/sadness-ratings-0to1.train.txt',
        '../data/WASSA2017/validation/sadness-ratings-0to1.dev.gold.txt'
    ]

    wassa_ds_list = []
    for emotion in allowed_labels:
        emotion_paths = [p for p in wassa_paths if emotion in p or (emotion=='worry' and 'fear' in p) or (emotion=='happiness' and 'joy' in p)]
        ds = load_wassa(emotion_paths, emotion, label_mapping[emotion], threshold=0.75)
        wassa_ds_list.append(ds)

    wassa_combined_ds = ConcatDataset(wassa_ds_list)
    torch.save(wassa_combined_ds, '../data/test.pt')

    print("Datasets prepared and saved:")
    print(f"- CrowdFlower train: {len(train_ds)} samples")
    print(f"- CrowdFlower val: {len(val_ds)} samples")
    print(f"- WASSA test: {len(wassa_combined_ds)} samples")
