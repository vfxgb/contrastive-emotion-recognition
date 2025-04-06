import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
import os
import spacy
from config import (
    BERT_MODEL,
    SPACY_MODEL,
    CROWDFLOWER_PATH,
    CROWDFLOWER_TEST_DS_PATH,
    CROWDFLOWER_TRAIN_DS_PATH,
)
from utils import clean_text, split_dataset, DualViewDataset

# Initialize tokenizer and spaCy model
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
nlp = spacy.load(SPACY_MODEL)


def load_crowdflower(path, max_length=128, min_samples=1000):
    """
    Load and preprocess the CrowdFlower dataset.

    Args:
        path (str): Path to the CSV file.
        max_length (int): Maximum sequence length for tokenization.
        min_samples (int): Minimum number of samples required per class to include in the dataset.

    Returns:
        TensorDataset: Dataset object containing input_ids, attention_masks, and labels.
        list: Original text data.
        list: Original labels.
        dict: Mapping from original labels to new label indices.

    """

    df = pd.read_csv(path)
    print(f"[CrowdFlower] Loaded {len(df)} rows from {path}")

    df["original"] = df["content"]  # Keep original for debugging
    df["content"] = df["content"].apply(clean_text)

    # Encode labels
    unique_sentiments = sorted(df["sentiment"].unique())
    label_map = {label: idx for idx, label in enumerate(unique_sentiments)}
    df["label"] = df["sentiment"].map(label_map)

    # Print original distribution
    print(f"[Original Label Distribution]")
    print(df["label"].value_counts())

    # Filter out classes with < min_samples
    class_counts = df["label"].value_counts()
    keep_classes = class_counts[class_counts >= min_samples].index.tolist()
    df_filtered = df[df["label"].isin(keep_classes)].reset_index(drop=True)

    # print info filtered dataset
    print(f"[Filtered Label Distribution (min {min_samples} per class)]")
    print(df_filtered["label"].value_counts())
    print("\n[Filtered Sentiment Distribution]")
    print(df_filtered["sentiment"].value_counts())

    # Re-map labels to be consecutive
    new_label_map = {
        old: i for i, old in enumerate(sorted(df_filtered["label"].unique()))
    }
    df_filtered["label"] = df_filtered["label"].map(new_label_map)

    texts = df_filtered["content"].tolist()
    labels = df_filtered["label"].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_masks = encodings["attention_mask"]
    label_tensor = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, label_tensor)
    print(f"[CrowdFlower] Final FILTERED dataset shape: {len(dataset)} samples")
    print(f"[Final Label Mapping] {new_label_map}")

    return dataset, texts, labels, new_label_map


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("[Main] Loading and processing CrowdFlower dataset...")
    full_dataset, all_texts, all_labels, label_map = load_crowdflower(CROWDFLOWER_PATH)

    print("[Main] Splitting dataset into train and test...")
    train_ds, test_ds = split_dataset(full_dataset, split_ratio=0.8)

    # Extract unique input_ids (as tuples) for train and test subsets
    train_ids = set(
        [tuple(train_ds.dataset.tensors[0][i].tolist()) for i in train_ds.indices]
    )
    test_ids = set(
        [tuple(test_ds.dataset.tensors[0][i].tolist()) for i in test_ds.indices]
    )

    # Compute the intersection of train and test sets
    overlap = train_ids.intersection(test_ids)
    print(f"[Leakage Check] Overlapping samples between train & test: {len(overlap)}")

    print("[Main] Saving datasets to disk...")
    torch.save(train_ds, CROWDFLOWER_TRAIN_DS_PATH)
    torch.save(test_ds, CROWDFLOWER_TEST_DS_PATH)
    print("[Main] Done.")
