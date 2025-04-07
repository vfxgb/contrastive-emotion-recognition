import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
import os
import spacy
import numpy as np 
from config import (
    BERT_MODEL,
    SPACY_MODEL,
    GLOVE_PATH,
    CROWDFLOWER_PATH,
    CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE,
    CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE,
    CROWDFLOWER_TEST_DS_PATH_WITH_GLOVE, 
    CROWDFLOWER_TRAIN_DS_PATH_WITH_GLOVE,
    CROWDFLOWER_GLOVE_EMBEDDINGS_PATH
)
from utils import clean_text, split_dataset, load_glove_embeddings
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

nlp = spacy.load(SPACY_MODEL)

def load_crowdflower_with_glove(path, max_length=128, min_samples=1000):
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
    tokenizer = Tokenizer(num_words=5000,oov_token="<UNK>")

    df = pd.read_csv(path)
    print(f"[CrowdFlower] Loaded {len(df)} rows from {path}")

    df["original"] = df["content"]  # Keep original for debugging
    df["content"] = df["content"].apply(clean_text, extended=True)

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

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) # Convert text to numerical sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

    input_tensor = torch.tensor(padded_sequences)
    label_tensor = torch.tensor(labels)

    dataset = TensorDataset(input_tensor, label_tensor)
    print(f"[CrowdFlower] Final FILTERED dataset shape: {len(dataset)} samples")
    print(f"[Final Label Mapping] {new_label_map}")

    return dataset, tokenizer

def load_crowdflower_without_glove(path, max_length=128, min_samples=1000):
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
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

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

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_glove", action="store_true", help="Use GloVe embeddings")
    args = parser.parse_args()
    with_glove = args.with_glove

    os.makedirs("data", exist_ok=True)

    print("[Main] Loading and processing CrowdFlower dataset...")

    if with_glove:
        crowdflower_dataset, tokenizer = load_crowdflower_with_glove(CROWDFLOWER_PATH)
        load_glove_embeddings(tokenizer, CROWDFLOWER_GLOVE_EMBEDDINGS_PATH)

        print("[Main] Splitting dataset into train and test...")
        train_ds, test_ds = split_dataset(crowdflower_dataset, split_ratio=0.8, glove=True)

        print("[Main] Saving datasets to disk...")
        torch.save(train_ds, CROWDFLOWER_TRAIN_DS_PATH_WITH_GLOVE)
        torch.save(test_ds, CROWDFLOWER_TEST_DS_PATH_WITH_GLOVE)

    else:
        crowdflower_dataset = load_crowdflower_without_glove(CROWDFLOWER_PATH)

        print("[Main] Splitting dataset into train and test...")
        train_ds, test_ds = split_dataset(crowdflower_dataset, split_ratio=0.8, glove=False)
    
        print("[Main] Saving datasets to disk...")
        torch.save(train_ds, CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE)
        torch.save(test_ds, CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE)
    
    print("[Main] Done.")
