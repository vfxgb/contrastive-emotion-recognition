import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
import os
from utils import clean_text, fetch_label_mapping, split_dataset
from config import BERT_MODEL, WASSA_PATH, WASSA_TEST_DS_PATH, WASSA_TRAIN_DS_PATH

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

# get label mapping for WASSA dataset
label_mapping = fetch_label_mapping(wassa=True)


def load_wassa(tsv_path, max_length=128):
    """
    Load and preprocess the WASSA 2021 dataset from a TSV file.
    Renames 'emotion_labels' to 'Emotion' and 'essays' to 'Text'.
    Filters to keep only rows with one of the six Ekman emotions,
    cleans the text, tokenizes it, and maps the labels.
    Returns a TensorDataset.
    """
    print("Loading TSV from:", tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")
    print("Original dataset shape:", df.shape)
    print("Columns in dataset:", df.columns.tolist())

    # Rename columns for consistency
    df = df.rename(columns={"emotion_label": "Emotion", "essay": "Text"})
    print("Renamed columns: 'emotion_labels' -> 'Emotion', 'essays' -> 'Text'")

    # Show original label distribution
    print("Original label distribution in 'Emotion':")
    print(df["Emotion"].value_counts())

    # Filter to keep only rows with desired emotions (exclude others such as 'neutral')
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)
    print("Filtered dataset shape:", df.shape)
    print("Filtered label distribution:")
    print(df["Emotion"].value_counts())

    # Clean the text in the 'Text' column
    df["content"] = df["Text"].apply(clean_text)
    print("Sample cleaned text:", df["content"].iloc[0])

    # Tokenize the cleaned text using the BERT tokenizer
    encodings = tokenizer(
        df["content"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    print("Tokenization complete.")
    print("Input IDs shape:", encodings["input_ids"].shape)
    print("Attention mask shape:", encodings["attention_mask"].shape)

    # Map emotion labels to integers using the defined mapping
    labels = torch.tensor(df["Emotion"].map(label_mapping).values)
    print("Labels tensor shape:", labels.shape)
    print("Unique label mapping:", label_mapping)

    # Create and return a TensorDataset
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    print(f"WASSA 2021 dataset loaded: {len(dataset)} samples")
    return dataset


# Check label distribution in train and test splits
def _get_label_distribution(subset):
    labels_list = [wassa_dataset[idx][2].item() for idx in subset.indices]
    return pd.Series(labels_list).value_counts().sort_index()


if __name__ == "__main__":
    # Ensure the data folder exists
    os.makedirs("data", exist_ok=True)

    print("[Main] Loading and processing WASSA 2021 dataset...")

    # Load the dataset
    wassa_dataset = load_wassa(WASSA_PATH, max_length=128)

    # Debug: print a sample from the dataset
    sample_idx = 0
    sample_input_ids, sample_attention, sample_label = wassa_dataset[sample_idx]
    print("\nSample from dataset:")
    print("Input IDs:", sample_input_ids.tolist())
    print("Attention mask:", sample_attention.tolist())
    print("Label:", sample_label.item())

    total_samples = len(wassa_dataset)
    wassa_train, wassa_test = split_dataset(wassa_dataset, split_ratio=0.8)
    print("\nTrain/Test split:")
    print("Total samples:", total_samples)
    print("Train size:", len(wassa_train))
    print("Test size:", len(wassa_test))

    print("\nTrain label distribution:")
    print(_get_label_distribution(wassa_train))
    print("\nTest label distribution:")
    print(_get_label_distribution(wassa_test))

    # Save final datasets
    torch.save(wassa_train, WASSA_TRAIN_DS_PATH)
    torch.save(wassa_test, WASSA_TEST_DS_PATH)

    print("\nDatasets prepared and saved:")
    print(f"- WASSA 2021 train: {len(wassa_train)} samples")
    print(f"- WASSA 2021 test: {len(wassa_test)} samples")
