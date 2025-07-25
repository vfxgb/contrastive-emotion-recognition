import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
import os
from utils import (
    clean_text,
    fetch_label_mapping,
    load_glove_embeddings,
    split_dataset,
)
from config import (
    BERT_MODEL,
    ISEAR_PATH,
    ISEAR_TEST_DS_PATH_WITHOUT_GLOVE,
    ISEAR_TRAIN_DS_PATH_WITHOUT_GLOVE,
    ISEAR_TRAIN_DS_PATH_WITH_GLOVE,
    ISEAR_TEST_DS_PATH_WITH_GLOVE,
    ISEAR_GLOVE_EMBEDDINGS_PATH,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse

# fetch label mapping for ISEAR dataset
label_mapping = fetch_label_mapping(isear=True)


def load_isear_with_glove(path, max_length=128):
    """
    Load and preprocess the ISEAR dataset from a CSV file for use with GloVe embeddings.

    Assumes the CSV has columns 'Field1' and 'SIT'.
    'Field1' is used as the label and 'SIT' as the text.
    Only rows with labels present in label_mapping are kept.

    Args:
        path (str): Path to the CSV file.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        TensorDataset: Dataset object containing input_ids and labels.
        Tokenizer: Fitted Keras Tokeniser object

    """
    # Initialize the BERT tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")

    # Load CSV using latin1 encoding and comma separator
    df = pd.read_csv(path, encoding="latin1", sep=",")
    df = df.rename(columns={"Field1": "Emotion", "SIT": "Text"})

    # Filter to keep only rows with desired emotions
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)

    # Clean the text in the 'Text' column
    df["content"] = df["Text"].apply(clean_text, extended=True)
    print("Sample cleaned text:", df["content"].iloc[0])

    texts = df["content"].tolist()

    # Convert text to numerical sequences
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(
        sequences, maxlen=max_length, padding="post", truncating="post"
    )

    # Map emotion labels to integers
    input_tensor = torch.tensor(padded_sequences)
    labels = torch.tensor(df["Emotion"].map(label_mapping).values)

    # Create and return a TensorDataset
    dataset = TensorDataset(input_tensor, labels)
    print(f"[Isear] Loaded crowdflower dataset: {len(dataset)} samples from {path}")
    print(f"[Isear] Label map: {label_mapping}")

    return dataset, tokenizer


def load_isear_without_glove(path, max_length=128):
    """
    Load and preprocess the ISEAR dataset from a CSV file.
    Assumes the CSV has columns 'Field1' and 'SIT'.
    'Field1' is used as the label and 'SIT' as the text.
    Only rows with labels present in label_mapping are kept.

    Args:
        path (str): Path to the CSV file.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        TensorDataset: Dataset object containing input_ids, attention_masks, and labels.

    """
    # Initialize the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    # Load CSV using latin1 encoding and comma separator
    df = pd.read_csv(path, encoding="latin1", sep=",")
    df = df.rename(columns={"Field1": "Emotion", "SIT": "Text"})

    # Filter to keep only rows with desired emotions
    df = df[df["Emotion"].isin(label_mapping.keys())].reset_index(drop=True)

    # Clean the text in the 'Text' column
    df["content"] = df["Text"].apply(clean_text)

    # Tokenize the cleaned text using the BERT tokenizer
    encodings = tokenizer(
        df["content"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    # Map emotion labels to integers
    labels = torch.tensor(df["Emotion"].map(label_mapping).values)

    # Create and return a TensorDataset
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    print(f"[Isear] Loaded crowdflower dataset: {len(dataset)} samples from {path}")
    print(f"[Isear] Label map: {label_mapping}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # when --with_glove is set, the dataset is preprocessed for use with GloVe embeddings.
    parser.add_argument(
        "--with_glove", action="store_true", help="Use GloVe embeddings"
    )

    # when --force_preprocess is set, the dataset is processed again even when the train.pt and test.pt are present
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Force reprocessing even if files exist",
    )

    args = parser.parse_args()
    with_glove = args.with_glove
    force_preprocess = args.force_preprocess

    # create preprocessing directory
    os.makedirs("data/preprocessed_dataset/isear/", exist_ok=True)

    print("[Main] Loading and processing ISEAR dataset...")

    if with_glove:
        if (
            not force_preprocess
            and os.path.exists(ISEAR_TRAIN_DS_PATH_WITH_GLOVE)
            and os.path.exists(ISEAR_TEST_DS_PATH_WITH_GLOVE)
        ):
            print("[Main] Dataset already preprocessed. Skipping...")
        else:
            print("[Main] Preprocessing Dataset.")
            isear_dataset, tokenizer = load_isear_with_glove(ISEAR_PATH, max_length=128)

            print("[Main] Loading glove embeddings.")
            load_glove_embeddings(tokenizer, ISEAR_GLOVE_EMBEDDINGS_PATH)

            print("[Main] Splitting dataset into train and test...")
            train_ds, test_ds = split_dataset(
                isear_dataset, split_ratio=0.8, glove=True
            )

            print("[Main] Saving datasets to disk...")
            torch.save(train_ds, ISEAR_TRAIN_DS_PATH_WITH_GLOVE)
            torch.save(test_ds, ISEAR_TEST_DS_PATH_WITH_GLOVE)

    else:
        if (
            not force_preprocess
            and os.path.exists(ISEAR_TRAIN_DS_PATH_WITHOUT_GLOVE)
            and os.path.exists(ISEAR_TEST_DS_PATH_WITHOUT_GLOVE)
        ):
            print("[Main] Dataset already preprocessed. Skipping...")
        else:
            print("[Main] Preprocessing Dataset.")
            isear_dataset = load_isear_without_glove(ISEAR_PATH, max_length=128)

            print("[Main] Splitting dataset into train and test...")
            train_ds, test_ds = split_dataset(
                isear_dataset, split_ratio=0.8, glove=False
            )

            print("[Main] Saving datasets to disk...")
            torch.save(train_ds, ISEAR_TRAIN_DS_PATH_WITHOUT_GLOVE)
            torch.save(test_ds, ISEAR_TEST_DS_PATH_WITHOUT_GLOVE)

    print("[Main] Done.")
