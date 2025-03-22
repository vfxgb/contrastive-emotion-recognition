# import pandas as pd
# import torch
# from transformers import AutoTokenizer
# from torch.utils.data import TensorDataset

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# def load_crowdflower(path, max_length=128):
#     df = pd.read_csv(path)
#     texts = df['content'].tolist()
#     labels = pd.factorize(df['sentiment'])[0]

#     encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
#     return TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(labels))

# def load_wassa(path, emotions, max_length=128):
#     texts, labels = [], []
#     for idx, emotion in enumerate(emotions):
#         with open(f"{path}/{emotion}", "r") as file:
#             lines = file.readlines()
#             texts.extend(lines)
#             labels.extend([idx] * len(lines))
    
#     encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
#     return TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(labels))

# if __name__ == "__main__":
#     train_dataset = load_crowdflower('../data/CrowdFlower/text_emotion.csv')
#     test_dataset = load_wassa('../data/WASSA2017/testing', [
#         'anger-ratings-0to1.test.target.txt',
#         'fear-ratings-0to1.test.target.txt',
#         'joy-ratings-0to1.test.target.txt',
#         'sadness-ratings-0to1.test.target.txt'
#     ])

#     torch.save(train_dataset, '../data/train.pt')
#     torch.save(test_dataset, '../data/test.pt')

# import pandas as pd
# import torch
# from transformers import AutoTokenizer
# from torch.utils.data import TensorDataset
# import re
# from sklearn.model_selection import train_test_split

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# def clean_text(text):
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'@\w+', '', text)
#     text = re.sub(r'#', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text.lower()

# def load_crowdflower(path, max_length=128):
#     df = pd.read_csv(path)
#     df['content'] = df['content'].apply(clean_text)

#     label_map = {label: idx for idx, label in enumerate(sorted(df['sentiment'].unique()))}
#     labels = df['sentiment'].map(label_map).tolist()
#     print(f"labels in crowdflower : {labels}")

#     encodings = tokenizer(
#         df['content'].tolist(),
#         truncation=True,
#         padding='max_length',
#         max_length=max_length,
#         return_tensors='pt'
#     )

#     input_ids = encodings['input_ids']
#     attention_masks = encodings['attention_mask']

#     return TensorDataset(input_ids, attention_masks, torch.tensor(labels))

# def load_wassa(path, emotions, max_length=128):
#     texts, labels = [], []
#     for idx, emotion in enumerate(emotions):
#         with open(f"{path}/{emotion}", "r") as file:
#             lines = [clean_text(line.strip()) for line in file]
#             texts.extend(lines)
#             labels.extend([idx] * len(lines))
#             print(f"wassa : {labels}")
    
#     encodings = tokenizer(
#         texts, truncation=True, padding='max_length', 
#         max_length=max_length, return_tensors='pt'
#     )

#     input_ids = encodings['input_ids']
#     attention_masks = encodings['attention_mask']

#     return TensorDataset(input_ids, attention_masks, torch.tensor(labels))

# if __name__ == "__main__":
#     train_dataset = load_crowdflower('../data/CrowdFlower/text_emotion.csv')

#     test_dataset = load_wassa('../data/WASSA2017/testing', [
#         'anger-ratings-0to1.test.target.txt',
#         'fear-ratings-0to1.test.target.txt',
#         'joy-ratings-0to1.test.target.txt',
#         'sadness-ratings-0to1.test.target.txt'
#     ])

#     torch.save(train_dataset, '../data/train.pt')
#     torch.save(test_dataset, '../data/test.pt')

import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
import re
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def load_crowdflower(path, max_length=128):
    # Load CSV file using pandas (header row is automatically handled)
    df = pd.read_csv(path)
    print(f"[CrowdFlower] Loaded {len(df)} rows from {path}")

    # Clean text
    df['content'] = df['content'].apply(clean_text)
    
    # Create a label map and print it for debugging
    unique_sentiments = sorted(df['sentiment'].unique())
    label_map = {label: idx for idx, label in enumerate(unique_sentiments)}
    print(f"[CrowdFlower] Label mapping: {label_map}")

    labels = df['sentiment'].map(label_map).tolist()
    print(f"[CrowdFlower] First 10 mapped labels: {labels[:10]}")

    # Tokenize texts
    texts = df['content'].tolist()
    print(f"[CrowdFlower] First text sample: {texts[0]}")
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']

    print(f"[CrowdFlower] Tokenized input_ids shape: {input_ids.shape}")
    print(f"[CrowdFlower] Tokenized attention_masks shape: {attention_masks.shape}")
    
    # Create TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, torch.tensor(labels))
    print(f"[CrowdFlower] Final dataset length: {len(dataset)}")
    print(f"[CrowdFlower] Unique labels in dataset: {torch.unique(torch.tensor(labels))}")
    return dataset

def load_wassa(path, emotions, max_length=128):
    texts, labels = [], []
    for idx, emotion in enumerate(emotions):
        file_path = f"{path}/{emotion}"
        print(f"[WASSA] Loading file: {file_path}")
        with open(file_path, "r") as file:
            # Read all lines and skip header (assume first line is header)
            all_lines = file.readlines()
            if all_lines:
                # If header exists, skip it
                header = all_lines[0].strip()
                print(f"[WASSA] Detected header: {header}")
                lines = all_lines[1:]
            else:
                lines = []
            # Clean each line and extend texts and labels
            cleaned_lines = [clean_text(line.strip()) for line in lines if line.strip()]
            texts.extend(cleaned_lines)
            labels.extend([idx] * len(cleaned_lines))
            print(f"[WASSA] Loaded {len(cleaned_lines)} lines for emotion index {idx}")
    
    print(f"[WASSA] Total texts loaded: {len(texts)}")
    print(f"[WASSA] Unique label values before tokenization: {set(labels)}")
    
    encodings = tokenizer(
        texts, truncation=True, padding='max_length', 
        max_length=max_length, return_tensors='pt'
    )

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    
    print(f"[WASSA] Tokenized input_ids shape: {input_ids.shape}")
    print(f"[WASSA] Tokenized attention_masks shape: {attention_masks.shape}")
    
    dataset = TensorDataset(input_ids, attention_masks, torch.tensor(labels))
    print(f"[WASSA] Final dataset length: {len(dataset)}")
    print(f"[WASSA] Unique labels in dataset: {torch.unique(torch.tensor(labels))}")
    return dataset

if __name__ == "__main__":
    print("[Main] Loading CrowdFlower dataset...")
    train_dataset = load_crowdflower('../data/CrowdFlower/text_emotion.csv')
    
    print("[Main] Loading WASSA dataset...")
    test_dataset = load_wassa('../data/WASSA2017/testing', [
        'anger-ratings-0to1.test.target.txt',
        'fear-ratings-0to1.test.target.txt',
        'joy-ratings-0to1.test.target.txt',
        'sadness-ratings-0to1.test.target.txt'
    ])
    
    print("[Main] Saving datasets...")
    torch.save(train_dataset, '../data/train.pt')
    torch.save(test_dataset, '../data/test.pt')
    print("[Main] Datasets saved.")
