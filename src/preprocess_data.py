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
    df = pd.read_csv(path)
    df['content'] = df['content'].apply(clean_text)

    label_map = {label: idx for idx, label in enumerate(sorted(df['sentiment'].unique()))}
    labels = df['sentiment'].map(label_map).tolist()

    encodings = tokenizer(
        df['content'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']

    return TensorDataset(input_ids, attention_masks, torch.tensor(labels))

def load_wassa(path, emotions, max_length=128):
    texts, labels = [], []
    for idx, emotion in enumerate(emotions):
        with open(f"{path}/{emotion}", "r") as file:
            lines = [clean_text(line.strip()) for line in file]
            texts.extend(lines)
            labels.extend([idx] * len(lines))
    
    encodings = tokenizer(
        texts, truncation=True, padding='max_length', 
        max_length=max_length, return_tensors='pt'
    )

    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']

    return TensorDataset(input_ids, attention_masks, torch.tensor(labels))

if __name__ == "__main__":
    train_dataset = load_crowdflower('../data/CrowdFlower/text_emotion.csv')

    test_dataset = load_wassa('../data/WASSA2017/testing', [
        'anger-ratings-0to1.test.target.txt',
        'fear-ratings-0to1.test.target.txt',
        'joy-ratings-0to1.test.target.txt',
        'sadness-ratings-0to1.test.target.txt'
    ])

    torch.save(train_dataset, '../data/train.pt')
    torch.save(test_dataset, '../data/test.pt')
