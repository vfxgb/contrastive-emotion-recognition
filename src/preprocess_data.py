import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def load_crowdflower(path, max_length=128):
    df = pd.read_csv(path)
    texts = df['content'].tolist()
    labels = pd.factorize(df['sentiment'])[0]

    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    return TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(labels))

def load_wassa(path, emotions, max_length=128):
    texts, labels = [], []
    for idx, emotion in enumerate(emotions):
        with open(f"{path}/{emotion}", "r") as file:
            lines = file.readlines()
            texts.extend(lines)
            labels.extend([idx] * len(lines))
    
    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    return TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(labels))

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
