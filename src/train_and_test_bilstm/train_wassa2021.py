import re
import string
import numpy as np
import spacy
import sys
sys.path.append("/home/UG/bhargavi005/contrastive-emotion-recognition")
# from tensorflow import keras
# from tensorflow.keras.preprocessing.text import Tokenizer
# from transformers import BertTokenizer, BertModel
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score
# from sklearn.model_selection import train_test_split
# from models.bilstm_model import BiLSTM
from src.models.bilstm_model import BiLSTM
from src.utils import set_seed
from src.config import bilstm_config
torch.serialization.add_safe_globals([TensorDataset])


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = [] 
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc="Validation"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            logits = model(input_ids, attention_mask)

            _, predicted = logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    print("\nDetailed Report:\n", classification_report(all_labels, all_preds))

    return accuracy, f1, recall, precision

def load_and_adapt_model(pretrained_model_path, num_classes, model_config):
    pretrained_state_dict = torch.load(pretrained_model_path, map_location=device)

    # Create a new model instance
    new_model = BiLSTM(
        bert_model_name=model_config["bert_model_name"],
        hidden_dim=model_config["hidden_dim"],
        num_classes=num_classes,
        dropout_rate=model_config["dropout_rate"], 
        lstm_layers=model_config["lstm_layers"],
    )

    # load parameters from pretrained model
    model_dict = new_model.state_dict()

    # filter out final classification layer
    pretrained_state_dict = {k:v for k,v in pretrained_state_dict.items() if k in model_dict and 'fc3' not in k}
    model_dict.update(pretrained_state_dict)
    new_model.load_state_dict(model_dict)
    
    return new_model

# Configurations
model_config = bilstm_config()

num_classes = 6  # For WASSA: anger, sadness, disgust, fear, joy, surprise
num_epochs = model_config["num_epochs"]
learning_rate = model_config["learning_rate"]
# batch_size = model_config["batch_size"]
batch_size = 32
device = model_config["device"]
isear_finetune_save_path = model_config["wassa21_finetune_save_path"]

best_accuracy = 0
trigger_times = 0
patience = 5
num_runs = 5
test_acc_list = []
test_recall_list = []
test_precision_list = []
test_f1_list = []


for run in range(num_runs):
    print(f"\n🔁 Run {run+1}/{num_runs}")
    set_seed(42 + run)

    train_ds = torch.load('data/preprocessed_dataset/wassa/train.pt', weights_only=False)
    test_ds = torch.load('data/preprocessed_dataset/wassa/test.pt', weights_only=False)

    train_len = int(0.90 * len(train_ds))
    val_len = len(train_ds) - train_len
    train_ds, val_ds = random_split(
        train_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42 + run)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # initialise model
    model = load_and_adapt_model(model_config["model_save_path"], num_classes = num_classes, model_config = model_config)

    # freeze bert and lstm layers and only train the final classification layer
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.lstm.parameters():
        param.requires_grad = True
    for param in model.fc1.parameters():
        param.requires_grad = True
    for param in model.fc2.parameters():
        param.requires_grad = True
    for param in model.fc3.parameters():
        param.requires_grad = True

    model.to(device)

    # initialse loss function
    criterion = nn.CrossEntropyLoss()

    # initialise optimiser
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_ids, attention_mask, labels in tqdm(train_loader,desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()            

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

        val_accuracy, _, _ ,_ = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy :
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), isear_finetune_save_path)  
            trigger_times = 0 
            print(f"Best model saved at cepoch {epoch+1} with accuracy: {val_accuracy:.4f}")
        else:
            trigger_times += 1 
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
    print("\n----- Starting Evaluation on Test Set -----\n")
    test_accuracy, test_f1, test_recall, test_precision = evaluate(model, test_loader, device)
    
    test_acc_list.append(test_accuracy)
    test_recall_list.append(test_recall)
    test_precision_list.append(test_precision)
    test_f1_list.append(test_f1)

mean_test_acc = np.mean(test_acc_list)
std_test_acc = np.std(test_acc_list)
mean_test_recall = np.mean(test_recall_list)
std_test_recall = np.std(test_recall_list)
mean_test_precision = np.mean(test_precision_list)
std_test_precision = np.std(test_precision_list)
mean_test_f1 = np.mean(test_f1_list)
std_test_f1 = np.std(test_f1_list)

print(f"\nFinal Test Accuracy over {num_runs} runs: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
print(f"Final Test Recall over {num_runs} runs: {mean_test_recall:.4f} ± {std_test_recall:.4f}")
print(f"Final Test Precision over {num_runs} runs: {mean_test_precision:.4f} ± {std_test_precision:.4f}")
print(f"Final Test F1 Score over {num_runs} runs: {mean_test_f1:.4f} ± {std_test_f1:.4f}")