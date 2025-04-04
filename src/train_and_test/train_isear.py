import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from utils import SupConLoss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from preprocess.preprocess_ISEAR import DualViewDataset
# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 1024
num_emotions = 7
batch_size = 128
num_epochs = 10
learning_rate = 6e-5
num_runs = 5

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(encoder, classifier, dataloader, device):
    encoder.eval()
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, _, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids, labels = input_ids.to(device), labels.to(device)
            embeddings = encoder(input_ids)
            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, recall, f1

test_acc_list = []
test_recall_list = []
test_precision_list = []
test_f1_list = []

for run in range(num_runs):
    print(f"\n🔁 Run {run+1}/{num_runs}")
    set_seed(42 + run)

    train_ds = torch.load('data/preprocessed_dataset/isear/train.pt', weights_only=False)
    test_ds = torch.load('data/preprocessed_dataset/isear/test.pt', weights_only=False)

    train_len = int(0.90 * len(train_ds))
    val_len = len(train_ds) - train_len
    train_subset, val_subset = random_split(
        train_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42 + run)
    )

    # Apply augmentation ONLY to training subset
    train_ds = DualViewDataset(train_subset, dropout_prob=0.1)
    val_ds = val_subset  # Keep validation data original

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    mamba_args = dict(d_model=2048, d_state=256, d_conv=4, expand=2)
    encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
    classifier = ClassifierHead(embed_dim, num_emotions).to(device)

    checkpoint = torch.load('results/mamba/contrastive_mamba_decoupled.pt')
    encoder.load_state_dict(checkpoint['encoder'])

    criterion_cls = CrossEntropyLoss()
    criterion_contrastive = SupConLoss()
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

    best_val_f1 = 0.0
    patience, trigger_times = 3, 0

    for epoch in range(num_epochs):
        encoder.train()
        classifier.train()
        total_loss = 0

        for view1, view2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            emb1 = encoder(view1)
            emb2 = encoder(view2)
            features = torch.stack([emb1, emb2], dim=1)

            loss_cls = criterion_cls(classifier(emb1), labels)
            loss_contrastive = criterion_contrastive(features, labels)
            loss = 0.9 * loss_cls + 0.1 * loss_contrastive 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_acc, val_recall, val_f1 = evaluate(encoder, classifier, val_loader, device)
        print(f"[Epoch {epoch+1}] Val F1: {val_f1:.4f}")

        # Save model based on best val F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_encoder = encoder.state_dict()
            best_classifier = classifier.state_dict()
            torch.save({
                'encoder': best_encoder,
                'classifier': best_classifier,
                'val_f1': best_val_f1,
                'epoch': epoch + 1
            }, 'results/mamba/mamba_isear.pt')
            trigger_times = 0
            print(f"✅ Best model saved at epoch {epoch+1} with val F1: {val_f1:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1} due to no improvement in val F1")
                break

    encoder.load_state_dict(best_encoder)
    classifier.load_state_dict(best_classifier)
    test_acc, test_recall, test_f1 = evaluate(encoder, classifier, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    test_acc_list.append(test_acc)
    test_recall_list.append(test_recall)
    test_f1_list.append(test_f1)

mean_acc = np.mean(test_acc_list)
std_acc = np.std(test_acc_list)
mean_recall = np.mean(test_recall_list)
std_recall = np.std(test_recall_list)
mean_f1 = np.mean(test_f1_list)
std_f1 = np.std(test_f1_list)

print(f"\n📊 Final Test Accuracy over {num_runs} runs: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"📊 Final Test Recall over {num_runs} runs: {mean_recall:.4f} ± {std_recall:.4f}")
print(f"📊 Final Test F1 Score over {num_runs} runs: {mean_f1:.4f} ± {std_f1:.4f}")
