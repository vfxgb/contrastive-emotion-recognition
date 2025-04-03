import torch
from torch.utils.data import DataLoader, TensorDataset
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from utils import SupConLoss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from preprocess.preprocess_crowdflower import load_crowdflower, split_dataset, DualViewDataset
from torch.utils.data import DataLoader, random_split

def main():
    # Configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 1024
    num_emotions = 9 
    batch_size = 128
    num_epochs = 10
    learning_rate = 6e-5
    patience = 3
    model_save_path = 'results/mamba/contrastive_mamba_decoupled.pt'
    torch.serialization.add_safe_globals([TensorDataset])

    # Load datasets
    print("Loading training data...")
    train_ds = torch.load('data/preprocessed_dataset/crowdflower/train.pt', weights_only=False)
    test_ds = torch.load('data/preprocessed_dataset/crowdflower/test.pt', weights_only=False)

    train_len = int(0.90 * len(train_ds))
    val_len = len(train_ds) - train_len
    train_subset, val_subset = random_split(
        train_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = DualViewDataset(train_subset, dropout_prob=0.1)
    val_ds = val_subset
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Mamba config
    mamba_args = dict(
        d_model=2048,
        d_state=256,
        d_conv=4,
        expand=2,
    )
    
    # Initialize encoder & classifier for training
    print("Initializing models...")
    encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
    classifier = ClassifierHead(embed_dim, num_emotions).to(device)
    
    # Losses & optimizer
    criterion_cls = CrossEntropyLoss()
    criterion_contrastive = SupConLoss()
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    
    # Training loop with validation
    print("Starting training...")
    best_accuracy = 0
    trigger_times = 0
    
    for epoch in range(num_epochs):
        encoder.train()
        classifier.train()
        total_loss = 0
        
        for view1, view2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            
            emb1, emb2 = encoder(view1), encoder(view2)
            features = torch.stack([emb1, emb2], dim=1)
            
            loss = 0.9*criterion_cls(classifier(emb1), labels) + 0.1*criterion_contrastive(features, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")
        
        # Validation step
        val_accuracy = evaluate(encoder, classifier, val_loader, device)
        print(f"[Epoch {epoch+1}] Validation Accuracy: {val_accuracy:.4f}")
        
        # Early stopping logic
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict(),
            }, model_save_path)
            trigger_times = 0
            print(f"Best model saved at epoch {epoch+1} with accuracy: {val_accuracy:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluation on test set
    print("\n----- Starting Evaluation on Test Set -----\n")
    
    # Load test dataset
    print("Loading test data...")
    test_dataset = torch.load('data/preprocessed_dataset/crowdflower/test.pt', weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize test model with new classifier head for test emotions
    test_encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
    test_classifier = ClassifierHead(embed_dim, num_emotions).to(device)
    
    # Load model weights
    print("Loading best model weights...")
    checkpoint = torch.load(model_save_path, map_location=device)
    test_encoder.load_state_dict(checkpoint['encoder'])
    
    checkpoint_test = torch.load('results/mamba/contrastive_mamba_decoupled.pt', map_location=device)
    test_classifier.load_state_dict(checkpoint_test['classifier'])
    
    # Evaluation loop
    test_encoder.eval()
    test_classifier.eval()
    
    preds, truths = [], []
    with torch.no_grad():
        for input_ids, _, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            emotion_emb = test_encoder(input_ids)
            logits = test_classifier(emotion_emb)
            predictions = logits.argmax(dim=-1)
            
            preds.extend(predictions.cpu().numpy())
            truths.extend(labels.cpu().numpy())
    
    # Final metrics
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='weighted')
    print(f"\n✅ Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(truths, preds))

def evaluate(encoder, classifier, dataloader, device):
    """Evaluate model on dataloader"""
    encoder.eval()
    classifier.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, _, labels in tqdm(dataloader, desc="Validation"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            embeddings = encoder(input_ids)
            logits = classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    main()
