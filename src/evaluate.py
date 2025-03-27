import torch
from torch.utils.data import DataLoader, TensorDataset
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.serialization.add_safe_globals([TensorDataset])

# Load test dataset
test_dataset = torch.load('data/test.pt')
test_loader = DataLoader(test_dataset, batch_size=32)

# Mamba config
mamba_args = dict(d_model=128, d_state=64, d_conv=4, expand=2)
embed_dim = 256
num_emotions = 13

# Initialize model
encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
classifier = ClassifierHead(embed_dim, num_emotions).to(device)

# Load model weights
checkpoint = torch.load('results/contrastive_mamba_decoupled.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
classifier.load_state_dict(checkpoint['classifier'])

encoder.eval()
classifier.eval()

# Evaluation loop
preds, truths = [], []
with torch.no_grad():
    for input_ids, _, labels in tqdm(test_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass
        emotion_emb = encoder(input_ids)
        logits = classifier(emotion_emb)
        predictions = logits.argmax(dim=-1)

        preds.extend(predictions.cpu().numpy())
        truths.extend(labels.cpu().numpy())

# Final metrics
acc = accuracy_score(truths, preds)
f1 = f1_score(truths, preds, average='weighted')
print(f"✅ Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
print("\nDetailed Report:\n", classification_report(truths, preds))
