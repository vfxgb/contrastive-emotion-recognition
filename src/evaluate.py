# import torch
# from torch.utils.data import DataLoader
# from models.contrastive_model import ContrastiveMambaModel
# from sklearn.metrics import accuracy_score, f1_score
# from torch.utils.data import TensorDataset
# from tqdm import tqdm
# from models.mamba import ModelArgs

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.serialization.add_safe_globals([TensorDataset])
# test_dataset = torch.load('data/test.pt')
# test_loader = DataLoader(test_dataset, batch_size=32)

# mamba_args = ModelArgs(d_model=128, n_layer=2, vocab_size=30522)
# model = ContrastiveMambaModel(mamba_args, num_emotions=13).to(device)
# model.load_state_dict(torch.load('results/contrastive_mamba.pt'))

# model.eval()
# preds, truths = [], []
# with torch.no_grad():
#     for input_ids, _, labels in tqdm(test_loader):
#         input_ids = input_ids.to(device)
#         logits, _ = model(input_ids)
#         predictions = logits.argmax(dim=-1).cpu().numpy()
#         preds.extend(predictions)
#         truths.extend(labels.numpy())

# acc = accuracy_score(truths, preds)
# f1 = f1_score(truths, preds, average='weighted')

# print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


import torch
from torch.utils.data import DataLoader, TensorDataset
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from models.mamba import ModelArgs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.serialization.add_safe_globals([TensorDataset])

# Load test dataset.
test_dataset = torch.load('data/test.pt')
test_loader = DataLoader(test_dataset, batch_size=32)

# Define model parameters.
mamba_args = ModelArgs(d_model=128, n_layer=2, vocab_size=30522)
embed_dim = 256
num_emotions = 13

# Instantiate the encoder and classifier.
encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
classifier = ClassifierHead(embed_dim, num_emotions).to(device)

# Load the decoupled model checkpoint.
checkpoint = torch.load('results/contrastive_mamba_decoupled.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
classifier.load_state_dict(checkpoint['classifier'])

encoder.eval()
classifier.eval()

preds, truths = [], []

with torch.no_grad():
    for input_ids, _, labels in tqdm(test_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        # Get embeddings from the encoder.
        emotion_emb = encoder(input_ids)
        # Get logits from the classifier head.
        logits = classifier(emotion_emb)
        # Determine predictions.
        predictions = logits.argmax(dim=-1).cpu().numpy()
        preds.extend(predictions)
        truths.extend(labels.numpy())

# Compute evaluation metrics.
acc = accuracy_score(truths, preds)
f1 = f1_score(truths, preds, average='weighted')

print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
