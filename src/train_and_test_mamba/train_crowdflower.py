from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
from config import (
    F1_AVERAGE_METRIC,
    SEED,
    mamba_config,
    CROWDFLOWER_CLASSES,
    CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE,
    CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE,
    USE_TQDM,
)
from utils import DualViewDataset, SupConLoss
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead

torch.serialization.add_safe_globals([TensorDataset])


def evaluate(encoder, classifier, dataloader, device, test=False):
    """
    Evaluates model.

    Args:
        encoder: The encoder of the model to evaluate.
        classifier: The classifier of the model to evaluate.
        dataloader : dataloader for val or test dataset.
        device : the device to perform computation on ( cuda or cpu ).
        test : whether we evaluation on train or val ds for tqdm status description.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the model on the dataset.
            - f1 (float): The F1 score (macro) of the model on the dataset.
    """
    encoder.eval()
    classifier.eval()

    all_labels = []
    all_preds = []
    desc = "Test" if test else "Validation"

    with torch.no_grad():
        for input_ids, _, labels in tqdm(dataloader, desc=desc, disable=not USE_TQDM):
            input_ids, labels = input_ids.to(device), labels.to(device)

            # get prediction
            embeddings = encoder(input_ids)
            logits = classifier(embeddings)

            _, predicted = logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=F1_AVERAGE_METRIC, zero_division=0)

    print(f"Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.4f}")
    print(
        "\Classification Report:\n",
        classification_report(all_labels, all_preds, zero_division=0),
    )

    return accuracy, f1


def main():
    # fetch bilstm model config
    model_config = mamba_config()
    print(f"\n [Main] Dataset : Crowdflower, Model Configuration : {model_config}")

    mamba_args = model_config["mamba_args"]
    device = model_config["device"]
    embed_dim = model_config["embed_dim"]
    batch_size = model_config["batch_size"]
    num_epochs = model_config["num_epochs"]
    learning_rate = model_config["learning_rate"]
    model_save_path = model_config["model_save_path"]

    # early stopping parameters
    best_val_f1 = 0
    trigger_times = 0
    patience = 5

    print("[Main] Loading training data...")
    train_ds = torch.load(CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE, weights_only=False)
    test_ds = torch.load(CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE, weights_only=False)

    train_len = int(0.90 * len(train_ds))
    val_len = len(train_ds) - train_len
    train_subset, val_subset = random_split(
        train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(SEED)
    )

    train_ds = DualViewDataset(train_subset, dropout_prob=0.1)
    val_ds = val_subset
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Initialize encoder & classifier for training
    print("[Main] Initializing models...")
    encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
    classifier = ClassifierHead(embed_dim, CROWDFLOWER_CLASSES).to(device)

    # Losses & optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_contrastive = SupConLoss()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate
    )

    print("[Main] Start training model...")
    for epoch in range(num_epochs):
        encoder.train()
        classifier.train()
        total_loss = 0

        for view1, view2, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}", disable=not USE_TQDM
        ):
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

            emb1, emb2 = encoder(view1), encoder(view2)
            features = torch.stack([emb1, emb2], dim=1)

            loss = 0.9 * criterion_cls(
                classifier(emb1), labels
            ) + 0.1 * criterion_contrastive(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}]")

        # print out evaluation metrics
        val_accuracy, val_f1 = evaluate(encoder, classifier, val_loader, device)

        # Early stopping logic
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                model_save_path,
            )
            trigger_times = 0
            print(
                f"Best model saved at epoch {epoch+1} with accuracy: {val_accuracy:.4f} and f1 score: {val_f1:.4f}"
            )
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("[Main] Start testing model...")
    checkpoint = torch.load(model_save_path, map_location=device)

    # Initialize test model with new classifier head for test emotions
    test_encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
    test_classifier = ClassifierHead(embed_dim, CROWDFLOWER_CLASSES).to(device)

    test_encoder.load_state_dict(checkpoint["encoder"])
    test_classifier.load_state_dict(checkpoint["classifier"])

    # print results on the test set
    evaluate(test_encoder, test_classifier, test_loader, device=device, test=True)


if __name__ == "__main__":
    main()
