from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
)
import numpy as np
from models.bilstm_model import BiLSTM_without_glove
from utils import set_seed, DualViewDataset, SupConLoss
from config import (
    WASSA_CLASSES,
    WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE,
    WASSA_TEST_DS_PATH_WITHOUT_GLOVE,
    mamba_config,
    F1_AVERAGE_METRIC,
)
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead

torch.serialization.add_safe_globals([TensorDataset])


def evaluate(encoder, classifier, dataloader, device, test=False):
    """
    Evaluates model.

    Args:
        model : The model to evaluate
        dataloader : dataloader for val or test dataset
        device : the device to perform computation on. ( cuda or cpu )
        test : whether evaluting on test or train ds.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the model on the dataset.
            - f1 (float): The F1 score (macro) of the model on the dataset.
            - recall (float): The recall (macro) of the model on the dataset.
            - precision (float): The precision (macro) of the model on the dataset.
    """
    encoder.eval()
    classifier.eval()

    all_labels = []
    all_preds = []
    desc = "Test" if test else "Validation"

    with torch.no_grad():
        for input_ids, _, labels in tqdm(dataloader, desc=desc):
            input_ids, labels = input_ids.to(device), labels.to(device)

            embeddings = encoder(input_ids)
            logits = classifier(embeddings)

            _, predicted = logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=F1_AVERAGE_METRIC)
    recall = recall_score(all_labels, all_preds, average=F1_AVERAGE_METRIC)
    precision = precision_score(all_labels, all_preds, average=F1_AVERAGE_METRIC)

    print(
        f"Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {predicted:.4f}"
    )
    print("\nDetailed Report:\n", classification_report(all_labels, all_preds))

    return accuracy, f1, recall, precision


def main():
    # Configurations
    model_config = mamba_config()

    num_classes = WASSA_CLASSES
    mamba_args = model_config["mamba_args"]
    device = model_config["device"]
    embed_dim = model_config["embed_dim"]
    batch_size = model_config["batch_size"]
    num_epochs = model_config["num_epochs"]
    learning_rate = model_config["learning_rate"]
    model_save_path = model_config["model_save_path"]
    wassa21_finetune_save_path = model_config["wassa21_finetune_save_path"]

    best_f1 = 0
    trigger_times = 0
    patience = 3
    num_runs = 5

    test_acc_list = []
    test_recall_list = []
    test_precision_list = []
    test_f1_list = []

    for run in range(num_runs):
        print(f"\n🔁 Run {run+1}/{num_runs}")
        set_seed(42 + run)

        train_ds = torch.load(WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE, weights_only=False)
        test_ds = torch.load(WASSA_TEST_DS_PATH_WITHOUT_GLOVE, weights_only=False)

        train_len = int(0.90 * len(train_ds))
        val_len = len(train_ds) - train_len
        train_subset, val_subset = random_split(
            train_ds,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42 + run),
        )

        # Apply augmentation ONLY to training subset
        train_ds = DualViewDataset(train_subset, dropout_prob=0.1)
        val_ds = val_subset  # Keep validation data original

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        encoder = ContrastiveMambaEncoder(mamba_args, embed_dim=embed_dim).to(device)
        classifier = ClassifierHead(embed_dim, num_classes).to(device)

        checkpoint = torch.load(model_save_path)
        encoder.load_state_dict(checkpoint["encoder"])

        criterion_cls = nn.CrossEntropyLoss()
        criterion_contrastive = SupConLoss()
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate
        )

        for epoch in range(num_epochs):
            encoder.train()
            classifier.train()
            total_loss = 0

            for view1, view2, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}", leave=False
            ):
                view1, view2, labels = (
                    view1.to(device),
                    view2.to(device),
                    labels.to(device),
                )
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

            # Validation step
            val_accuracy, val_f1, val_recall, val_precision = evaluate(
                encoder, classifier, val_loader, device
            )
            print(
                f"[Epoch {epoch+1}] Validation Accuracy: {val_accuracy:.4f} Val F1 Macro: {val_f1:.4f}"
            )

            # Save model based on best val F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_encoder = encoder.state_dict()
                best_classifier = classifier.state_dict()
                torch.save(
                    {
                        "encoder": best_encoder,
                        "classifier": best_classifier,
                        "val_f1": best_f1,
                        "epoch": epoch + 1,
                    },
                    wassa21_finetune_save_path,
                )
                trigger_times = 0
                print(f"Best model saved at epoch {epoch+1} with val F1: {val_f1:.4f}")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1} due to no improvement in val F1"
                    )
                    break

        print("\n----- Starting Evaluation on Test Set -----\n")
        encoder.load_state_dict(best_encoder)
        classifier.load_state_dict(best_classifier)
        test_accuracy, test_f1, test_recall, test_precision = evaluate(
            encoder, classifier, test_loader, device, test=True
        )

        test_acc_list.append(test_accuracy)
        test_recall_list.append(test_recall)
        test_precision_list.append(test_precision)
        test_f1_list.append(test_f1)

    mean_test_acc, std_test_acc = np.mean(test_acc_list), np.std(test_acc_list)
    mean_test_recall, std_test_recall = np.mean(test_recall_list), np.std(test_recall_list)
    mean_test_precision, std_test_precision = np.mean(test_precision_list), np.std(test_precision_list)
    mean_test_f1, std_test_f1 = np.mean(test_f1_list), np.std(test_f1_list)

    print(f"\nFinal Test Accuracy over {num_runs} runs: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"Final Test Recall over {num_runs} runs: {mean_test_recall:.4f} ± {std_test_recall:.4f}")
    print(f"Final Test Precision over {num_runs} runs: {mean_test_precision:.4f} ± {std_test_precision:.4f}")
    print(f"Final Test F1 Score over {num_runs} runs: {mean_test_f1:.4f} ± {std_test_f1:.4f}")

if __name__ == "__main__":
    main()
