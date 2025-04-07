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
from models.bilstm_model import BiLSTM_bert
from config import (
    F1_AVERAGE_METRIC,
    bilstm_without_glove_config,
    CROWDFLOWER_CLASSES,
    CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE,
    CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE,
)

torch.serialization.add_safe_globals([TensorDataset])


def evaluate(model, dataloader, device, test=False):
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
    model.eval()
    all_labels = []
    all_preds = []
    desc = "Test" if test else "Validation"

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader, desc=desc):
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            logits = model(input_ids, attention_mask)

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
    # fetch bilstm model config
    model_config = bilstm_without_glove_config()

    model_save_path = model_config["model_save_path"]
    num_epochs = model_config["num_epochs"]
    learning_rate = model_config["learning_rate"]
    batch_size = model_config["batch_size"]
    device = model_config["device"]

    # early stopping parameters
    best_val_f1 = 0
    trigger_times = 0
    patience = 5

    print(f"Using device : {device}")

    print("Loading training data...")
    train_ds = torch.load(CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE, weights_only=False)
    test_ds = torch.load(CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE, weights_only=False)

    train_len = int(0.90 * len(train_ds))
    val_len = len(train_ds) - train_len
    train_ds, val_ds = random_split(
        train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # initialise model
    model = BiLSTM_bert(
        bert_model_name=model_config["bert_model_name"],
        hidden_dim=model_config["hidden_dim"],
        num_classes=CROWDFLOWER_CLASSES,
        dropout_rate=model_config["dropout_rate"],
        lstm_layers=model_config["lstm_layers"],
    )
    model.to(device)

    # initialse loss function
    criterion = nn.CrossEntropyLoss()

    # initialise optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for input_ids, attention_mask, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}"
        ):
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

        val_accuracy, val_f1, val_recall, val_precision = evaluate(
            model, val_loader, device
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            trigger_times = 0
            print(
                f"Best model saved at epoch {epoch+1} with accuracy: {val_accuracy:.4f} and val: {val_f1:.4f}"
            )
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\n----- Starting Evaluation on Test Set -----\n")
    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict)
    evaluate(model, test_loader, device, test=True)


if __name__ == "__main__":
    main()
