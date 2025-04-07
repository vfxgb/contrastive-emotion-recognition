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
from models.bilstm_model import BiLSTM_bert
from utils import set_seed
from config import (
    ISEAR_CLASSES,
    ISEAR_TEST_DS_PATH_WITHOUT_GLOVE,
    ISEAR_TRAIN_DS_PATH_WITHOUT_GLOVE,
    bilstm_without_glove_config,
    F1_AVERAGE_METRIC,
)

torch.serialization.add_safe_globals([TensorDataset])


def load_and_adapt_model(pretrained_model_path, num_classes, model_config):
    """
    Loads a pretrained model's state dictionary, adapts it by exluding the final classification layer,
    and returns a new model instance with the loaded parameters.

    Args:
        pretrained_model_path (str): The file path to the saved pretrained model.
        num_classes (int): The number of output classes for the final classification layer.
        model_config (dict):  model config containing hyperparameter and other config.
    """
    pretrained_state_dict = torch.load(
        pretrained_model_path, map_location=model_config["device"]
    )

    # Create a new model instance
    new_model = BiLSTM_bert(
        bert_model_name=model_config["bert_model_name"],
        hidden_dim=model_config["hidden_dim"],
        num_classes=num_classes,
        dropout_rate=model_config["dropout_rate"],
        lstm_layers=model_config["lstm_layers"],
    )

    # load parameters from pretrained model
    model_dict = new_model.state_dict()

    # filter out final classification layer
    pretrained_state_dict = {
        k: v
        for k, v in pretrained_state_dict.items()
        if k in model_dict and "fc3" not in k
    }
    model_dict.update(pretrained_state_dict)
    new_model.load_state_dict(model_dict)

    return new_model


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
    # Configurations
    model_config = bilstm_without_glove_config()

    num_classes = ISEAR_CLASSES
    num_epochs = model_config["num_epochs"]
    learning_rate = model_config["learning_rate"]
    batch_size = model_config["finetune_batch_size"]
    device = model_config["device"]
    isear_finetune_save_path = model_config["isear_finetune_save_path"]

    best_val_f1 = 0
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

        train_ds = torch.load(ISEAR_TRAIN_DS_PATH_WITHOUT_GLOVE, weights_only=False)
        test_ds = torch.load(ISEAR_TEST_DS_PATH_WITHOUT_GLOVE, weights_only=False)

        train_len = int(0.90 * len(train_ds))
        val_len = len(train_ds) - train_len
        train_ds, val_ds = random_split(
            train_ds,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42 + run),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # initialise model
        model = load_and_adapt_model(
            model_config["model_save_path"],
            num_classes=num_classes,
            model_config=model_config,
        )
        
        # freeze bert and lstm layers and only train the final classification layer
        for param in model.bert.parameters():
            param.requires_grad = False  # freeze all
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
            print(f"[Epoch {epoch+1}] Validation Accuracy: {val_accuracy:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), isear_finetune_save_path)
                trigger_times = 0
                print(
                    f"Best model saved at cepoch {epoch+1} with accuracy: {val_accuracy:.4f}"
                )
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print("\n----- Starting Evaluation on Test Set -----\n")
        state_dict = torch.load(isear_finetune_save_path, map_location=device)
        model.load_state_dict(state_dict)
        test_accuracy, test_f1, test_recall, test_precision = evaluate(
            model, test_loader, device, test=True
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
