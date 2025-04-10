from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score
)
from models.bilstm_model import BiLSTM_GloVe_Encoder, BiLSTM_Classifier
from config import (
    F1_AVERAGE_METRIC,
    bilstm_glove_config,
    CROWDFLOWER_CLASSES,
    CROWDFLOWER_TRAIN_DS_PATH_WITH_GLOVE,
    CROWDFLOWER_TEST_DS_PATH_WITH_GLOVE,
    CROWDFLOWER_GLOVE_EMBEDDINGS_PATH,
    USE_TQDM, 
    SEED
)
from utils import split_dataset, set_seed, print_test_stats

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
    """
    encoder.eval()
    classifier.eval()

    all_labels = []
    all_preds = []
    desc = "Test" if test else "Validation"

    with torch.no_grad():
        for input_ids, labels in tqdm(dataloader, desc=desc, disable=not USE_TQDM):
            input_ids, labels = (
                input_ids.to(device),
                labels.to(device),
            )

            max_pool_encodings = encoder(input_ids)
            logits = classifier(max_pool_encodings)

            _, predicted = logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=F1_AVERAGE_METRIC, zero_division=0)

    print(
        f"Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.4f}"
    )
    print("\nDetailed Report:\n", classification_report(all_labels, all_preds, zero_division=0))

    return accuracy, f1

def main():
    # Configurations
    model_config = bilstm_glove_config()
    print(f"\n 🌟🌟 Model Configuration : {model_config}")

    num_classes = CROWDFLOWER_CLASSES
    num_epochs = model_config["num_epochs"]
    learning_rate = 1e-4
    batch_size = model_config["batch_size"]
    device = model_config["device"]
    crowdflower_model_save_path = model_config["crowdflower_model_save_path"]

    patience = 5
    num_runs = 5
    test_acc_list = []
    test_f1_list = []

    for run in range(num_runs):
        print(f"\n🔁 Run {run+1}/{num_runs}")
        set_seed(SEED + run)

        best_val_f1 = 0
        trigger_times = 0

        train_ds = torch.load(CROWDFLOWER_TRAIN_DS_PATH_WITH_GLOVE, weights_only=False)
        test_ds = torch.load(CROWDFLOWER_TEST_DS_PATH_WITH_GLOVE, weights_only=False)

        train_ds, val_ds = split_dataset(
            dataset=train_ds, split_ratio=0.9, seed=SEED + run, glove=True
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # initialise model 
        encoder = BiLSTM_GloVe_Encoder(
            embedding_matrix_path=CROWDFLOWER_GLOVE_EMBEDDINGS_PATH,
            hidden_dim=model_config["hidden_dim"],
            lstm_layers=model_config["lstm_layers"]
        )

        classifier = BiLSTM_Classifier(
            hidden_dim=model_config["hidden_dim"],
            num_classes=num_classes,
            dropout_rate=model_config["dropout_rate"]
        )

        encoder.to(device)
        classifier.to(device)

        # initialse loss function and optimiser
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate
        )

        for epoch in range(num_epochs):
            encoder.train()
            classifier.train()

            total_loss = 0

            for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not USE_TQDM):
                input_ids, labels = (
                    input_ids.to(device),
                    labels.to(device),
                )

                optimizer.zero_grad()

                max_pool_encodings = encoder(input_ids)
                logits = classifier(max_pool_encodings)
                loss = criterion(logits, labels)
                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

            val_accuracy, val_f1 = evaluate(
                encoder, classifier, val_loader, device
            )
            print(f"[Epoch {epoch+1}] Validation Accuracy: {val_accuracy:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "classifier": classifier.state_dict(),
                    },
                    crowdflower_model_save_path,
                )
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
        checkpoint = torch.load(crowdflower_model_save_path, map_location=device)

        encoder.load_state_dict(checkpoint["encoder"])
        classifier.load_state_dict(checkpoint["classifier"])

        # fetch results on the test set
        test_accuracy, test_f1 = evaluate(encoder, classifier, test_loader, device, test=True)
        test_acc_list.append(test_accuracy)
        test_f1_list.append(test_f1)

    print_test_stats(
        test_acc_list, test_f1_list, num_runs
    )


if __name__ == "__main__":
    main()
