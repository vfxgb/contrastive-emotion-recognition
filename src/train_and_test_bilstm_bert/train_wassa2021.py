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
from models.bilstm_model import BiLSTM_BERT_Encoder, BiLSTM_Classifier
from utils import print_test_stats, set_seed, split_dataset, get_versioned_path
from config import (
    SEED,
    USE_TQDM,
    WASSA_CLASSES,
    WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE,
    WASSA_TEST_DS_PATH_WITHOUT_GLOVE,
    bilstm_bert_config,
    F1_AVERAGE_METRIC,
)
import argparse

torch.serialization.add_safe_globals([TensorDataset])

finetune_mode = 1

def load_and_adapt_model(pretrained_model_path, num_classes, model_config):
    """
    Loads a pretrained model's state dictionary, adapts it by exluding the final classification layer,
    and returns a new model instance with the loaded parameters.

    Args:
        pretrained_model_path (str): The file path to the saved pretrained model.
        num_classes (int): The number of output classes for the final classification layer.
        model_config (dict):  model config containing hyperparameter and other config.
    Returns:
        encoder (BiLSTM_BERT_Encoder): encoder according to the finetune_mode set
        classifier (BiLSTM_Classifier):  classifier according to the finetune_mode set
    """
    print(f"[Finetune mode] : finetune mode is {finetune_mode}")
    if finetune_mode == 1:
        # ==== load checkpoint ===
        checkpoint = torch.load(pretrained_model_path, map_location=model_config["device"])
        # initialise model
        encoder = BiLSTM_BERT_Encoder(
            bert_model_name=model_config["bert_model_name"],
            hidden_dim=model_config["hidden_dim"],
            lstm_layers=model_config["lstm_layers"]
        )
        encoder.load_state_dict(checkpoint["encoder"])

        classifier = BiLSTM_Classifier(
            hidden_dim=model_config["hidden_dim"],
            num_classes=num_classes,
            dropout_rate=model_config["dropout_rate"]
        )
        
        classifier_dict = classifier.state_dict()
        pretrained_classifier_dict = checkpoint["classifier"]

        # filter out final classification layer
        pretrained_classifier_dict = {
            k: v
            for k, v in pretrained_classifier_dict.items()
            if k in classifier_dict and "fc3" not in k
        }
        classifier_dict.update(pretrained_classifier_dict)
        classifier.load_state_dict(classifier_dict)

        # === freeze encoder === 
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder, classifier
    
    elif finetune_mode == 2:
        # load checkpoint 
        checkpoint = torch.load(pretrained_model_path, map_location=model_config["device"])
        # initialise model
        encoder = BiLSTM_BERT_Encoder(
            bert_model_name=model_config["bert_model_name"],
            hidden_dim=model_config["hidden_dim"],
            lstm_layers=model_config["lstm_layers"]
        )
        encoder.load_state_dict(checkpoint["encoder"])

        classifier = BiLSTM_Classifier(
            hidden_dim=model_config["hidden_dim"],
            num_classes=num_classes,
            dropout_rate=model_config["dropout_rate"]
        )
        
        classifier_dict = classifier.state_dict()
        pretrained_classifier_dict = checkpoint["classifier"]

        # filter out final classification layer
        pretrained_classifier_dict = {
            k: v
            for k, v in pretrained_classifier_dict.items()
            if k in classifier_dict and "fc3" not in k
        }
        classifier_dict.update(pretrained_classifier_dict)
        classifier.load_state_dict(classifier_dict)

        return encoder, classifier
    
    elif finetune_mode == 3:
        encoder = BiLSTM_BERT_Encoder(
            bert_model_name=model_config["bert_model_name"],
            hidden_dim=model_config["hidden_dim"],
            lstm_layers=model_config["lstm_layers"]
        )
        classifier = BiLSTM_Classifier(
            hidden_dim=model_config["hidden_dim"],
            num_classes=num_classes,
            dropout_rate=model_config["dropout_rate"]
        )

        return encoder, classifier
    
    else:
        raise ValueError("Invalid finetune mode.")


def evaluate(encoder, classifier, dataloader, device, test=False):
    """
    Evaluates model.

    Args:
        encoder: The encoder of the model to evaluate
        classifier: The classifier of the model to evaluate
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
        for input_ids, attention_mask, labels in tqdm(dataloader, desc=desc, disable=not USE_TQDM):
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            max_pool_encodings = encoder(input_ids, attention_mask)
            logits = classifier(max_pool_encodings)

            _, predicted = logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=F1_AVERAGE_METRIC, zero_division=0)

    print(
        f"Accuracy: {accuracy*100:.4f}%, F1 Score: {f1:.4f}"
    )
    print("\nDetailed Report:\n", classification_report(all_labels, all_preds, zero_division=0))

    return accuracy, f1


def main():
    # Configurations
    model_config = bilstm_bert_config()
    print(f"\n 🌟🌟 Model Configuration : {model_config}, Finetune Mode : {finetune_mode}")

    num_classes = WASSA_CLASSES
    num_epochs = model_config["num_epochs"]
    learning_rate = model_config["learning_rate"]
    batch_size = model_config["finetune_batch_size"]
    device = model_config["device"]
    wassa21_finetune_save_path = get_versioned_path(model_config["wassa21_finetune_save_path"], finetune_mode)

    patience = 5
    num_runs = 5
    test_acc_list = []
    test_f1_list = []

    for run in range(num_runs):
        best_val_f1 = 0
        trigger_times = 0

        print(f"\n🔁 Run {run+1}/{num_runs}")
        set_seed(SEED + run)

        train_ds = torch.load(WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE, weights_only=False)
        test_ds = torch.load(WASSA_TEST_DS_PATH_WITHOUT_GLOVE, weights_only=False)

        train_ds, val_ds = split_dataset(
            dataset=train_ds, split_ratio=0.9, seed=SEED+run, glove=False
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

       # initialise model
        encoder, classifier = load_and_adapt_model(
            model_config["model_save_path"],
            num_classes=num_classes,
            model_config=model_config,
        )

        encoder.to(device)
        classifier.to(device)

        # initialse loss function
        criterion = nn.CrossEntropyLoss()

        # initialise optimiser
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate
        )

        for epoch in range(num_epochs):
            encoder.train()
            classifier.train()
            total_loss = 0

            for input_ids, attention_mask, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}", disable=not USE_TQDM
            ):
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )

                optimizer.zero_grad()

                max_pool_encodings = encoder(input_ids, attention_mask)
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
                    wassa21_finetune_save_path,
                )
                trigger_times = 0
                print(
                    f"Best model saved at epoch {epoch+1} with accuracy: {val_accuracy:.4f}"
                )
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print("\n----- Starting Evaluation on Test Set -----\n")
        checkpoint = torch.load(wassa21_finetune_save_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        classifier.load_state_dict(checkpoint["classifier"])
        test_accuracy, test_f1 = evaluate(
            encoder, classifier, test_loader, device, test=True
        )

        test_acc_list.append(test_accuracy)
        test_f1_list.append(test_f1)

    print_test_stats(
        test_acc_list, test_f1_list, num_runs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Accepts values 1, 2, or 3 to represent embedding type (e.g., 1: GloVe, 2: BERT, 3: Both or other config)
    parser.add_argument(
        "--finetune_mode",
        type=int,
        choices=[1, 2, 3],
        required=True,  # Optional: Force the user to provide this
        help="Embedding mode: 1 for GloVe, 2 for BERT, 3 for both"
    )

    args = parser.parse_args()
    finetune_mode = args.finetune_mode
    main()
