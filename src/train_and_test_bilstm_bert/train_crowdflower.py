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
from config import (
    F1_AVERAGE_METRIC,
    bilstm_bert_config,
    CROWDFLOWER_CLASSES,
    CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE,
    CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE,
    USE_TQDM
)
from utils import split_dataset

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
        for input_ids, attention_mask, labels in tqdm(dataloader, desc=desc, disable=not USE_TQDM):
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            # get prediction
            max_pool_encodings = encoder(input_ids, attention_mask)
            logits = classifier(max_pool_encodings)

            _, predicted = logits.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=F1_AVERAGE_METRIC, zero_division=0)

    print(
        f"Accuracy: {accuracy*100:.2f}%, F1 Score: {f1:.4f}"
    )
    print("\Classification Report:\n", classification_report(all_labels, all_preds, zero_division=0))

    return accuracy, f1


def main():
    # fetch bilstm model config
    model_config = bilstm_bert_config()
    print(f"\n [Main] Dataset : Crowdflower, Model Configuration : {model_config}")

    num_classes = CROWDFLOWER_CLASSES
    model_save_path = model_config["model_save_path"]
    num_epochs = model_config["num_epochs"]
    learning_rate = model_config["learning_rate"]
    batch_size = model_config["batch_size"]
    device = model_config["device"]

    # early stopping parameters
    best_val_f1 = 0
    trigger_times = 0
    patience = 5

    print("[Main] Loading training data...")
    train_ds = torch.load(CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE, weights_only=False)
    test_ds = torch.load(CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE, weights_only=False)

    train_ds, val_ds = split_dataset(train_ds, split_ratio=0.9, glove=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # initialise model
    encoder = BiLSTM_BERT_Encoder(
        bert_model_name=model_config["bert_model_name"],
        hidden_dim=model_config["hidden_dim"],
        lstm_layers=model_config["lstm_layers"]
    )
    encoder.to(device)

    classifier = BiLSTM_Classifier(
        hidden_dim=model_config["hidden_dim"],
        num_classes=num_classes,
        dropout_rate=model_config["dropout_rate"]
    )
    classifier.to(device)
    
    # initialse loss function
    criterion = nn.CrossEntropyLoss()

    # initialise optimiser
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate
    )

    print("[Main] Start training model...")
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

        print(f"[Epoch {epoch+1}]")

        # print out evaluation metrics
        val_accuracy, val_f1 = evaluate(
            encoder, classifier, val_loader, device
        )

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

    # load saved best model 
    checkpoint = torch.load(model_save_path, map_location=device)

    # initialise model
    test_encoder = BiLSTM_BERT_Encoder(
        bert_model_name=model_config["bert_model_name"],
        hidden_dim=model_config["hidden_dim"],
        lstm_layers=model_config["lstm_layers"]
    )
    test_encoder.load_state_dict(checkpoint["encoder"])
    test_encoder.to(device)

    test_classifier = BiLSTM_Classifier(
        hidden_dim=model_config["hidden_dim"],
        num_classes=num_classes,
        dropout_rate=model_config["dropout_rate"]
    )
    test_classifier.load_state_dict(checkpoint["classifier"])
    test_classifier.to(device)
    
    
    # print results on the test set
    evaluate(test_encoder, test_classifier, test_loader, device, test=True)


if __name__ == "__main__":
    main()
