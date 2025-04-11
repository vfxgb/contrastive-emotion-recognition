from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
from utils import (
    print_test_stats,
    set_seed,
    DualViewDataset,
    SupConLoss,
    get_versioned_path,
)
from config import (
    SEED,
    WASSA_CLASSES,
    WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE,
    WASSA_TEST_DS_PATH_WITHOUT_GLOVE,
    mamba_config,
    F1_AVERAGE_METRIC,
    USE_TQDM,
)
from models.contrastive_model import ContrastiveMambaEncoder, ClassifierHead
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
        encoder (ContrastiveMambaEncoder): encoder according to the finetune_mode set
        classifier (ClassifierHead):  classifier according to the finetune_mode set
    """
    print(f"[Finetune mode] : Finetune mode set to {finetune_mode}")

    if finetune_mode == 1 or finetune_mode == 2:
        # load checkpoint
        checkpoint = torch.load(
            pretrained_model_path, map_location=model_config["device"]
        )

        # initialise model
        encoder = ContrastiveMambaEncoder(
            mamba_args=model_config["mamba_args"], embed_dim=model_config["embed_dim"]
        )
        encoder.load_state_dict(checkpoint["encoder"])

        classifier = ClassifierHead(
            embed_dim=model_config["embed_dim"], num_emotions=num_classes
        )

        classifier_dict = classifier.state_dict()
        pretrained_classifier_dict = checkpoint["classifier"]
        filtered_dict = {
            k: v
            for k, v in pretrained_classifier_dict.items()
            if k in classifier_dict and "classifier.6" not in k
        }
        classifier_dict.update(filtered_dict)
        classifier.load_state_dict(classifier_dict)

        # freeze encoder
        if finetune_mode == 1:
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder, classifier

    elif finetune_mode == 3:
        encoder = ContrastiveMambaEncoder(
            mamba_args=model_config["mamba_args"], embed_dim=model_config["embed_dim"]
        )
        classifier = ClassifierHead(
            embed_dim=model_config["embed_dim"], num_emotions=num_classes
        )

        return encoder, classifier

    else:
        raise ValueError("Invalid finetune mode.")


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
        "Classification Report:\n",
        classification_report(all_labels, all_preds, zero_division=0),
    )

    return accuracy, f1


def main():
    # fetch model config
    model_config = mamba_config()
    print(f"\n [Main] Dataset : Wassa, Model Configuration : {model_config}")

    num_classes = WASSA_CLASSES
    mamba_args = model_config["mamba_args"]
    device = model_config["device"]
    embed_dim = model_config["embed_dim"]
    batch_size = model_config["batch_size"]
    num_epochs = model_config["num_epochs"]
    learning_rate = model_config["learning_rate"]
    model_save_path = model_config["model_save_path"]
    wassa21_finetune_save_path = get_versioned_path(
        model_config["wassa21_finetune_save_path"], finetune_mode
    )

    patience = 5
    num_runs = 5

    test_acc_list = []
    test_f1_list = []

    for run in range(num_runs):
        best_val_f1 = 0
        trigger_times = 0

        set_seed(SEED + run)

        print(f"\n[Main] Run {run+1}/{num_runs}")
        print("[Main] Loading training data...")
        train_ds = torch.load(WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE, weights_only=False)
        test_ds = torch.load(WASSA_TEST_DS_PATH_WITHOUT_GLOVE, weights_only=False)

        train_len = int(0.90 * len(train_ds))
        val_len = len(train_ds) - train_len
        train_subset, val_subset = random_split(
            train_ds,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(SEED + run),
        )

        # Apply augmentation ONLY to training subset
        train_ds = DualViewDataset(train_subset, dropout_prob=0.1)
        val_ds = val_subset  # Keep validation data original

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        encoder, classifier = load_and_adapt_model(
            pretrained_model_path=model_save_path,
            num_classes=num_classes,
            model_config=model_config,
        )

        encoder.to(device)
        classifier.to(device)

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
                train_loader, desc=f"Epoch {epoch+1}", leave=False, disable=not USE_TQDM
            ):
                view1, view2, labels = (
                    view1.to(device),
                    view2.to(device),
                    labels.to(device),
                )
                emb1 = encoder(view1)
                emb2 = encoder(view2)
                features = torch.stack([emb1, emb2], dim=1)

                if finetune_mode == 1 or finetune_mode == 2:
                    # use only CE when finetuning
                    loss = criterion_cls(classifier(emb1), labels)

                elif finetune_mode == 3:
                    # use supcon + CE when training from scratch
                    loss_cls = criterion_cls(classifier(emb1), labels)
                    loss_contrastive = criterion_contrastive(features, labels)
                    loss = 0.9 * loss_cls + 0.1 * loss_contrastive

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch+1}]")

            # Validation step
            val_accuracy, val_f1 = evaluate(encoder, classifier, val_loader, device)

            # Save model based on best val F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "classifier": classifier.state_dict(),
                        "val_f1": best_val_f1,
                        "epoch": epoch + 1,
                    },
                    wassa21_finetune_save_path,
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

        print("\n[Main] Start testing model...")
        # load saved best model
        checkpoint = torch.load(wassa21_finetune_save_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        classifier.load_state_dict(checkpoint["classifier"])

        # print results on test set
        test_accuracy, test_f1 = evaluate(
            encoder, classifier, test_loader, device, test=True
        )

        test_acc_list.append(test_accuracy)
        test_f1_list.append(test_f1)

    # print avg stats across all runs
    print_test_stats(test_acc_list, test_f1_list, num_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # takes in values 1,2,3 depending on how to train the model
    parser.add_argument(
        "--finetune_mode",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Select Training Mode",
    )

    args = parser.parse_args()
    finetune_mode = args.finetune_mode
    main()
