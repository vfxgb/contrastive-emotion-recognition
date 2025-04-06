import torch

# Configuration for CrowdFlower dataset
CROWDFLOWER_CLASSES = 9
CROWDFLOWER_PATH = "data/CrowdFlower/text_emotion.csv"
CROWDFLOWER_TRAIN_DS_PATH = "data/preprocessed_dataset/crowdflower/train.pt"
CROWDFLOWER_TEST_DS_PATH = "data/preprocessed_dataset/crowdflower/test.pt"

# Configuration for ISEAR dataset
ISEAR_CLASSES = 7
ISEAR_PATH = "data/ISEAR/isear_data.csv"
ISEAR_TRAIN_DS_PATH = "data/preprocessed_dataset/isear/train.pt"
ISEAR_TEST_DS_PATH = "data/preprocessed_dataset/isear/test.pt"

# Configuration for WASSA dataset
WASSA_CLASSES = 6
WASSA_PATH = "data/WASSA2021/wassa_2021.tsv"
WASSA_TRAIN_DS_PATH = "data/preprocessed_dataset/wassa/train.pt"
WASSA_TEST_DS_PATH = "data/preprocessed_dataset/wassa/test.pt"

# Configuration for BERT and spaCy models
BERT_MODEL = "bert-large-uncased"
SPACY_MODEL = "en_core_web_sm"

F1_AVERAGE_METRIC = "macro"


def bilstm_config():
    """
    Configuration for the BiLSTM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "bert_model_name": "bert-large-uncased",
        "hidden_dim": 256,
        "dropout_rate": 0.3,
        "lstm_layers": 1,
        "num_epochs": 30,
        "learning_rate": 0.001,
        "batch_size": 1024,
        "finetune_batch_size": 32,
        "device": device,
        "model_save_path": "results/bilstm/bilstm.pt",
        "isear_finetune_save_path": "results/bilstm/isear_finetune_bilstm.pt",
        "wassa21_finetune_save_path": "results/bilstm/wassa21_finetune_bilstm.pt",
    }

    return model_config


def mamba_config():
    """
    Configuration for the Mamba model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "embed_dim" : 1024, 
        "batch_size" : 128, 
        "num_epochs" : 30, 
        "learning_rate" : 6e-5,
        "model_save_path": "results/mamba/contrastive_mamba_decoupled.pt", 
        "mamba_args" : dict(
        d_model=2048,
        d_state=256,
        d_conv=4,
        expand=2,
        )
    }

    return model_config
