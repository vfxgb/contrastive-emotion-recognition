import torch

# Configuration for CrowdFlower dataset
CROWDFLOWER_CLASSES = 9
CROWDFLOWER_PATH = "data/CrowdFlower/text_emotion.csv"
CROWDFLOWER_TRAIN_DS_PATH_WITHOUT_GLOVE = "data/preprocessed_dataset/crowdflower/train_wo_glove.pt"
CROWDFLOWER_TEST_DS_PATH_WITHOUT_GLOVE = "data/preprocessed_dataset/crowdflower/test_wo_glove.pt"
CROWDFLOWER_TRAIN_DS_PATH_WITH_GLOVE = "data/preprocessed_dataset/crowdflower/train_w_glove.pt"
CROWDFLOWER_TEST_DS_PATH_WITH_GLOVE = "data/preprocessed_dataset/crowdflower/test_w_glove.pt"
CROWDFLOWER_GLOVE_EMBEDDINGS_PATH = "data/preprocessed_dataset/crowdflower/glove_embedding_matrix.npy"

# Configuration for ISEAR dataset
ISEAR_CLASSES = 7
ISEAR_PATH = "data/ISEAR/isear_data.csv"
ISEAR_TRAIN_DS_PATH_WITHOUT_GLOVE = "data/preprocessed_dataset/isear/train_wo_glove.pt"
ISEAR_TEST_DS_PATH_WITHOUT_GLOVE = "data/preprocessed_dataset/isear/test_wo_glove.pt"
ISEAR_TRAIN_DS_PATH_WITH_GLOVE = "data/preprocessed_dataset/isear/train_w_glove.pt"
ISEAR_TEST_DS_PATH_WITH_GLOVE = "data/preprocessed_dataset/isear/test_w_glove.pt"
ISEAR_GLOVE_EMBEDDINGS_PATH = "data/preprocessed_dataset/isear/glove_embedding_matrix.npy"

# Configuration for WASSA dataset
WASSA_CLASSES = 6
WASSA_PATH = "data/WASSA2021/wassa_2021.tsv"
WASSA_TRAIN_DS_PATH_WITHOUT_GLOVE = "data/preprocessed_dataset/wassa/train_wo_glove.pt"
WASSA_TEST_DS_PATH_WITHOUT_GLOVE = "data/preprocessed_dataset/wassa/test_wo_glove.pt"
WASSA_TRAIN_DS_PATH_WITH_GLOVE = "data/preprocessed_dataset/wassa/train_w_glove.pt"
WASSA_TEST_DS_PATH_WITH_GLOVE = "data/preprocessed_dataset/wassa/test_w_glove.pt"
WASSA_GLOVE_EMBEDDINGS_PATH = "data/preprocessed_dataset/wassa/glove_embedding_matrix.npy"

# Configuration for BERT and spaCy models
BERT_MODEL = "bert-large-uncased"
SPACY_MODEL = "en_core_web_sm"
GLOVE_PATH = "glove/glove.6B.300d.txt"

F1_AVERAGE_METRIC = "macro"
SEED = 42
USE_TQDM = False

def bilstm_bert_config():
    """
    Configuration for the BiLSTM_without_glove model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "name": "bilstm_bert",
        "bert_model_name": "bert-large-uncased",
        "hidden_dim": 256,
        "dropout_rate": 0.3,
        "lstm_layers": 1,
        "num_epochs": 30,
        "learning_rate": 0.001,
        "batch_size": 1024,
        "finetune_batch_size": 32,
        "device": device,
        "model_save_path": "results/bilstm_bert/bilstm_bert.pt",
        "isear_finetune_save_path": "results/bilstm_bert/isear_finetune_bilstm_bert.pt",
        "wassa21_finetune_save_path": "results/bilstm_bert/wassa21_finetune_bilstm_bert.pt",
    }

    return model_config


def bilstm_glove_config():
    """
    Configuration for the BiLSTM_without_glove model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "name": "bilstm_glove",
        "bert_model_name": "bert-large-uncased",
        "hidden_dim": 256,
        "dropout_rate": 0.3,
        "lstm_layers": 1,
        "num_epochs": 30,
        "learning_rate": 0.001,
        "batch_size": 128,
        "device": device,
        "crowdflower_model_save_path": "results/bilstm_glove/bilstm_glove_crowdflower.pt",
        "isear_model_save_path": "results/bilstm_glove/bilstm_glove_isear.pt",
        "wassa_model_save_path": "results/bilstm_glove/bilstm_glove_wassa.pt",
    }

    return model_config


def mamba_config():
    """
    Configuration for the Mamba model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "name": "mamba",
        "device": device,
        "embed_dim": 1024,
        "batch_size": 128,
        "num_epochs": 30,
        "learning_rate": 6e-5,
        "mamba_args": dict(
            d_model=2048,
            d_state=256,
            d_conv=4,
            expand=2,
        ),
        "model_save_path": "results/mamba/contrastive_mamba_decoupled.pt",
        "isear_finetune_save_path": "results/mamba/isear_finetune_contrastive_mamba_decoupled.pt",
        "wassa21_finetune_save_path": "results/mamba/wassa21_finetune_contrastive_mamba_decoupled.pt",
    }

    return model_config
