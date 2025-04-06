import torch

WASSA_CLASSES = 6
ISEAR_CLASSES = 7 
CROWDFLOWER_CLASSES = 9

def bilstm_config():
    """
    Configuration for the BiLSTM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "bert_model_name": "bert-large-uncased",
        "hidden_dim" : 256,
        "dropout_rate" : 0.3, 
        "lstm_layers": 1,
        "num_epochs": 30,
        "learning_rate": 0.001,
        "batch_size": 1024, 
        "finetune_batch_size": 32,
        "device":device,
        "model_save_path": "results/bilstm/bilstm.pt",
        "isear_finetune_save_path": "results/bilstm/isear_finetune_bilstm.pt",
        "wassa21_finetune_save_path": "results/bilstm/wassa21_finetune_bilstm.pt",
    }
    
    return model_config

def bigru_config():
    """
    Configuration for the BiLSTM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "bert_model_name": "bert-large-uncased",
        "hidden_dim" : 256,
        "dropout_rate" : 0.3, 
        "gru_layers": 2,
        "num_epochs": 30,
        "learning_rate": 0.001,
        "batch_size": 32, 
        "finetune_batch_size": 32,
        "device":device,
        "model_save_path": "results/bigru/bigru.pt",
        "isear_finetune_save_path": "results/bigru/isear_finetune_bigru.pt",
        "wassa21_finetune_save_path": "results/bigru/wassa21_finetune_bigru.pt",
    }
    
    return model_config