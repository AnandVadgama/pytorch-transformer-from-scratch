from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 24,
        "d_model": 512,
        "datasource": 'english-to-colloquial-hindi-dataset',
        "src_lang": "english",
        "trg_lang": "hindi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "29",
        "translation": "Bhoomi06/english-to-colloquial-hindi-dataset",
        "tokenizer_path": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)