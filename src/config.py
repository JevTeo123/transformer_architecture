from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 5,
        "lr": 1e-5,
        "seq_len": 400,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "zh",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "datasource": "Helsinki-NLP/opus-100"
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)