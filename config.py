
from pathlib import Path

def get_config():
    config = {
        'batch_size_train': 32,
        'batch_size_eval': 1,
        'val_set_coef': 0.15,
        'dataset_path': Path("./datasets/pets/"),
        'epoch':25,
        'weight_decay':1e-5,
        'learning_rate':1e-3,
        'patience':3,
        'factor':0.5,
        
    }
    return config
