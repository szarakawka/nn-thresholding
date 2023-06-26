from typing import Dict


default_nn_config = {
    'k': 50,
    'metric': 'cosine'
}


default_scoring_optimization_config = {
    "name": 'scores_optimization',
    "version": 'v0.1',
    "grid_hyperparameters": {
        "k": [30, 40, 50],
        "alpha": [1.0, 2.0, 3.0, 4.0]
    },
    "optimized_metric": 'map'   # ['map', 'p@3', 'p@5', 'ndcg@5']
}


default_thresholding_optimization_config = {
    "name": 'thresholding_optimization',
    "version": 'v0.1',
    "selection_method": 'one_held_out',   # any(['validation_based', 'one_held_out'])
    "grid_hyperparameters": {
        "thr_method": ["s", "ss", "cs", "css", "r", "p", "ps"]
    },
    "optimized_metric": 'ma_f'   # ['ma_f', 'mi_f']
}


def load_config_or_create_and_save_the_default_one(config_file_path: str, default_config: Dict = None) -> Dict:
    import os.path
    import json
    import logging
    if not os.path.exists(config_file_path):
        if default_config is None:
            logging.error('{} does not exist and you didn\' specify the default one.'.format(config_file_path))
            raise AssertionError()
        logging.warning('{} does not exist. I\'m creating the default one'.format(config_file_path))
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        with open(config_file_path, 'w') as f:
            json.dump(default_config, f)
        nn_config = default_config
    else:
        with open(config_file_path, 'r') as f:
            nn_config = json.load(f)
    return nn_config
