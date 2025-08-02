import yaml

_config = {}

def load_config(config_file: str = "configs/config.yaml"):
    global _config

    with open(config_file, "r") as file:
        _config = yaml.safe_load(file)
    return _config


def get_config():
    return _config.copy()
