import yaml


CONFIG_FILE_PATH = './config.yaml'


def load_config(config_file=CONFIG_FILE_PATH):
    with open(config_file) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config