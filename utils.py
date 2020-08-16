import yaml
import os

CONFIG_FILE_PATH = './config.yaml'
root_logdir = os.path.join(os.curdir, 'my_logs')


def load_config(config_file=CONFIG_FILE_PATH):
    with open(config_file) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)