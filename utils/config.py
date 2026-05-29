import yaml
from easydict import EasyDict
import os
import shutil
from .logger import print_log

def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger = logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger = logger)

def _expand_env(val, strict=True):
    expanded = os.path.expandvars(val)
    if strict and '${' in expanded:
        raise EnvironmentError(
            f"Config value '{val}' contains an unset environment variable. "
            f"Set the variable or edit the yaml directly."
        )
    return expanded

def merge_new_config(config, new_config, strict_env=True):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    val = yaml.safe_load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val, strict_env=strict_env)
            else:
                if isinstance(val, str):
                    val = _expand_env(val, strict=strict_env)
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val, strict_env=strict_env)
    return config

def cfg_from_yaml_file(cfg_file, strict_env=True):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.safe_load(f)
    merge_new_config(config=config, new_config=new_config, strict_env=strict_env)        
    return config

def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume:
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    shutil.copy(args.config, config_path)
    print_log(f'Copy the Config file from {args.config} to {config_path}',logger = logger )
