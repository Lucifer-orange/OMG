from glob import glob
import numpy as np
import os
import collections
from functools import reduce
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
from utils.MY_EXP_PATH import EXP_DATA_PATH
import yaml
import re
from run import run
import argparse


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

#results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
results_path = os.path.join(EXP_DATA_PATH, "SMAC")

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)

def _get_other_config(params):
    config_dict = {}
    new_params = []
    for _i, _v in enumerate(params):
        print(_v)
        if len(_v.split("=")) == 2 and _v.split("=")[0][:2] == "--":
            v = _v.split("=")[1]
            try:
                int(v)
                config_dict[_v.split("=")[0][2:]] = int(v)
            except:
                try:
                    float(v)
                    config_dict[_v.split("=")[0][2:]] = float(v)
                except:
                    config_dict[_v.split("=")[0][2:]] = _v.split("=")[1]
        else:
            new_params.append(_v)
    return new_params, config_dict

def _get_config(params, arg_name, subfolder, single_config=True):
    config_name = None

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break
    
    if config_name is not None:
        if single_config:
            with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
                try:
                    config_dict = yaml.load(f)
                except yaml.YAMLError as exc:
                    assert False, "{}.yaml error: {}".format(config_name, exc)
            return config_dict
        else:
            del params[params.index("--multi_algs")]
            config_dict = {"name":config_name}
            algs = re.split(r"[\u0030-\u0039\s]+", config_name)[1:]
            if not algs:
                algs = [config_name]
            for conf_name in algs:
                with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(conf_name)), "r") as f:
                    try:
                        temp = yaml.load(f)
                        config_dict[conf_name] = temp
                    except yaml.YAMLError as exc:
                        assert False, "{}.yaml error: {}".format(conf_name, exc)
            return config_dict

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
     
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs", single_config= False if '--multi_algs' in params else True)
    params, other_arg_config = _get_other_config(params)

    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict = recursive_dict_update(config_dict, other_arg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    if config_dict["debug"]:
        logger.info("Saving to FileStorageObserver in results/debug.")
        file_obs_path = os.path.join(results_path, "debug")
        ex.observers.append(FileStorageObserver.create(file_obs_path))
    else:
        logger.info("Saving to FileStorageObserver in results/sacred.")
        file_obs_path = os.path.join(results_path, "sacred")
        ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

