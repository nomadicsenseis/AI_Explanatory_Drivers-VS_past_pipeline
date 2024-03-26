"""
Utils to use for create pipeline.
"""
from os.path import dirname, realpath
from typing import Dict

from yaml import safe_load


def read_config_data() -> Dict:
    """Read the config.yml file asociated.

    The config.yml file asociated is the one in the same path.

    Returns
    -------
        Dictionary with the configuration of the process.
    """
    base_path = dirname(realpath(__file__))
    config_file_path = f"{base_path}/config.yml"
    with open(config_file_path) as conf_file:
        configuration = conf_file.read()
    return safe_load(configuration)
