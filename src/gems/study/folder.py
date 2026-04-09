"""
This module provides functions to load and run simulation studies.

A study is defined by a directory containing:
- `input/system.yml`: A file describing the system to be simulated.
- `input/model-libraries/`: A folder containing model library files in YAML format.
- `input/data-series/`: A folder containing data series files.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.optim_config import load_optim_config
from gems.simulation import TimeBlock, build_problem
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    build_network,
    consistency_check,
    resolve_system,
)


def load_study(study_dir: Path):
    """
    Loads a study from a given directory.

    This function reads the system definition, model libraries, and data series
    from the study directory, resolves them, and builds the simulation network
    and database.

    Args:
        study_dir: The path to the study directory.

    Returns:
        A tuple containing the simulation network and the database.
    """
    study_file = study_dir / "input" / "system.yml"
    lib_folder = study_dir / "input" / "model-libraries"
    series_dir = study_dir / "input" / "data-series"
    config_file = study_dir / "input" / "optim-config.yml"

    if config_file.exists():
        optim_config = load_optim_config(config_file)
        raise Warning('An optim config file has been provided but is not '
                      'used in the current version of problem definition')

    input_libraries = []
    for lib_file in lib_folder.glob("*.yml"):
        with lib_file.open() as lib:
            input_libraries.append(parse_yaml_library(lib))

    with study_file.open() as c:
        input_study = parse_yaml_components(c)
    lib_dict = resolve_library(input_libraries)
    network_components = resolve_system(input_study, lib_dict)
    model_dict = {}
    for lib in lib_dict.values():
        model_dict |= lib.models
    consistency_check(network_components.components, model_dict)

    database = build_data_base(input_study, series_dir)
    network = build_network(network_components)
    return network, database


def run_study(study_dir: Path, scenarios: int, time_block: TimeBlock):
    """
    Runs a simulation study.

    This function loads a study, builds a simulation problem, and solves it.

    Args:
        study_dir: The path to the study directory.
        scenarios: The number of scenarios to run.
        time_block: The time block for the simulation.

    Returns:
        The solved simulation problem.
    """
    
    
    network, database = load_study(study_dir)
    problem = build_problem(network, database, time_block, scenarios)
    problem.solve()
    
    return problem