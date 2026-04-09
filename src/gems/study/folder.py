"""
This module provides functions to load and run simulation studies.

A study is defined by a directory containing:
- `input/system.yml`: A file describing the system to be simulated.
- `input/model-libraries/`: A folder containing model library files in YAML format.
- `input/data-series/`: A folder containing data series files.
"""
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gems.model.model import Model
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.optim_config import load_optim_config
from gems.simulation import TimeBlock, build_problem
from gems.simulation.linopy_problem import LinopyOptimizationProblem
from gems.study.data import DataBase
from gems.study.network import System
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)


def load_study(study_dir: Path) -> tuple[System, DataBase]:
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
    system_file = study_dir / "input" / "system.yml"
    lib_folder = study_dir / "input" / "model-libraries"
    series_dir = study_dir / "input" / "data-series"
    config_file = study_dir / "input" / "optim-config.yml"

    if config_file.exists():
        optim_config = load_optim_config(config_file)
        raise Warning(
            "An optim config file has been provided but is not "
            "used in the current version of problem definition"
        )

    input_libraries = []
    for lib_file in lib_folder.glob("*.yml"):
        with lib_file.open() as lib:
            input_libraries.append(parse_yaml_library(lib))

    with system_file.open() as c:
        input_study = parse_yaml_components(c)
    lib_dict = resolve_library(input_libraries)
    system = resolve_system(input_study, lib_dict)
    model_dict: dict[str, Model] = {}
    for library in lib_dict.values():
        model_dict |= library.models
    consistency_check(system, model_dict)

    database = build_data_base(input_study, series_dir)
    return system, database


def run_study(
    study_dir: Path,
    scenarios: int,
    time_block: TimeBlock,
    export_simulation_table: Optional[bool] = False,
) -> LinopyOptimizationProblem:
    """
    Runs a simulation study.

    This function loads a study, builds a simulation problem, and solves it.

    Args:
        study_dir: The path to the study directory.
        scenarios: The number of scenarios to run.
        time_block: The time block for the simulation.
        export_simulation_table: Whether to export a simulation table CSV file.

    Returns:
        The solved simulation problem.
    """

    network, database = load_study(study_dir)
    problem = build_problem(network, database, time_block, scenarios)
    problem.solve()
    if export_simulation_table:
        from gems.simulation.simulation_table import SimulationTableBuilder

        builder = SimulationTableBuilder(simulation_id=study_dir.stem)
        df = builder.build(problem)
        output_dir = study_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = output_dir / f"{study_dir.stem}_simulation_table_{timestamp}.csv"

        df.to_csv(output_file, index=False)

    return problem
