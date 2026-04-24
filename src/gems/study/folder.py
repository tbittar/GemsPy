"""
This module provides functions to load and run simulation studies.

A study is defined by a directory containing:
- `input/system.yml`: A file describing the system to be simulated.
- `input/model-libraries/`: A folder containing model library files in YAML format.
- `input/data-series/`: A folder containing data series files.
- `input/optim-config.yml` (optional): Run parameters (time scope, solver options,
  scenario scope, and decomposition configuration).
"""

from pathlib import Path
from typing import Optional

from gems.model.model import Model
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.optim_config.parsing import (
    OptimConfig,
    load_optim_config,
    validate_optim_config,
)
from gems.session.session import SimulationSession
from gems.study.parsing import parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)
from gems.study.scenario_builder import ScenarioBuilder
from gems.study.study import Study


def load_study(study_dir: Path) -> Study:
    """
    Loads a study from a given directory.

    This function reads the system definition, model libraries, and data series
    from the study directory, resolves them, and builds the simulation system
    and database.

    Args:
        study_dir: The path to the study directory.

    Returns:
        A Study container holding the resolved system and database.
    """
    system_file = study_dir / "input" / "system.yml"
    lib_folder = study_dir / "input" / "model-libraries"
    series_dir = study_dir / "input" / "data-series"

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
    scenario_builder_path = study_dir / "input" / "scenariobuilder.dat"
    scenario_builder = (
        ScenarioBuilder.load(scenario_builder_path)
        if scenario_builder_path.exists()
        else ScenarioBuilder()
    )
    return Study(system=system, database=database, scenario_builder=scenario_builder)


def run_study(
    study_dir: Path,
    optim_config_path: Optional[Path] = None,
) -> None:
    """
    Runs a simulation study and exports results to CSV.

    Run parameters (time scope, solver options, scenario scope) are read from
    ``study_dir/input/optim-config.yml``; defaults apply when the file is absent.
    Results are written to ``study_dir/output/``.

    Args:
        study_dir: The path to the study directory.
        optim_config_path: Optional custom path to an optim-config YAML file.
            If not provided, defaults to ``study_dir/input/optim-config.yml``.
    """
    study = load_study(study_dir)

    resolved_config_path = optim_config_path or (
        study_dir / "input" / "optim-config.yml"
    )
    optim_config = load_optim_config(resolved_config_path) or OptimConfig()
    validate_optim_config(optim_config, study.system)

    session = SimulationSession(
        study=study,
        optim_config=optim_config,
        scenario_ids=optim_config.scenario_scope.scenario_ids,
    )
    table = session.run()
    table.to_csv(study_dir / "output")
