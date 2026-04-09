# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.


from pathlib import Path
from typing import Dict, List, Optional

from gems.model.library import Library
from gems.model.parsing import parse_yaml_library
from gems.model.resolve_library import resolve_library
from gems.optim_config.parsing import (
    OptimConfig,
    ResolutionMode,
    load_optim_config,
    validate_optim_config,
)
from gems.simulation import (
    BendersRunner,
    DecomposedProblems,
    TimeBlock,
    build_couplings,
    build_decomposed_problems,
    build_problem,
    dump_couplings,
)
from gems.study import DataBase, System
from gems.study.parsing import parse_cli, parse_yaml_components
from gems.study.resolve_components import (
    build_data_base,
    consistency_check,
    resolve_system,
)


class AntaresTimeSeriesImportError(Exception):
    pass


def input_libs(yaml_lib_paths: List[Path]) -> dict[str, Library]:
    yaml_libraries = []
    yaml_library_ids = set()

    for path in yaml_lib_paths:
        with path.open("r") as file:
            yaml_lib = parse_yaml_library(file)

            if yaml_lib.id in yaml_library_ids:
                raise ValueError(f"The identifier '{yaml_lib.id}' is defined twice")

            yaml_libraries.append(yaml_lib)
            yaml_library_ids.add(yaml_lib.id)

    return resolve_library(yaml_libraries)


def input_database(study_path: Path, timeseries_path: Optional[Path]) -> DataBase:
    with study_path.open() as comp:
        return build_data_base(parse_yaml_components(comp), timeseries_path)


def input_study(study_path: Path, librairies: dict[str, Library]) -> System:
    with study_path.open() as comp:
        return resolve_system(parse_yaml_components(comp), librairies)


def _write_structure_txt(
    decomposed: DecomposedProblems,
    optim_config: OptimConfig,
    output_dir: Path,
) -> None:
    dump_couplings(build_couplings(decomposed, optim_config), output_dir)


def main_cli() -> None:
    parsed_args = parse_cli()

    lib_dict = input_libs(parsed_args.models_path)
    study = input_study(parsed_args.components_path, lib_dict)

    models = {}
    for lib in lib_dict.values():
        models.update(lib.models)

    consistency_check(study, models)

    try:
        database = input_database(
            parsed_args.components_path, parsed_args.timeseries_path
        )

    except UnboundLocalError:
        raise AntaresTimeSeriesImportError(
            "An error occurred while importing time series."
        )

    timeblock = TimeBlock(1, list(range(parsed_args.duration)))
    scenario = parsed_args.nb_scenarios

    # Load optional optim-config.yml
    optim_config = load_optim_config(parsed_args.components_path)

    if optim_config is not None:
        validate_optim_config(optim_config, study)

        try:
            decomposed = build_decomposed_problems(
                study, database, timeblock, scenario, optim_config
            )
        except IndexError as e:
            raise IndexError(
                f"{e}. Did parameters '--duration' and '--scenario' were correctly set?"
            )

        if optim_config.resolution_mode == ResolutionMode.BENDERS_DECOMPOSITION:
            # Generate structure.txt then hand off to the external Benders solver
            if decomposed.master is not None:
                _write_structure_txt(
                    decomposed,
                    optim_config,
                    output_dir=parsed_args.components_path.parent,
                )
            BendersRunner(emplacement=parsed_args.components_path.parent).run()
        else:
            # sequential-subproblems (default): solve the subproblem directly
            decomposed.subproblem.solve(solver_name="highs")
            print("status : ", decomposed.subproblem.termination_condition)
            print("final average cost : ", decomposed.subproblem.objective_value)

    else:
        # No optim-config.yml — original unchanged behaviour
        try:
            problem = build_problem(study, database, timeblock, scenario)

        except IndexError as e:
            raise IndexError(
                f"{e}. Did parameters '--duration' and '--scenario' were correctly set?"
            )

        problem.solve(solver_name="highs")
        print("status : ", problem.termination_condition)

        print("final average cost : ", problem.objective_value)


if __name__ == "__main__":
    main_cli()
