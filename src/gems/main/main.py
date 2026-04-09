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
    ElementLocation,
    OptimConfig,
    ResolutionMode,
    load_optim_config,
    validate_optim_config,
)
from gems.simulation import (
    BendersRunner,
    DecomposedProblems,
    TimeBlock,
    build_decomposed_problems,
    build_problem,
)
from gems.study import DataBase
from gems.study.parsing import parse_cli, parse_yaml_components
from gems.study.resolve_components import (
    System,
    build_data_base,
    build_network,
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


def _structure_row(problem_id: str, component_id: str, variable_int_id: int) -> str:
    return f"{problem_id:>24}{component_id:>48}{variable_int_id:>9}"


def _master_rows(
    decomposed: DecomposedProblems,
    model_id: str,
    var_name: str,
    comp_id: str,
) -> List[str]:
    if decomposed.master is None:
        return []
    labels = decomposed.master.get_variable_labels(model_id, var_name)
    if labels is None:
        return []
    return [
        _structure_row(
            decomposed.master.name, comp_id, int(labels.sel(component=comp_id).item())
        )
    ]


def _subproblem_rows(
    decomposed: DecomposedProblems,
    model_id: str,
    var_name: str,
    comp_id: str,
    scenarios: int,
) -> List[str]:
    labels = decomposed.subproblem.get_variable_labels(model_id, var_name)
    if labels is None:
        return []
    sid = int(labels.sel(component=comp_id).item())
    return [
        _structure_row(decomposed.subproblem.name, comp_id, sid)
        for _ in range(1, scenarios + 1)
    ]


def _write_structure_txt(
    decomposed: DecomposedProblems,
    optim_config: OptimConfig,
    scenarios: int,
    output_dir: Path,
) -> None:
    """Write structure.txt for master-and-subproblems variables.

    Each such variable produces one row per component in the master problem
    and one row per (scenario, component) in the subproblem.  All scenarios
    share identical variable IDs because they use the same model structure.
    """
    lines: List[str] = []

    for mc in optim_config.models:
        if mc.model_decomposition is None:
            continue
        for var_cfg in mc.model_decomposition.variables:
            if var_cfg.location != ElementLocation.MASTER_AND_SUBPROBLEMS:
                continue
            components = decomposed.subproblem.model_components.get(mc.id, [])
            for comp in components:
                lines.extend(_master_rows(decomposed, mc.id, var_cfg.id, comp.id))
                lines.extend(
                    _subproblem_rows(decomposed, mc.id, var_cfg.id, comp.id, scenarios)
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structure.txt").write_text("\n".join(lines) + "\n")


def main_cli() -> None:
    parsed_args = parse_cli()

    lib_dict = input_libs(parsed_args.models_path)
    study = input_study(parsed_args.components_path, lib_dict)

    models = {}
    for lib in lib_dict.values():
        models.update(lib.models)

    consistency_check(study.components, models)

    try:
        database = input_database(
            parsed_args.components_path, parsed_args.timeseries_path
        )

    except UnboundLocalError:
        raise AntaresTimeSeriesImportError(
            "An error occurred while importing time series."
        )

    network = build_network(study)

    timeblock = TimeBlock(1, list(range(parsed_args.duration)))
    scenario = parsed_args.nb_scenarios

    # Load optional optim-config.yml
    optim_config = load_optim_config(parsed_args.components_path)

    if optim_config is not None:
        validate_optim_config(optim_config, network)

        try:
            decomposed = build_decomposed_problems(
                network, database, timeblock, scenario, optim_config
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
                    scenario,
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
            problem = build_problem(network, database, timeblock, scenario)

        except IndexError as e:
            raise IndexError(
                f"{e}. Did parameters '--duration' and '--scenario' were correctly set?"
            )

        problem.solve(solver_name="highs")
        print("status : ", problem.termination_condition)

        print("final average cost : ", problem.objective_value)


if __name__ == "__main__":
    main_cli()
