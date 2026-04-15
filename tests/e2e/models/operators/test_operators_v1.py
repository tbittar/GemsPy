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

import math
from pathlib import Path

import pandas as pd
import pytest

from gems.simulation import build_problem
from gems.simulation.simulation_table import SimulationTableBuilder
from gems.simulation.time_block import TimeBlock
from gems.study.folder import load_study


@pytest.fixture
def data_dir(request: pytest.FixtureRequest) -> Path:
    return request.param


@pytest.fixture
def results_dir(data_dir: Path) -> Path:
    return data_dir / "output"


@pytest.fixture
def batch() -> int:
    return 1


@pytest.fixture
def relative_accuracy() -> float:
    return 1e-6


@pytest.mark.parametrize(
    "data_dir",
    [
        Path(__file__).parent / "optest1",
        Path(__file__).parent / "optest2",
        Path(__file__).parent / "optest3",
        Path(__file__).parent / "optest4",
    ],
    indirect=True,
)
def test_model_behaviour(
    data_dir: Path,
    results_dir: Path,
    batch: int,
    relative_accuracy: float,
) -> None:
    scenarios = 1

    # Hardcoded timestep range
    first_timestep = 0
    last_timestep = 167
    timesteps = list(range(first_timestep, last_timestep + 1))

    study = load_study(data_dir)

    optim_result_file = results_dir / "simulation_table--20260323-1953.csv"
    df_ref = pd.read_csv(optim_result_file)
    expected_objective = df_ref[df_ref["output"] == "OBJECTIVE_VALUE"]["value"].iloc[0]
    ref_gen3 = (
        df_ref[
            (df_ref["component"] == "unique_prod3") & (df_ref["output"] == "generation")
        ]
        .sort_values("block_time_index")["value"]
        .tolist()
    )

    for _ in range(0, batch):
        problem = build_problem(
            study,
            TimeBlock(1, timesteps),
            scenarios,
        )
        problem.solve(solver_name="highs")
        assert problem.termination_condition == "optimal"
        assert math.isclose(
            expected_objective,
            problem.objective_value,
            rel_tol=relative_accuracy,
        )

        df = SimulationTableBuilder().build(problem)
        gen3_values = (
            df.component("unique_prod3")
            .output("generation")
            .value(scenario_index=0)
            .tolist()
        )

        for t, (ref_val, sol_val) in enumerate(zip(ref_gen3, gen3_values)):
            assert math.isclose(
                ref_val,
                sol_val,
                rel_tol=relative_accuracy,
            ), f"unique_prod3.generation mismatch at timestep {t}: expected {ref_val}, got {sol_val}"
