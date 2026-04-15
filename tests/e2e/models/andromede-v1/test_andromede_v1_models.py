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
from gems.simulation.time_block import TimeBlock
from gems.study.folder import load_study

_STUDIES_DIR = Path(__file__).parent / "studies"
_RESULTS_DIR = Path(__file__).parent / "results"


@pytest.mark.parametrize(
    "study_name, optim_result_file, timespan, batch, relative_accuracy",
    [
        (
            "dsr",
            "dsr_case.csv",
            168,
            20,
            1e-6,
        ),
        (
            "base",
            "base_case.csv",
            168,
            20,
            1e-6,
        ),
        (
            "electrolyser",
            "electrolyser_case.csv",
            168,
            20,
            1e-6,
        ),
        (
            "storage",
            "storage_case.csv",
            168,
            20,
            1e-6,
        ),
        (
            "bde",
            "bde_case.csv",
            168,
            20,
            1e-6,
        ),
        (
            "cluster1",
            "cluster_testing1.csv",
            168,
            20,
            1e-6,
        ),
        (
            "cluster2",
            "cluster_testing2.csv",
            168,
            20,
            1e-4,  # Default Fico XPRESS Tolerance
        ),
    ],
)
def test_model_behaviour(
    study_name: str,
    optim_result_file: str,
    timespan: int,
    batch: int,
    relative_accuracy: float,
) -> None:
    scenarios = 1
    study = load_study(_STUDIES_DIR / study_name)
    reference_values = pd.read_csv(_RESULTS_DIR / optim_result_file, header=None).values
    for k in range(batch):
        problem = build_problem(
            study,
            TimeBlock(1, [i for i in range(k * timespan, (k + 1) * timespan)]),
            scenarios,
        )
        problem.solve(solver_name="highs")
        assert problem.termination_condition == "optimal"
        assert math.isclose(
            reference_values[k, 0],
            problem.objective_value,
            rel_tol=relative_accuracy,
        )
