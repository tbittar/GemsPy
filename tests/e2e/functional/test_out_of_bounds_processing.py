# Copyright (c) 2026, RTE (https://www.rte-france.com)
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

"""
Functional tests for out-of-bounds-processing in optim-config.yml.

The ``state_of_charge_balance`` constraint in the storage_unit model
references ``state_of_charge[t-1]``.  On the first timestep (t=0), this
shift falls outside the current 3-step block.

Two modes are tested:

cyclic (default)
    The shifted term wraps around: ``state_of_charge[-1]`` maps to
    ``state_of_charge[2]``.  The storage state is coupled across the
    full block, so the net dispatch is zero and the generator must
    supply the entire load (50 units/step × 3 steps × cost 10 = 1500).

drop
    The constraint is not instantiated at t=0.  The storage can freely
    dispatch up to 40 units at t=0 (no energy balance required there),
    and continues to satisfy the balance at t=1 and t=2.  The generator
    only needs to cover the gap (50 − 40 = 10 units/step × 3 steps
    × cost 10 = 300).
"""

from pathlib import Path

import pytest

from gems.simulation import TimeBlock, build_decomposed_problems
from gems.study.folder import load_study

STUDIES_DIR = Path(__file__).parent / "studies"


@pytest.mark.parametrize(
    "study_id, expected_objective",
    [
        ("simple_system_cyclic", 1500.0),
        ("simple_system_drop", 300.0),
    ],
)
def test_out_of_bounds_processing(study_id: str, expected_objective: float) -> None:
    study = load_study(STUDIES_DIR / study_id)

    assert (
        study.optim_config is not None
    ), f"optim-config.yml not found in {STUDIES_DIR / study_id / 'input'}"

    # 3-step block matching the study parameters (first-time-step: 0, last-time-step: 2)
    time_block = TimeBlock(1, [0, 1, 2])
    scenarios = 1

    decomposed = build_decomposed_problems(
        study, time_block, scenarios, study.optim_config
    )
    decomposed.subproblem.solve(solver_name="highs")

    assert decomposed.subproblem.termination_condition == "optimal"
    assert decomposed.subproblem.objective_value == pytest.approx(expected_objective)
