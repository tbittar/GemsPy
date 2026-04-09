# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from gems.simulation.simulation_table import (
    SimulationColumns,
    SimulationTableBuilder,
    SimulationTableWriter,
)


@dataclass(frozen=True)
class FakeBlock:
    """Fake time block with an id."""

    id: int = 1


@dataclass
class FakeLinopyVar:
    """Minimal linopy variable stub exposing name and component coords."""

    name: str
    coords: dict  # {"component": xr.DataArray}


@dataclass
class FakeModel:
    """Fake model with no extra outputs."""

    extra_outputs: dict = field(default_factory=dict)


@dataclass
class FakeLinopyModel:
    """Fake linopy model exposing a solution dataset."""

    solution: dict  # lv.name -> xr.DataArray


@dataclass
class FakeProblem:
    """Fake LinopyOptimizationProblem with the attributes used by SimulationTableBuilder."""

    block: FakeBlock = field(default_factory=FakeBlock)
    block_length: int = 3
    objective_value: float = 42.0
    linopy_model: Optional[FakeLinopyModel] = None
    _linopy_vars: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    model_components: dict = field(default_factory=dict)


def test_simulation_table_builder_manual(tmp_path: Path) -> None:
    """Test SimulationTableBuilder and SimulationTableWriter with fake data."""
    sol_da = xr.DataArray(
        np.array([[[10.0], [20.0]]]),
        dims=["component", "time", "scenario"],
        coords={"component": ["compA"], "time": [0, 1], "scenario": [0]},
    )

    fake_var = FakeLinopyVar(
        name="test_model__p",
        coords={"component": xr.DataArray(["compA"])},
    )

    problem = FakeProblem(
        linopy_model=FakeLinopyModel(solution={"test_model__p": sol_da}),
        _linopy_vars={(0, "p"): fake_var},
        models={0: FakeModel()},
        model_components={},
    )

    builder = SimulationTableBuilder(simulation_id="test")
    df = builder.build(problem)  # type: ignore

    expected_rows = [
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: "compA",
            SimulationColumns.OUTPUT: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX: 0,
            SimulationColumns.BLOCK_TIME_INDEX: 0,
            SimulationColumns.SCENARIO_INDEX: 0,
            SimulationColumns.VALUE: 10.0,
            SimulationColumns.BASIS_STATUS: None,
        },
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: "compA",
            SimulationColumns.OUTPUT: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX: 1,
            SimulationColumns.BLOCK_TIME_INDEX: 1,
            SimulationColumns.SCENARIO_INDEX: 0,
            SimulationColumns.VALUE: 20.0,
            SimulationColumns.BASIS_STATUS: None,
        },
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: None,
            SimulationColumns.OUTPUT: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX: None,
            SimulationColumns.BLOCK_TIME_INDEX: None,
            SimulationColumns.SCENARIO_INDEX: None,
            SimulationColumns.VALUE: 42.0,
            SimulationColumns.BASIS_STATUS: None,
        },
    ]
    expected_df = pd.DataFrame(expected_rows)

    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        expected_df,
        check_dtype=False,
    )

    writer = SimulationTableWriter(df)
    csv_path = writer.write_csv(tmp_path, simulation_id="test", optim_nb=1)

    assert csv_path.exists(), "CSV file was not created"

    with csv_path.open("r") as f:
        first_line = f.readline().strip()

    expected_header = ",".join(col.value for col in SimulationColumns)
    assert first_line == expected_header, "CSV header does not match expected columns"

    csv_path.unlink()
