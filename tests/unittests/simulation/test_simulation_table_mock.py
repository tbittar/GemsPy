# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from gems.simulation.output_values import OutputModel, OutputValues
from gems.simulation.output_values_base import OutputVariable
from gems.simulation.simulation_table import (
    SimulationColumns,
    SimulationTableBuilder,
    SimulationTableWriter,
)


@dataclass(frozen=True)
class FakeBlock:
    """Fake time block with an id."""

    id: int = 1


@dataclass(frozen=True)
class FakeProblem:
    """Fake problem that binds block and objective value."""

    block: FakeBlock = FakeBlock()
    block_length: int = 3
    objective_value: float = 42.0


class FakeOutputValues(OutputValues):
    """OutputValues backed by pre-built OutputModel instances."""

    def __init__(
        self,
        problem: FakeProblem,
        models: dict,
        comp_to_model_key: dict,
    ) -> None:
        self.problem = problem  # type: ignore[assignment]
        self._models = models
        self._comp_to_model_key = comp_to_model_key


def test_simulation_table_builder_manual(tmp_path):
    """Test SimulationTableBuilder and SimulationTableWriter with fake data."""
    problem = FakeProblem()

    var = OutputVariable("p")
    var._data = xr.DataArray(
        np.array([[[10.0], [20.0]]]),
        dims=["component", "time", "scenario"],
        coords={"component": ["compA"], "time": [0, 1], "scenario": [0]},
    )
    var._basis_status = xr.DataArray(
        np.array([[["BASIC"], ["NONBASIC"]]], dtype=object),
        dims=["component", "time", "scenario"],
        coords={"component": ["compA"], "time": [0, 1], "scenario": [0]},
    )

    model_out = OutputModel("test_model")
    model_out._variables["p"] = var

    output_values = FakeOutputValues(
        problem=problem,
        models={0: model_out},
        comp_to_model_key={"compA": 0},
    )

    builder = SimulationTableBuilder(simulation_id="test")
    df = builder.build(output_values)  # type: ignore

    expected_rows = [
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: "compA",
            SimulationColumns.OUTPUT: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX: 0,
            SimulationColumns.BLOCK_TIME_INDEX: 0,
            SimulationColumns.SCENARIO_INDEX: 0,
            SimulationColumns.VALUE: 10.0,
            SimulationColumns.BASIS_STATUS: "BASIC",
        },
        {
            SimulationColumns.BLOCK: 1,
            SimulationColumns.COMPONENT: "compA",
            SimulationColumns.OUTPUT: "p",
            SimulationColumns.ABSOLUTE_TIME_INDEX: 1,
            SimulationColumns.BLOCK_TIME_INDEX: 1,
            SimulationColumns.SCENARIO_INDEX: 0,
            SimulationColumns.VALUE: 20.0,
            SimulationColumns.BASIS_STATUS: "NONBASIC",
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
