# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from gems.simulation.linopy_problem import LinopyOptimizationProblem
from gems.simulation.output_values import OutputComponent, OutputValues


def _make_mock_problem(
    component_ids: list,
    var_name: str,
    solution_values: xr.DataArray,
) -> MagicMock:
    """Build a minimal mock LinopyOptimizationProblem."""
    mock_problem = MagicMock()
    mock_problem.__class__ = LinopyOptimizationProblem

    # Mock network: all_components
    mock_components = []
    for comp_id in component_ids:
        mock_comp = MagicMock()
        mock_comp.id = comp_id
        mock_comp.model = MagicMock()
        mock_comp.model.extra_outputs = None
        mock_components.append(mock_comp)

    mock_problem.network.all_components = mock_components

    # Mock linopy model solution
    import linopy

    mock_lv = MagicMock(spec=linopy.Variable)
    mock_lv.name = f"test_model__{var_name}"
    # Set up coords so _evaluate_variables can find component IDs.
    component_coord = MagicMock()
    component_coord.values = np.array(component_ids)
    mock_lv.coords = {"component": component_coord}

    mock_problem._linopy_vars = {(0, var_name): mock_lv}
    mock_problem.linopy_model.solution = {mock_lv.name: solution_values}

    return mock_problem


def test_output_values_single_component() -> None:
    """Variables are correctly extracted from the linopy solution."""
    comp_ids = ["comp_1"]
    T, S = 2, 1
    sol = xr.DataArray(
        np.array([[[10.0], [20.0]]]),
        dims=["component", "time", "scenario"],
        coords={"component": comp_ids, "time": range(T), "scenario": range(S)},
    )

    mock_problem = _make_mock_problem(comp_ids, "x", sol)

    with patch.object(OutputComponent, "evaluate_extra_outputs", return_value=None):
        actual = OutputValues(mock_problem)

    expected = OutputValues()
    assert actual != expected

    expected.component("comp_1").var("x").value = 10.0
    # OutputVariable._set stores both t=0 and t=1
    from gems.study.data import TimeScenarioIndex

    expected.component("comp_1").var("x")._set(0, 0, 10.0)
    expected.component("comp_1").var("x")._set(1, 0, 20.0)

    assert actual.component("comp_1").var("x")._value == {
        TimeScenarioIndex(0, 0): 10.0,
        TimeScenarioIndex(1, 0): 20.0,
    }


def test_output_values_ignore_flag() -> None:
    """OutputComponent.ignore allows skipping equality checks."""
    comp_ids = ["comp_1"]
    sol = xr.DataArray(
        np.array([[[5.0]]]),
        dims=["component", "time", "scenario"],
        coords={"component": comp_ids, "time": [0], "scenario": [0]},
    )

    mock_problem = _make_mock_problem(comp_ids, "y", sol)

    with patch.object(OutputComponent, "evaluate_extra_outputs", return_value=None):
        actual = OutputValues(mock_problem)

    empty = OutputValues()
    assert actual != empty

    empty.component("comp_1").ignore = True
    assert actual == empty


def test_output_values_empty() -> None:
    """An OutputValues with no problem is empty and equals another empty one."""
    a = OutputValues()
    b = OutputValues()
    assert a == b
