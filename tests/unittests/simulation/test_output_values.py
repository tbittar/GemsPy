# Copyright (c) 2024, RTE (https://www.rte-france.com)
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

import numpy as np
import xarray as xr

from gems.simulation.linopy_problem import LinopyOptimizationProblem
from gems.simulation.output_values import OutputValues


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

    # Extra output evaluation skipped when models is empty.
    mock_problem.models = {}
    mock_problem.model_components = {}
    mock_problem.param_arrays = {}

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
    actual = OutputValues(mock_problem)

    expected = OutputValues()
    assert actual != expected

    from gems.study.data import TimeScenarioIndex

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


def test_extra_output_with_sum_connections() -> None:
    """
    Extra output using sum_connections is evaluated correctly via
    VectorizedExtraOutputBuilder + _build_port_arrays_xarray.

    Setup: gen_1 (GEN model, variable gen=5) connects to node_1 (NODE model).
    NODE model has extra output 'total_flow' = sum_connections(balance_port.flow).
    GEN model defines balance_port.flow = var("gen").
    Expected: total_flow at node_1 == 5.0.
    """
    from gems.expression import var
    from gems.expression.expression import literal, port_field
    from gems.model.model import ModelPort, model
    from gems.model.port import PortField, PortFieldDefinition, PortFieldId, PortType
    from gems.model.variable import float_variable
    from gems.simulation import TimeBlock, build_problem
    from gems.study import (
        DataBase,
        Network,
        Node,
        PortRef,
        TimeScenarioIndex,
        create_component,
    )

    BALANCE_PORT_TYPE = PortType(id="balance", fields=[PortField("flow")])

    # Generator: variable gen fixed to 5, exposes it as port flow.
    GEN_MODEL = model(
        id="GEN_EXTRA",
        variables=[
            float_variable("gen", lower_bound=literal(5), upper_bound=literal(5))
        ],
        ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
        port_fields_definitions=[
            PortFieldDefinition(
                port_field=PortFieldId("balance_port", "flow"),
                definition=var("gen"),
            )
        ],
    )

    # Node: slave port, extra output = sum of incoming flows (no binding constraint).
    NODE_MODEL = model(
        id="NODE_EXTRA",
        ports=[ModelPort(port_type=BALANCE_PORT_TYPE, port_name="balance_port")],
        extra_outputs={
            "total_flow": port_field("balance_port", "flow").sum_connections()
        },
    )

    database = DataBase()

    gen_comp = create_component(model=GEN_MODEL, id="gen_1")
    node_comp = Node(model=NODE_MODEL, id="node_1")

    network = Network("test_sum_connections")
    network.add_component(gen_comp)
    network.add_node(node_comp)
    network.connect(
        PortRef(gen_comp, "balance_port"), PortRef(node_comp, "balance_port")
    )

    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios=1)
    problem.solve(solver_name="highs")

    output = OutputValues(problem)

    assert output.component("node_1").extra_output("total_flow")._value == {
        TimeScenarioIndex(0, 0): 5.0
    }


def test_extra_output_nonlinear() -> None:
    """
    Nonlinear extra output (var * var) is correctly evaluated.

    VectorizedExtraOutputBuilder allows products of variables since extra
    outputs are not solver constraints. Equivalent VectorizedLinopyBuilder
    would raise NotImplementedError for the same expression.

    Setup: one component with variable a=3 (fixed). Extra output squared = a*a.
    Expected: squared = 9.0.
    """
    from gems.expression import var
    from gems.expression.expression import literal
    from gems.model.model import model
    from gems.model.variable import float_variable
    from gems.simulation import TimeBlock, build_problem
    from gems.study import DataBase, Network, TimeScenarioIndex, create_component

    SIMPLE_MODEL = model(
        id="SIMPLE_NL",
        variables=[float_variable("a", lower_bound=literal(3), upper_bound=literal(3))],
        extra_outputs={"squared": var("a") * var("a")},
    )

    database = DataBase()
    comp = create_component(model=SIMPLE_MODEL, id="comp_1")

    network = Network("test_nonlinear")
    network.add_component(comp)

    problem = build_problem(network, database, TimeBlock(1, [0]), scenarios=1)
    problem.solve(solver_name="highs")

    output = OutputValues(problem)

    assert output.component("comp_1").extra_output("squared")._value == {
        TimeScenarioIndex(0, 0): 9.0
    }
