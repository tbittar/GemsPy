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

"""
Extra output storage and vectorized evaluation.

Provides :class:`ExtraOutput` for storing post-solve expression values and
:class:`VectorizedExtraOutputBuilder`, an xarray-based visitor that evaluates
model-level extra output expressions (potentially nonlinear) over the full
``[component, time, scenario]`` space in one pass.

Port arrays for ``sum_connections`` support are built by calling
:func:`~gems.simulation.linopy_problem.build_port_arrays` with a
:class:`VectorizedExtraOutputBuilder` factory (see ``output_values.py``).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from gems.expression.expression import (
    AdditionNode,
    AllTimeSumNode,
    CeilNode,
    ComparisonNode,
    ComponentParameterNode,
    ComponentVariableNode,
    DivisionNode,
    ExpressionNode,
    FloorNode,
    LiteralNode,
    MaxNode,
    MinNode,
    MultiplicationNode,
    NegationNode,
    ParameterNode,
    PortFieldAggregatorNode,
    PortFieldNode,
    ProblemParameterNode,
    ProblemVariableNode,
    ScenarioOperatorNode,
    TimeEvalNode,
    TimeShiftNode,
    TimeSumNode,
    VariableNode,
)
from gems.expression.visitor import ExpressionVisitor, visit
from gems.model.model import Model
from gems.model.port import PortFieldId
from gems.simulation.linopy_linearize import _da_to_int, _eval_int, _has_dim
from gems.simulation.output_values_base import BaseOutputValue
from gems.study.data import TimeScenarioIndex
from gems.study.network import Component, Network


@dataclass
class ExtraOutput(BaseOutputValue):
    """
    Stores evaluated outputs (from ExpressionNodes), not solver variables.
    Inherits all common fields (_name, _value, _size, ignore) and methods
    (__eq__, is_close, __str__, value property) from BaseOutputValue.
    """

    def _set(
        self,
        timestep: Optional[int],
        scenario: Optional[int],
        value: float,
    ) -> None:
        timestep = 0 if timestep is None else timestep
        scenario = 0 if scenario is None else scenario
        key = TimeScenarioIndex(timestep, scenario)

        if key not in self._value:
            size_s = max(self._size[0], scenario + 1)
            size_t = max(self._size[1], timestep + 1)
            self._size = (size_s, size_t)

        self._value[key] = value


@dataclass
class VectorizedExtraOutputBuilder(ExpressionVisitor[xr.DataArray]):
    """
    Evaluates a model-level extra output expression as a vectorized xr.DataArray.

    Similar to VectorizedLinopyBuilder but returns only xr.DataArray objects
    (no linopy types), enabling nonlinear operations such as products of
    variables, floor, ceil, min, and max.

    Parameters
    ----------
    model_key:
        Python id() of the Model object whose AST is being visited.
    model_name:
        Human-readable model identifier (used in error messages).
    param_arrays:
        Mapping from (model_key, param_name) to a DataArray of parameter values,
        with dims in {component, time, scenario} (or a subset).
    var_solution_arrays:
        Mapping from (model_key, var_name) to a DataArray of solution values,
        with dims in {component, time, scenario} (or a subset).
    port_arrays:
        Pre-computed xr.DataArray for each PortFieldId of this model.
        Keyed by PortFieldId(port_name, field_name).
    block_length:
        Number of time steps in the current time block.
    scenarios_count:
        Number of scenarios.
    """

    model_key: int
    model_name: str
    param_arrays: Dict[Tuple[int, str], xr.DataArray]
    var_solution_arrays: Dict[Tuple[int, str], xr.DataArray]
    port_arrays: Dict[PortFieldId, xr.DataArray]
    block_length: int
    scenarios_count: int

    # ------------------------------------------------------------------ #
    # Leaf nodes                                                            #
    # ------------------------------------------------------------------ #

    def literal(self, node: LiteralNode) -> xr.DataArray:
        return xr.DataArray(node.value)

    def variable(self, node: VariableNode) -> xr.DataArray:
        key = (self.model_key, node.name)
        if key not in self.var_solution_arrays:
            raise KeyError(
                f"Variable {node.name!r} not found in solution for model "
                f"{self.model_name!r}."
            )
        return self.var_solution_arrays[key]

    def parameter(self, node: ParameterNode) -> xr.DataArray:
        key = (self.model_key, node.name)
        if key not in self.param_arrays:
            raise KeyError(
                f"Parameter {node.name!r} not found for model {self.model_name!r}."
            )
        return self.param_arrays[key]

    # ------------------------------------------------------------------ #
    # Arithmetic operators                                                  #
    # ------------------------------------------------------------------ #

    def comparison(self, node: ComparisonNode) -> xr.DataArray:
        raise NotImplementedError(
            "ComparisonNode should not appear in extra output expressions."
        )

    def negation(self, node: NegationNode) -> xr.DataArray:
        return -visit(node.operand, self)  # type: ignore[operator]

    def addition(self, node: AdditionNode) -> xr.DataArray:
        operands = [visit(op, self) for op in node.operands]
        result: xr.DataArray = operands[0]
        for op in operands[1:]:
            result = result + op  # type: ignore[operator]
        return result

    def multiplication(self, node: MultiplicationNode) -> xr.DataArray:
        left = visit(node.left, self)
        right = visit(node.right, self)
        return left * right  # type: ignore[operator]

    def division(self, node: DivisionNode) -> xr.DataArray:
        left = visit(node.left, self)
        right = visit(node.right, self)
        return left / right  # type: ignore[operator]

    # ------------------------------------------------------------------ #
    # Time operators                                                        #
    # ------------------------------------------------------------------ #

    def _apply_time_shift(self, operand: xr.DataArray, shift: int) -> xr.DataArray:
        """Apply a cyclic time shift to operand (same logic as VectorizedLinopyBuilder)."""
        if not _has_dim(operand, "time"):
            return operand
        T = self.block_length
        positions = (np.arange(T) + shift) % T
        indexer = xr.DataArray(positions, dims="time")
        result = operand.isel(time=indexer)
        if "time" in result.coords:
            result = result.assign_coords(time=list(range(T)))
        return result

    def _eval_int_expr(self, node: ExpressionNode) -> int:
        try:
            return _eval_int(node)
        except KeyError:
            result = visit(node, self)
            if isinstance(result, xr.DataArray):
                return _da_to_int(result)
            raise ValueError(
                f"Expected a constant integer expression for time operation, "
                f"got {result!r} from {node!r}."
            )

    def time_shift(self, node: TimeShiftNode) -> xr.DataArray:
        operand = visit(node.operand, self)

        try:
            shift = _eval_int(node.time_shift)
            return self._apply_time_shift(operand, shift)
        except (ValueError, KeyError):
            pass

        shift_result = visit(node.time_shift, self)
        if not isinstance(shift_result, xr.DataArray):
            raise ValueError(
                f"Time shift expression must evaluate to a parameter (DataArray), "
                f"got {type(shift_result).__name__!r}."
            )
        if not shift_result.dims:
            return self._apply_time_shift(operand, _da_to_int(shift_result))

        shift_int = shift_result.astype(int)
        unique_shifts = np.unique(shift_int.values)
        acc: Optional[xr.DataArray] = None
        for s in unique_shifts:
            mask: xr.DataArray = (shift_int == s).astype(float)
            shifted = self._apply_time_shift(operand, int(s))
            contrib: xr.DataArray = shifted * mask  # type: ignore[operator]
            acc = contrib if acc is None else acc + contrib  # type: ignore[operator]
        return acc  # type: ignore[return-value]

    def time_eval(self, node: TimeEvalNode) -> xr.DataArray:
        timestep = self._eval_int_expr(node.eval_time) % self.block_length
        operand = visit(node.operand, self)
        if not _has_dim(operand, "time"):
            return operand
        return operand.isel(time=timestep)

    def time_sum(self, node: TimeSumNode) -> xr.DataArray:
        try:
            from_shift_scalar: Optional[int] = _eval_int(node.from_time)
        except (ValueError, KeyError):
            from_shift_scalar = None

        try:
            to_shift_scalar: Optional[int] = _eval_int(node.to_time)
        except (ValueError, KeyError):
            to_shift_scalar = None

        operand = visit(node.operand, self)

        if from_shift_scalar is not None and to_shift_scalar is not None:
            result: xr.DataArray = self._apply_time_shift(operand, from_shift_scalar)
            for shift in range(from_shift_scalar + 1, to_shift_scalar + 1):
                result = result + self._apply_time_shift(operand, shift)  # type: ignore[operator]
            return result

        from_da = (
            xr.DataArray(float(from_shift_scalar))
            if from_shift_scalar is not None
            else visit(node.from_time, self)
        )
        to_da = (
            xr.DataArray(float(to_shift_scalar))
            if to_shift_scalar is not None
            else visit(node.to_time, self)
        )
        if not isinstance(from_da, xr.DataArray):
            raise ValueError(
                f"time_sum from_time must be a parameter expression (DataArray), "
                f"got {type(from_da).__name__!r}."
            )
        if not isinstance(to_da, xr.DataArray):
            raise ValueError(
                f"time_sum to_time must be a parameter expression (DataArray), "
                f"got {type(to_da).__name__!r}."
            )
        from_int = from_da.astype(int)
        to_int = to_da.astype(int)
        min_from = int(from_int.values.min())
        max_to = int(to_int.values.max())

        acc2: Optional[xr.DataArray] = None
        for shift in range(min_from, max_to + 1):
            shifted = self._apply_time_shift(operand, shift)
            include_from = (
                (from_int <= shift).astype(float)
                if isinstance(from_int, xr.DataArray) and from_int.dims
                else xr.DataArray(1.0)
            )
            include_to = (
                (to_int >= shift).astype(float)
                if isinstance(to_int, xr.DataArray) and to_int.dims
                else xr.DataArray(1.0)
            )
            mask2: xr.DataArray = include_from * include_to  # type: ignore[operator]
            contrib2: xr.DataArray = shifted * mask2  # type: ignore[operator]
            acc2 = contrib2 if acc2 is None else acc2 + contrib2  # type: ignore[operator]
        return acc2  # type: ignore[return-value]

    def all_time_sum(self, node: AllTimeSumNode) -> xr.DataArray:
        operand = visit(node.operand, self)
        if _has_dim(operand, "time"):
            return operand.sum("time")
        return operand * self.block_length  # type: ignore[operator]

    # ------------------------------------------------------------------ #
    # Scenario operators                                                    #
    # ------------------------------------------------------------------ #

    def scenario_operator(self, node: ScenarioOperatorNode) -> xr.DataArray:
        if node.name != "Expectation":
            raise NotImplementedError(
                f"Scenario operator {node.name!r} is not supported. "
                "Only 'Expectation' is currently implemented."
            )
        operand = visit(node.operand, self)
        if _has_dim(operand, "scenario"):
            return operand.sum("scenario") / self.scenarios_count  # type: ignore[operator]
        return operand

    # ------------------------------------------------------------------ #
    # Port fields                                                           #
    # ------------------------------------------------------------------ #

    def port_field(self, node: PortFieldNode) -> xr.DataArray:
        key = PortFieldId(node.port_name, node.field_name)
        if key not in self.port_arrays:
            raise KeyError(
                f"No port array found for {node.port_name}.{node.field_name} "
                f"in model {self.model_name!r}."
            )
        return self.port_arrays[key]

    def port_field_aggregator(self, node: PortFieldAggregatorNode) -> xr.DataArray:
        if node.aggregator != "PortSum":
            raise NotImplementedError(
                f"Port aggregator {node.aggregator!r} is not supported. "
                "Only 'PortSum' is currently implemented."
            )
        if not isinstance(node.operand, PortFieldNode):
            raise ValueError(
                f"PortFieldAggregatorNode operand must be a PortFieldNode, "
                f"got {type(node.operand).__name__!r}."
            )
        port_field_node: PortFieldNode = node.operand
        key = PortFieldId(port_field_node.port_name, port_field_node.field_name)
        if key not in self.port_arrays:
            return xr.DataArray(0.0)
        return self.port_arrays[key]

    # ------------------------------------------------------------------ #
    # Nonlinear math functions (all allowed — no solver constraint)         #
    # ------------------------------------------------------------------ #

    def floor(self, node: FloorNode) -> xr.DataArray:
        operand = visit(node.operand, self)
        return np.floor(operand)  # type: ignore[return-value]

    def ceil(self, node: CeilNode) -> xr.DataArray:
        operand = visit(node.operand, self)
        return np.ceil(operand)  # type: ignore[return-value]

    def maximum(self, node: MaxNode) -> xr.DataArray:
        operands = [visit(op, self) for op in node.operands]
        result: xr.DataArray = operands[0]
        for op in operands[1:]:
            result = xr.where(result >= op, result, op)  # type: ignore[no-untyped-call,assignment]
        return result

    def minimum(self, node: MinNode) -> xr.DataArray:
        operands = [visit(op, self) for op in node.operands]
        result = operands[0]
        for op in operands[1:]:
            result = xr.where(result <= op, result, op)  # type: ignore[no-untyped-call,assignment]
        return result

    # ------------------------------------------------------------------ #
    # Nodes that should not appear in model-level extra output ASTs         #
    # ------------------------------------------------------------------ #

    def comp_parameter(self, node: ComponentParameterNode) -> xr.DataArray:
        raise ValueError(
            f"ComponentParameterNode {node!r} should not appear in a model-level AST."
        )

    def comp_variable(self, node: ComponentVariableNode) -> xr.DataArray:
        raise ValueError(
            f"ComponentVariableNode {node!r} should not appear in a model-level AST."
        )

    def pb_parameter(self, node: ProblemParameterNode) -> xr.DataArray:
        raise ValueError(
            f"ProblemParameterNode {node!r} should not appear in a model-level AST."
        )

    def pb_variable(self, node: ProblemVariableNode) -> xr.DataArray:
        raise ValueError(
            f"ProblemVariableNode {node!r} should not appear in a model-level AST."
        )
