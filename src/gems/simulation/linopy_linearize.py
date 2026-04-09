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
Vectorized linopy expression builder.

Provides :class:`VectorizedLinopyBuilder`, a pre-solve
:class:`~gems.expression.visitor.ExpressionVisitor` that traverses a
model-level AST once and produces a linopy ``LinearExpression`` covering all
components × all time steps × all scenarios simultaneously.

The shared visitor logic (time operators, port fields, arithmetic defaults)
lives in :class:`~gems.simulation.vectorized_builder.VectorizedBuilderBase`.
This class only adds the linopy-specific field and the six methods that differ
from the DataArray-only default: ``variable``, ``addition``, ``floor``,
``ceil``, ``maximum``, and ``minimum``.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import linopy
import xarray as xr
import numpy as np

from gems.expression.expression import (
    AdditionNode,
    CeilNode,
    FloorNode,
    MaxNode,
    MinNode,
    VariableNode,
)
from gems.expression.visitor import visit
from gems.model.port import PortFieldId
from gems.simulation.vectorized_builder import (
    LinopyExpression,
    VectorizedBuilderBase,
    _linopy_add,
)

# Re-export so that existing ``from gems.simulation.linopy_linearize import
# LinopyExpression, _linopy_add`` call sites (e.g. linopy_problem.py) keep
# working without change.
__all__ = [
    "LinopyExpression",
    "VectorizedLinopyBuilder",
    "_linopy_add",
]


@dataclass(kw_only=True)
class VectorizedLinopyBuilder(VectorizedBuilderBase[LinopyExpression]):
    """
    Builds a linopy LinearExpression from a model-level AST.

    Receives pre-computed linopy variables, parameter DataArrays, and port
    arrays indexed on the ``[component, time, scenario]`` dimensions (or a
    subset thereof).  Produces a result with the same combined dimensions.

    Inherits all visitor logic from
    :class:`~gems.simulation.vectorized_builder.VectorizedBuilderBase`.
    Only the six methods that differ from the DataArray-only default are
    overridden here:

    - :meth:`variable` — returns the ``linopy.Variable`` directly.
    - :meth:`addition` — ensures linopy types are on the left-hand side.
    - :meth:`floor`, :meth:`ceil`, :meth:`maximum`, :meth:`minimum` — guard
      against nonlinear use of decision variables (not expressible in LP).

    Parameters
    ----------
    model_key:
        Python ``id()`` of the ``Model`` object whose AST is being visited.
        Using ``id()`` (not ``model.id`` string) ensures two distinct ``Model``
        objects with the same ``.id`` string (e.g. two ``"GEN"`` models) are
        never confused.
    model_name:
        Human-readable model identifier (``model.id``), used only for error
        messages.
    linopy_vars:
        Mapping from ``(model_key, var_name)`` to the corresponding linopy
        Variable, with dims at least ``[component]`` and optionally
        ``[time, scenario]``.
    param_arrays:
        Mapping from ``(model_key, param_name)`` to a DataArray of parameter
        values, with dims in ``{component, time, scenario}`` (or a subset).
    port_arrays:
        Pre-computed linopy expressions for each ``PortFieldId`` of this model,
        resulting from the incidence-matrix port resolution pass.
        Keyed by ``PortFieldId(port_name, field_name)``.
    block_length:
        Number of time steps in the current time block.
    scenarios_count:
        Number of scenarios.
    """

    linopy_vars: Dict[Tuple[int, str], linopy.Variable]

    # ------------------------------------------------------------------ #
    # Leaf nodes                                                            #
    # ------------------------------------------------------------------ #

    def variable(self, node: VariableNode) -> linopy.Variable:
        key = (self.model_key, node.name)
        if key not in self.linopy_vars:
            raise KeyError(
                f"Variable {node.name!r} not found for model {self.model_name!r}. "
                "Ensure all linopy variables are created before building constraints."
            )
        return self.linopy_vars[key]

    # ------------------------------------------------------------------ #
    # Arithmetic operators                                                  #
    # ------------------------------------------------------------------ #

    def addition(self, node: AdditionNode) -> LinopyExpression:
        operands = [visit(op, self) for op in node.operands]
        result: LinopyExpression = operands[0]
        for op in operands[1:]:
            result = _linopy_add(result, op)
        return result

    # ------------------------------------------------------------------ #
    # Math functions — guarded: nonlinear ops on linopy types are invalid  #
    # ------------------------------------------------------------------ #

    def floor(self, node: FloorNode) -> LinopyExpression:
        operand = visit(node.operand, self)
        if isinstance(operand, xr.DataArray):
            return np.floor(operand)  # type: ignore[return-value]
        raise NotImplementedError(
            "floor() is only supported for parameter (DataArray) expressions "
            "in a linopy constraint context."
        )

    def ceil(self, node: CeilNode) -> LinopyExpression:
        operand = visit(node.operand, self)
        if isinstance(operand, xr.DataArray):
            return np.ceil(operand)  # type: ignore[return-value]
        raise NotImplementedError(
            "ceil() is only supported for parameter (DataArray) expressions "
            "in a linopy constraint context."
        )

    def maximum(self, node: MaxNode) -> LinopyExpression:
        operands = [visit(op, self) for op in node.operands]
        if all(isinstance(op, xr.DataArray) for op in operands):
            result: xr.DataArray = operands[0]  # type: ignore[assignment]
            for op in operands[1:]:
                result = xr.where(result >= op, result, op)  # type: ignore[no-untyped-call]
            return result
        raise NotImplementedError(
            "maximum() is only supported for parameter (DataArray) expressions "
            "in a linopy constraint context."
        )

    def minimum(self, node: MinNode) -> LinopyExpression:
        operands = [visit(op, self) for op in node.operands]
        if all(isinstance(op, xr.DataArray) for op in operands):
            result = operands[0]
            for op in operands[1:]:
                result = xr.where(result <= op, result, op)  # type: ignore[no-untyped-call,assignment]
            return result
        raise NotImplementedError(
            "minimum() is only supported for parameter (DataArray) expressions "
            "in a linopy constraint context."
        )
