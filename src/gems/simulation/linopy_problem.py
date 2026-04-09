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
Linopy-based optimization problem builder.

Provides :func:`build_problem` and :class:`LinopyOptimizationProblem`.
The builder groups network components by model, then constructs the full
optimization problem in four phases:

1. Parameter arrays — convert database values to xarray DataArrays indexed
   on ``[component, time, scenario]``.
2. Decision variables — create one linopy ``Variable`` per model variable,
   covering all components of that model at once.
3. Port arrays — resolve port connections via an incidence matrix so that
   port-field expressions are available as linopy ``LinearExpression`` objects.
4. Constraints and objective — traverse each constraint AST once with
   :class:`~gems.simulation.linopy_linearize.VectorizedLinopyBuilder` to
   produce vectorized linopy constraints added in a single
   ``Model.add_constraints()`` call per constraint type.
"""

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import linopy
import numpy as np
import xarray as xr

from gems.expression.expression import is_unbounded
from gems.expression.visitor import visit
from gems.model.common import ValueType
from gems.model.model import Model
from gems.model.port import PortField, PortFieldId
from gems.simulation.linopy_linearize import (
    LinopyExpression,
    VectorizedLinopyBuilder,
    _linopy_add,
)
from gems.simulation.time_block import TimeBlock
from gems.study.data import ConstantData, DataBase, ScenarioSeriesData, TimeSeriesData
from gems.study.network import Component, Network


def build_port_arrays(
    model: Model,
    components: List[Component],
    models: Dict[str, Model],
    model_components: Dict[str, List[Component]],
    network: "Network",
    make_builder: Callable[[str, Model], Any],
) -> Dict[PortFieldId, Any]:
    """Build port arrays for all ports of *model*.

    For each PortFieldId (port_name, field_name):
    - If *model* defines the field (master): evaluate the definition with
      ``make_builder(model.id, model)``.
    - Otherwise (slave): sum contributions from connected master components
      via incidence matrices.

    Parameters
    ----------
    model :
        The model for which to build port arrays.
    components :
        Components of this model.
    models :
        All models keyed by ``model.id``.
    model_components :
        Components grouped by ``model.id``.
    network :
        The network, used for connection lookup.
    make_builder :
        Factory ``(model_key: str, model: Model) -> builder``.
        Called with an empty port_arrays context for master-field evaluation.
    """
    comp_ids = [c.id for c in components]
    n = len(components)
    port_arrays: Dict[PortFieldId, Any] = {}

    for port_name, model_port in model.ports.items():
        for port_field_obj in model_port.port_type.fields:
            field_name = port_field_obj.name
            pf_id = PortFieldId(port_name, field_name)

            if pf_id in model.port_fields_definitions:
                builder = make_builder(model.id, model)
                defn = model.port_fields_definitions[pf_id].definition
                port_arrays[pf_id] = visit(defn, builder)
            else:
                port_arrays[pf_id] = _build_slave_port_array(
                    comp_ids,
                    n,
                    port_name,
                    field_name,
                    models,
                    model_components,
                    network,
                    make_builder,
                )

    return port_arrays


def _build_slave_port_array(
    comp_ids: List[str],
    n_components: int,
    port_name: str,
    field_name: str,
    models: Dict[str, Model],
    model_components: Dict[str, List[Component]],
    network: "Network",
    make_builder: Callable[[str, Model], Any],
) -> Any:
    """Build a slave port array by summing contributions from connected masters.

    Groups connections by (master_model.id, master_port_field_id), builds an
    incidence matrix A[i, j] for each group, and accumulates
    ``sum_j A[i,j] * expr_master[j]`` into the result.
    """
    per_master: Dict[
        Tuple[str, PortFieldId], List[Tuple[int, Component]]
    ] = defaultdict(list)

    comp_index = {comp_id: i for i, comp_id in enumerate(comp_ids)}
    comp_id_set = set(comp_ids)
    for cnx in network.connections:
        for port_ref in [cnx.port1, cnx.port2]:
            if (
                port_ref.port_id != port_name
                or port_ref.component.id not in comp_id_set
            ):
                continue
            i = comp_index[port_ref.component.id]
            master_ref = cnx.master_port.get(PortField(name=field_name))
            if master_ref is None:
                continue
            master_comp = master_ref.component
            master_pf_id = PortFieldId(master_ref.port_id, field_name)
            per_master[(master_comp.model.id, master_pf_id)].append((i, master_comp))

    if not per_master:
        return xr.DataArray(0.0)

    total: Optional[Any] = None

    for (master_mk, master_pf_id), conn_list in per_master.items():
        master_comps = model_components[master_mk]
        master_comp_ids = [c.id for c in master_comps]
        n_prime = len(master_comps)

        A_data = np.zeros((n_components, n_prime))
        for i, master_comp in conn_list:
            j = master_comp_ids.index(master_comp.id)
            A_data[i, j] += 1.0

        A = xr.DataArray(
            A_data,
            dims=["component", "component_master"],
            coords={"component": comp_ids, "component_master": master_comp_ids},
        )

        master_model = models[master_mk]
        defn = master_model.port_fields_definitions[master_pf_id].definition
        master_builder = make_builder(master_mk, master_model)
        expr_master = visit(defn, master_builder)

        expr_master_r = expr_master.rename({"component": "component_master"})  # type: ignore[union-attr]
        contribution = (A * expr_master_r).sum("component_master")  # type: ignore[operator]

        total = contribution if total is None else _linopy_add(total, contribution)

    return total if total is not None else xr.DataArray(0.0)


class BlockBorderManagement(Enum):
    """
    Specifies how the time horizon border is handled.

    - CYCLE: All time steps are addressed modulo the horizon length (Antares default).
    - IGNORE_OUT_OF_FRAME: Terms leading to out-of-horizon data are ignored.
    """

    CYCLE = "CYCLE"
    IGNORE_OUT_OF_FRAME = "IGNORE"


class LinopyOptimizationProblem:
    """
    Wraps a linopy.Model and provides the high-level API for solving and
    extracting results.
    """

    def __init__(
        self,
        name: str,
        linopy_model: linopy.Model,
        network: Network,
        database: DataBase,
        block: TimeBlock,
        scenarios: int,
        linopy_vars: Dict[Tuple[str, str], linopy.Variable],
        param_arrays: Dict[Tuple[str, str], xr.DataArray],
        model_components: Dict[str, List[Component]],
        models: Dict[str, Model],
        objective_constant: float = 0.0,
    ) -> None:
        self.name = name
        self.linopy_model = linopy_model
        self.network = network
        self.database = database
        self.block = block
        self.scenarios = scenarios
        self._linopy_vars = linopy_vars
        self.param_arrays = param_arrays
        self.model_components = model_components
        self.models = models
        # Constant term of the objective (linopy cannot represent pure-constant objectives).
        self._objective_constant: float = objective_constant

    @property
    def block_length(self) -> int:
        return len(self.block.timesteps)

    def solve(self, solver_name: str = "highs", **kwargs: object) -> None:
        """Solve the problem using the specified solver."""
        # Use io_api="direct" to bypass LP file writing and avoid LP name parsing
        # issues in linopy's set_int_index (e.g. constraint names with spaces or
        # variables with non-standard characters).
        kwargs.setdefault("io_api", "direct")  # type: ignore[call-overload]
        self.linopy_model.solve(solver_name=solver_name, **kwargs)  # type: ignore[arg-type]

    @property
    def status(self) -> str:
        """Solver status: 'ok' (optimal found) or 'warning' (infeasible / other)."""
        return str(self.linopy_model.status)

    @property
    def termination_condition(self) -> str:
        """Termination condition string, e.g., 'optimal', 'infeasible'."""
        return str(self.linopy_model.termination_condition)

    @property
    def objective_value(self) -> float:
        """Objective function value after solving."""
        return float(self.linopy_model.objective.value) + self._objective_constant  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------


class _LinopyProblemBuilder:
    """
    Builds the linopy problem in 4 phases:
      1. Build parameter DataArrays for all models.
      2. Create all linopy Variables (uses param arrays for bounds).
      3. Build port arrays via incidence matrices.
      4. Add constraints and objectives to the linopy model.
    """

    def __init__(
        self,
        name: str,
        network: Network,
        database: DataBase,
        block: TimeBlock,
        scenarios: int,
    ) -> None:
        self.name = name
        self.network = network
        self.database = database
        self.block = block
        self.scenarios = scenarios

        self.block_length = len(block.timesteps)
        self.time_coord = list(range(self.block_length))
        self.scenario_coord = list(range(scenarios))

        # Populated during build
        self.linopy_model = linopy.Model()
        self.linopy_vars: Dict[Tuple[str, str], linopy.Variable] = {}
        self.param_arrays: Dict[Tuple[str, str], xr.DataArray] = {}
        self.port_arrays: Dict[str, Dict[PortFieldId, LinopyExpression]] = {}

        # Group components by model.id.
        self.model_components: Dict[str, List[Component]] = defaultdict(list)
        self.models: Dict[str, Model] = {}
        for component in network.all_components:
            m = component.model
            mk = m.id
            if mk not in self.models:
                self.models[mk] = m
            self.model_components[mk].append(component)

    def build(self) -> LinopyOptimizationProblem:
        # Phase 1: parameter arrays
        for mk, components in self.model_components.items():
            self._build_param_arrays_for_model(self.models[mk], components)

        # Phase 2: linopy variables
        for mk, components in self.model_components.items():
            self._create_variables_for_model(self.models[mk], components)

        # Phase 3: port arrays
        for mk, components in self.model_components.items():
            self._build_port_arrays_for_model(self.models[mk], components)

        # Phase 4: constraints + objectives
        total_obj: Optional[LinopyExpression] = None
        for mk in self.model_components.keys():
            model = self.models[mk]
            port_arrays_for_model = self.port_arrays.get(mk, {})
            self._create_constraints_for_model(model, port_arrays_for_model)
            total_obj = self._add_objectives_for_model(
                model, port_arrays_for_model, total_obj
            )

        # Extract constant objective contribution (linopy cannot hold pure constants).
        objective_constant = 0.0
        if total_obj is not None and not isinstance(
            total_obj, (xr.DataArray, int, float)
        ):
            self.linopy_model.add_objective(total_obj)  # type: ignore[arg-type]
        elif total_obj is not None:
            if isinstance(total_obj, xr.DataArray):
                objective_constant = float(total_obj.sum())
            else:
                objective_constant = float(total_obj)

        # linopy requires at least one variable to solve; add a fixed dummy if needed.
        if len(self.linopy_model.variables) == 0:
            dummy = self.linopy_model.add_variables(
                lower=xr.DataArray([0.0], dims=["__dummy_dim"]),
                upper=xr.DataArray([0.0], dims=["__dummy_dim"]),
                name="__dummy",
            )
            self.linopy_model.add_objective(0 * dummy)  # type: ignore[operator]

        return LinopyOptimizationProblem(
            name=self.name,
            linopy_model=self.linopy_model,
            network=self.network,
            database=self.database,
            block=self.block,
            scenarios=self.scenarios,
            linopy_vars=self.linopy_vars,
            param_arrays=self.param_arrays,
            model_components=dict(self.model_components),
            models=self.models,
            objective_constant=objective_constant,
        )

    # ------------------------------------------------------------------
    # Phase 1 — Parameter arrays
    # ------------------------------------------------------------------

    def _build_param_arrays_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        T = self.block_length
        S = self.scenarios
        C = len(components)
        comp_ids = [c.id for c in components]
        abs_timesteps = self.block.timesteps  # mapping: block_t → abs_t

        for param in model.parameters.values():
            use_time = param.structure.time
            use_scenario = param.structure.scenario

            # Determine the minimal shape for this parameter based on its
            # declared structure. Using minimal shapes avoids spurious broadcasting
            # (e.g., invest_cost * p_max should not gain time/scenario dims).
            if use_time and use_scenario:
                data = np.zeros((C, T, S))
                dims = ["component", "time", "scenario"]
                coords: Dict[str, object] = {
                    "component": comp_ids,
                    "time": self.time_coord,
                    "scenario": self.scenario_coord,
                }
            elif use_time:
                data = np.zeros((C, T))
                dims = ["component", "time"]
                coords = {"component": comp_ids, "time": self.time_coord}
            elif use_scenario:
                data = np.zeros((C, S))
                dims = ["component", "scenario"]
                coords = {"component": comp_ids, "scenario": self.scenario_coord}
            else:
                data = np.zeros((C,))
                dims = ["component"]
                coords = {"component": comp_ids}

            for i, c in enumerate(components):
                param_data = self.database.get_data(c.id, param.name)
                if isinstance(param_data, ConstantData):
                    data[i] = param_data.value  # broadcasts into remaining dims
                elif isinstance(param_data, TimeSeriesData):
                    v = param_data.get_value(abs_timesteps, None)
                    if use_time and use_scenario:
                        data[i, :, :] = v[:, np.newaxis]  # broadcast T across S
                    elif use_time:
                        data[i, :] = v
                    else:
                        data[i] = v  # constant in time
                elif isinstance(param_data, ScenarioSeriesData):
                    for s in range(S):
                        v = param_data.get_value(None, s)  # type: ignore[assignment]
                        if use_time and use_scenario:
                            data[i, :, s] = v
                        elif use_scenario:
                            data[i, s] = v
                        else:
                            data[i] = v  # constant in scenario
                else:
                    # TimeScenarioSeriesData, TreeData, or other
                    for s in range(S):
                        v = param_data.get_value(  # type: ignore[assignment]
                            abs_timesteps, s
                        )
                        if use_time and use_scenario:
                            data[i, :, s] = v
                        elif use_time:
                            data[i, :] = v
                        elif use_scenario:
                            data[i, s] = v
                        else:
                            data[i] = v  # take any single value

            arr = xr.DataArray(data, dims=dims, coords=coords)
            self.param_arrays[(model.id, param.name)] = arr

    # ------------------------------------------------------------------
    # Phase 2 — Variables
    # ------------------------------------------------------------------

    def _create_variables_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        comp_ids = [c.id for c in components]

        for var in model.variables.values():
            # Build coords for this variable
            coords: Dict[str, object] = {"component": comp_ids}
            dims = ["component"]
            if var.structure.time:
                coords["time"] = self.time_coord
                dims.append("time")
            if var.structure.scenario:
                coords["scenario"] = self.scenario_coord
                dims.append("scenario")

            # Shape of this variable (used to broadcast scalar bounds)
            var_shape = tuple(
                (
                    len(comp_ids)
                    if d == "component"
                    else (
                        len(self.time_coord)
                        if d == "time"
                        else len(self.scenario_coord)
                    )
                )
                for d in dims
            )

            # Build a minimal builder for bound expressions (no variables needed)
            bound_builder = VectorizedLinopyBuilder(
                model_id=model.id,
                linopy_vars={},
                param_arrays=self.param_arrays,
                port_arrays={},
                block_length=self.block_length,
                scenarios_count=self.scenarios,
            )

            lower: object = (
                np.full(var_shape, -np.inf)
                if var.lower_bound is None
                else self._to_bound_array(
                    visit(var.lower_bound, bound_builder), var_shape, dims
                )
            )
            upper: object = (
                np.full(var_shape, np.inf)
                if var.upper_bound is None
                else self._to_bound_array(
                    visit(var.upper_bound, bound_builder), var_shape, dims
                )
            )

            # Validate bounds: upper must be >= lower across all timesteps/scenarios
            lower_arr = lower if isinstance(lower, np.ndarray) else np.array(lower)
            upper_arr = upper if isinstance(upper, np.ndarray) else np.array(upper)
            for ci, comp_id in enumerate(comp_ids):
                lo = np.asarray(lower_arr[ci] if lower_arr.ndim > 0 else lower_arr)
                up = np.asarray(upper_arr[ci] if upper_arr.ndim > 0 else upper_arr)
                finite = np.isfinite(lo) & np.isfinite(up)
                violation = finite & (up < lo)
                if np.any(violation):
                    idx = int(np.argmax(violation))
                    raise ValueError(
                        f"Upper bound ({float(up.flat[idx]):g}) must be strictly "
                        f"greater than lower bound ({float(lo.flat[idx]):g}) "
                        f"for variable {comp_id}.{var.name}"
                    )

            prefix = model.id.replace("-", "_")
            name = f"{prefix}__{var.name}"
            binary = var.data_type == ValueType.BOOLEAN
            integer = var.data_type == ValueType.INTEGER

            lv = self.linopy_model.add_variables(
                lower=lower,
                upper=upper,
                coords=coords,
                name=name,
                binary=binary,
                integer=integer,
            )
            self.linopy_vars[(model.id, var.name)] = lv

    # ------------------------------------------------------------------
    # Phase 3 — Port arrays
    # ------------------------------------------------------------------

    def _build_port_arrays_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        self.port_arrays[model.id] = build_port_arrays(
            model,
            components,
            self.models,
            dict(self.model_components),
            self.network,
            lambda mk_, m: self._make_builder(m, port_arrays={}),
        )

    # ------------------------------------------------------------------
    # Phase 4 — Constraints and Objectives
    # ------------------------------------------------------------------

    def _create_constraints_for_model(
        self,
        model: Model,
        port_arrays_for_model: Dict[PortFieldId, LinopyExpression],
    ) -> None:
        """Add all constraints for *model* to the linopy model."""
        builder = self._make_builder(model, port_arrays=port_arrays_for_model)

        prefix = model.id.replace("-", "_")
        for constraint in model.get_all_constraints():
            lhs = visit(constraint.expression, builder)

            # Sanitize constraint name for LP format (spaces → underscores)
            safe_name = constraint.name.replace(" ", "_").replace("-", "_")

            # Lower bound constraint: lhs >= lb  (if lb != -inf)
            if not is_unbounded(constraint.lower_bound):
                lb = visit(constraint.lower_bound, builder)
                name = f"{prefix}__{safe_name}__lb"
                con_lb = lhs >= lb  # type: ignore[operator]
                self.linopy_model.add_constraints(con_lb, name=name)  # type: ignore[arg-type]

            # Upper bound constraint: lhs <= ub  (if ub != +inf)
            if not is_unbounded(constraint.upper_bound):
                ub = visit(constraint.upper_bound, builder)
                name = f"{prefix}__{safe_name}__ub"
                con_ub = lhs <= ub  # type: ignore[operator]
                self.linopy_model.add_constraints(con_ub, name=name)  # type: ignore[arg-type]

    def _add_objectives_for_model(
        self,
        model: Model,
        port_arrays_for_model: Dict[PortFieldId, LinopyExpression],
        total_obj: Optional[LinopyExpression],
    ) -> Optional[LinopyExpression]:
        """Accumulate objective contributions from *model*."""
        builder = self._make_builder(model, port_arrays=port_arrays_for_model)

        def _accumulate(
            acc: Optional[LinopyExpression], contribution: LinopyExpression
        ) -> LinopyExpression:
            if isinstance(contribution, xr.DataArray):
                summed: LinopyExpression = float(contribution.sum().item())  # type: ignore[assignment]
            else:
                summed = contribution.sum()  # type: ignore[union-attr]
            return summed if acc is None else _linopy_add(acc, summed)

        if model.objective_contributions:
            for expr in model.objective_contributions.values():
                if expr is not None:
                    obj_term = visit(expr, builder)
                    total_obj = _accumulate(total_obj, obj_term)

        return total_obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bound_array(
        val: object,
        var_shape: Tuple[int, ...],
        dims: List[str],
    ) -> np.ndarray:
        """Convert a bound value to a numpy array shaped like *var_shape*.

        Handles scalar DataArrays, partial-dim DataArrays (e.g. a constant
        parameter used as a bound for a time×scenario variable), plain floats,
        and raw numpy arrays.
        """
        if isinstance(val, xr.DataArray):
            if val.dims == ():
                return np.full(var_shape, float(val.item()))
            arr = val.values  # shape may be a subset of var_shape dims
            for ax, d in enumerate(dims):
                if d not in val.dims:
                    arr = np.expand_dims(arr, axis=ax)
            return np.broadcast_to(arr, var_shape).copy()  # type: ignore[return-value]
        if isinstance(val, (int, float)):
            return np.full(var_shape, float(val))
        return val  # type: ignore[return-value]

    def _make_builder(
        self,
        model: Model,
        port_arrays: Dict[PortFieldId, LinopyExpression],
    ) -> VectorizedLinopyBuilder:
        return VectorizedLinopyBuilder(
            model_id=model.id,
            linopy_vars=self.linopy_vars,
            param_arrays=self.param_arrays,
            port_arrays=port_arrays,
            block_length=self.block_length,
            scenarios_count=self.scenarios,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_problem(
    network: Network,
    database: DataBase,
    block: TimeBlock,
    scenarios: int,
    *,
    problem_name: str = "optimization_problem",
    border_management: BlockBorderManagement = BlockBorderManagement.CYCLE,
) -> LinopyOptimizationProblem:
    """
    Build and return a LinopyOptimizationProblem for the given time block.

    Parameters
    ----------
    network:
        Network of components and connections.
    database:
        Parameter data for all components.
    block:
        The time block to optimize.
    scenarios:
        Number of scenarios.
    problem_name:
        Label for the linopy model.
    border_management:
        How to handle time steps at block borders (only CYCLE is implemented).
    """
    if border_management != BlockBorderManagement.CYCLE:
        raise NotImplementedError(
            f"Border management {border_management} is not yet implemented. "
            "Only BlockBorderManagement.CYCLE is supported."
        )

    database.requirements_consistency(network)

    builder = _LinopyProblemBuilder(
        name=problem_name,
        network=network,
        database=database,
        block=block,
        scenarios=scenarios,
    )
    return builder.build()
