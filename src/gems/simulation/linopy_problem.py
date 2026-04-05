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

Replaces the OR-Tools based OptimizationProblem / build_problem pipeline with
a vectorized linopy pipeline that processes all components of a model in a
single pass, instead of iterating per (component, time-step, scenario).
"""

from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import linopy
import numpy as np
import xarray as xr

from gems.expression.evaluate import EvaluationContext, EvaluationVisitor
from gems.expression.expression import ExpressionNode, is_unbounded
from gems.expression.indexing import IndexingStructureProvider, compute_indexation
from gems.expression.indexing_structure import IndexingStructure
from gems.expression.operators_expansion import ProblemDimensions, expand_operators
from gems.expression.visitor import visit
from gems.model.common import ValueType
from gems.model.model import Model
from gems.model.port import PortField, PortFieldId
from gems.simulation.linopy_linearize import LinopyExpression, VectorizedLinopyBuilder
from gems.simulation.strategy import (
    MergedProblemStrategy,
    ModelSelectionStrategy,
    RiskManagementStrategy,
    UniformRisk,
)
from gems.simulation.time_block import TimeBlock
from gems.study.data import (
    ConstantData,
    DataBase,
    ScenarioSeriesData,
    TimeScenarioSeriesData,
    TimeSeriesData,
    TreeData,
)
from gems.study.network import Component, Network, PortsConnection


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
        linopy_vars: Dict[Tuple[int, str], linopy.Variable],
        build_strategy: ModelSelectionStrategy,
        decision_tree_node: str,
        objective_constant: float = 0.0,
    ) -> None:
        self.name = name
        self.linopy_model = linopy_model
        self.network = network
        self.database = database
        self.block = block
        self.scenarios = scenarios
        self._linopy_vars = linopy_vars
        self._build_strategy = build_strategy
        self._decision_tree_node = decision_tree_node
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

    def get_solution(self, model_id: str, var_name: str) -> xr.DataArray:
        """
        Return the solution DataArray for a given (model_id, var_name).

        Dims: a subset of [component, time, scenario].
        """
        lv = self._linopy_vars.get((model_id, var_name))
        if lv is None:
            raise KeyError(
                f"No linopy variable found for ({model_id!r}, {var_name!r})."
            )
        return self.linopy_model.solution[lv.name]

    def expand_operators_for_extra_output(
        self,
        expression: ExpressionNode,
        component_id: str,
    ) -> ExpressionNode:
        """
        Expand temporal and scenario operators for extra output evaluation.
        Replicates the role of OptimizationContext.expand_operators().
        """
        from gems.expression.context_adder import add_component_context

        structure_provider = _make_network_structure_provider(self.network)
        with_context = add_component_context(component_id, expression)
        return expand_operators(
            with_context,
            ProblemDimensions(self.block_length, self.scenarios),
            _make_constant_evaluator(),
            structure_provider,
        )


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
        build_strategy: ModelSelectionStrategy,
        risk_strategy: RiskManagementStrategy,
        decision_tree_node: str,
    ) -> None:
        self.name = name
        self.network = network
        self.database = database
        self.block = block
        self.scenarios = scenarios
        self.build_strategy = build_strategy
        self.risk_strategy = risk_strategy
        self.decision_tree_node = decision_tree_node

        self.block_length = len(block.timesteps)
        self.time_coord = list(range(self.block_length))
        self.scenario_coord = list(range(scenarios))

        # Populated during build
        self.linopy_model = linopy.Model()
        # Keys use id(model) (int) so two distinct Model objects with the same .id
        # string are never confused (e.g. GENERATOR_MODEL vs thermal_candidate both
        # having model.id == "GEN").
        self.linopy_vars: Dict[Tuple[int, str], linopy.Variable] = {}
        self.param_arrays: Dict[Tuple[int, str], xr.DataArray] = {}
        self.port_arrays: Dict[int, Dict[PortFieldId, LinopyExpression]] = {}

        # Group components by model object identity.
        # model_var_prefix gives each unique Model object a distinct linopy name prefix
        # so two models with the same .id string (e.g. "GEN") don't collide.
        self.model_components: Dict[int, List[Component]] = defaultdict(list)
        self.models: Dict[int, Model] = {}
        self.model_var_prefix: Dict[int, str] = {}
        _id_usage: Dict[str, int] = defaultdict(int)
        for component in network.all_components:
            m = component.model
            mk = id(m)
            if mk not in self.models:
                count = _id_usage[m.id]
                _id_usage[m.id] += 1
                suffix = f"_{count}" if count > 0 else ""
                self.model_var_prefix[mk] = (m.id + suffix).replace("-", "_")
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
        for mk, components in self.model_components.items():
            model = self.models[mk]
            port_arrays_for_model = self.port_arrays.get(mk, {})
            total_obj = self._create_constraints_for_model(
                model, components, port_arrays_for_model, total_obj
            )
            total_obj = self._add_objectives_for_model(
                model, components, port_arrays_for_model, total_obj
            )

        # Extract constant objective contribution (linopy cannot hold pure constants).
        objective_constant = 0.0
        if total_obj is not None and not isinstance(total_obj, (xr.DataArray, int, float)):
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
            build_strategy=self.build_strategy,
            decision_tree_node=self.decision_tree_node,
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
                    for t in range(T):
                        v = param_data.get_value(
                            abs_timesteps[t], None, self.decision_tree_node
                        )
                        if use_time and use_scenario:
                            data[i, t, :] = v
                        elif use_time:
                            data[i, t] = v
                        else:
                            data[i] = v  # constant in time
                elif isinstance(param_data, ScenarioSeriesData):
                    for s in range(S):
                        v = param_data.get_value(None, s, self.decision_tree_node)
                        if use_time and use_scenario:
                            data[i, :, s] = v
                        elif use_scenario:
                            data[i, s] = v
                        else:
                            data[i] = v  # constant in scenario
                else:
                    # TimeScenarioSeriesData, TreeData, or other
                    for t in range(T):
                        for s in range(S):
                            v = param_data.get_value(
                                abs_timesteps[t], s, self.decision_tree_node
                            )
                            if use_time and use_scenario:
                                data[i, t, s] = v
                            elif use_time:
                                data[i, t] = v
                            elif use_scenario:
                                data[i, s] = v
                            else:
                                data[i] = v  # take any single value

            arr = xr.DataArray(data, dims=dims, coords=coords)
            self.param_arrays[(id(model), param.name)] = arr

    # ------------------------------------------------------------------
    # Phase 2 — Variables
    # ------------------------------------------------------------------

    def _create_variables_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        comp_ids = [c.id for c in components]

        for var in self.build_strategy.get_variables(model):
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
                len(comp_ids) if d == "component"
                else len(self.time_coord) if d == "time"
                else len(self.scenario_coord)
                for d in dims
            )

            # Build a minimal builder for bound expressions (no variables needed)
            bound_builder = VectorizedLinopyBuilder(
                model_key=id(model),
                model_name=model.id,
                linopy_vars={},
                param_arrays=self.param_arrays,
                port_arrays={},
                block_length=self.block_length,
                scenarios_count=self.scenarios,
            )

            def _to_bound_array(val: object) -> np.ndarray:
                """Convert a bound value to a numpy array shaped like var_shape.

                Handles scalar DataArrays, partial-dim DataArrays (e.g. a
                constant parameter used as bound for a time×scenario variable),
                plain floats, and raw numpy arrays.
                """
                if isinstance(val, xr.DataArray):
                    if val.dims == ():
                        return np.full(var_shape, float(val.item()))
                    # Expand missing dims so numpy can broadcast to var_shape.
                    arr = val.values  # shape may be a subset of var_shape dims
                    for ax, d in enumerate(dims):
                        if d not in val.dims:
                            arr = np.expand_dims(arr, axis=ax)
                    return np.broadcast_to(arr, var_shape).copy()  # type: ignore[return-value]
                if isinstance(val, (int, float)):
                    return np.full(var_shape, float(val))
                return val  # type: ignore[return-value]

            # Lower bound
            lower: object
            if var.lower_bound is None:
                lower = np.full(var_shape, -np.inf)
            else:
                lower = _to_bound_array(visit(var.lower_bound, bound_builder))

            # Upper bound
            upper: object
            if var.upper_bound is None:
                upper = np.full(var_shape, np.inf)
            else:
                upper = _to_bound_array(visit(var.upper_bound, bound_builder))

            # Validate bounds: upper must be strictly > lower for each component
            lower_arr = lower if isinstance(lower, np.ndarray) else np.array(lower)
            upper_arr = upper if isinstance(upper, np.ndarray) else np.array(upper)
            for ci, comp_id in enumerate(comp_ids):
                # Slice first element if multi-dim, else use scalar
                lo_val = float(lower_arr[ci].flat[0]) if lower_arr.ndim > 0 else float(lower_arr)
                up_val = float(upper_arr[ci].flat[0]) if upper_arr.ndim > 0 else float(upper_arr)
                if not np.isinf(lo_val) and not np.isinf(up_val) and up_val < lo_val:
                    raise ValueError(
                        f"Upper bound ({up_val:g}) must be strictly greater than "
                        f"lower bound ({lo_val:g}) for variable {comp_id}.{var.name}"
                    )

            prefix = self.model_var_prefix[id(model)]
            name = f"{prefix}__{var.name}"
            binary = var.data_type == ValueType.BOOLEAN
            integer = var.data_type in (ValueType.INTEGER, ValueType.BOOLEAN)

            lv = self.linopy_model.add_variables(
                lower=lower,
                upper=upper,
                coords=coords,
                name=name,
                binary=binary,
                integer=integer and not binary,
            )
            self.linopy_vars[(id(model), var.name)] = lv

    # ------------------------------------------------------------------
    # Phase 3 — Port arrays
    # ------------------------------------------------------------------

    def _build_port_arrays_for_model(
        self, model: Model, components: List[Component]
    ) -> None:
        """
        Pre-compute port arrays for all ports of *model*.

        For each PortFieldId (port_name, field_name) that M owns a port for:
          - If M defines the field (M is master): evaluate definition directly.
          - If a connected model M' defines the field: use incidence matrix A to
            compute sum_{j} A[i,j] * expr_M'[j] for all i.
        """
        comp_ids = [c.id for c in components]
        n = len(components)
        port_arrays: Dict[PortFieldId, LinopyExpression] = {}

        for port_name, model_port in model.ports.items():
            for port_field_obj in model_port.port_type.fields:
                field_name = port_field_obj.name
                pf_id = PortFieldId(port_name, field_name)

                if pf_id in model.port_fields_definitions:
                    # M is the master — evaluate definition using M's own vars/params
                    builder = self._make_builder(model, port_arrays={})
                    defn = model.port_fields_definitions[pf_id].definition
                    port_arrays[pf_id] = visit(defn, builder)
                else:
                    # M is the slave — collect contributions from connected masters
                    port_arrays[pf_id] = self._build_slave_port_array(
                        model, comp_ids, n, port_name, field_name
                    )

        self.port_arrays[id(model)] = port_arrays

    def _build_slave_port_array(
        self,
        model: Model,
        comp_ids: List[str],
        n: int,
        port_name: str,
        field_name: str,
    ) -> LinopyExpression:
        """
        Build port array by summing contributions from all connected master components,
        using an incidence matrix for each contributing master model.
        """
        # Group connections by (id(master_model), master_port_field_id).
        # Using id(master_model) instead of master_model.id ensures two distinct Model
        # objects with the same .id string are never confused.
        # Grouping by master_pf_id is critical: a component can connect via different
        # ports (e.g. link.in_port and link.out_port) which have different definitions.
        per_master: Dict[
            Tuple[int, PortFieldId], List[Tuple[int, Component]]
        ] = defaultdict(list)

        for i, comp_m in enumerate(comp_ids):
            for cnx in self.network.connections:
                if not _involves(cnx, comp_m, port_name):
                    continue
                master_ref = cnx.master_port.get(PortField(name=field_name))
                if master_ref is None:
                    continue
                master_comp = master_ref.component
                master_pf_id = PortFieldId(master_ref.port_id, field_name)
                per_master[(id(master_comp.model), master_pf_id)].append(
                    (i, master_comp)
                )

        if not per_master:
            return xr.DataArray(0.0)

        total: Optional[LinopyExpression] = None

        for (master_mk, master_pf_id), conn_list in per_master.items():
            master_comps = self.model_components[master_mk]
            master_comp_ids = [c.id for c in master_comps]
            n_prime = len(master_comps)

            # Incidence matrix A[i, j] = 1 if master_comps[j] connects to comp_ids[i]
            A_data = np.zeros((n, n_prime))
            for i, master_comp in conn_list:
                j = master_comp_ids.index(master_comp.id)
                A_data[i, j] += 1.0

            A = xr.DataArray(
                A_data,
                dims=["component", "component_master"],
                coords={"component": comp_ids, "component_master": master_comp_ids},
            )

            # Visit the master's port field definition for this specific port
            master_model = self.models[master_mk]
            defn = master_model.port_fields_definitions[master_pf_id].definition
            master_builder = self._make_builder(master_model, port_arrays={})
            expr_master = visit(defn, master_builder)

            # Rename master's 'component' dim to 'component_master' for broadcasting
            expr_master_r = expr_master.rename({"component": "component_master"})  # type: ignore[union-attr]

            contribution = (A * expr_master_r).sum("component_master")  # type: ignore[operator]

            total = (
                contribution
                if total is None
                else _linopy_add(total, contribution)
            )

        return total if total is not None else xr.DataArray(0.0)

    # ------------------------------------------------------------------
    # Phase 4 — Constraints and Objectives
    # ------------------------------------------------------------------

    def _create_constraints_for_model(
        self,
        model: Model,
        components: List[Component],
        port_arrays_for_model: Dict[PortFieldId, LinopyExpression],
        total_obj: Optional[LinopyExpression],
    ) -> Optional[LinopyExpression]:
        """Add all constraints for *model* to the linopy model."""
        builder = self._make_builder(model, port_arrays=port_arrays_for_model)

        prefix = self.model_var_prefix[id(model)]
        for constraint in self.build_strategy.get_constraints(model):
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

        return total_obj

    def _add_objectives_for_model(
        self,
        model: Model,
        components: List[Component],
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
                    modified = self.risk_strategy(expr)
                    obj_term = visit(modified, builder)
                    total_obj = _accumulate(total_obj, obj_term)
        else:
            for obj_expr in self.build_strategy.get_objectives(model):
                if obj_expr is not None:
                    modified = self.risk_strategy(obj_expr)
                    obj_term = visit(modified, builder)
                    total_obj = _accumulate(total_obj, obj_term)

        return total_obj

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _make_builder(
        self,
        model: Model,
        port_arrays: Dict[PortFieldId, LinopyExpression],
    ) -> VectorizedLinopyBuilder:
        return VectorizedLinopyBuilder(
            model_key=id(model),
            model_name=model.id,
            linopy_vars=self.linopy_vars,
            param_arrays=self.param_arrays,
            port_arrays=port_arrays,
            block_length=self.block_length,
            scenarios_count=self.scenarios,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linopy_add(
    a: LinopyExpression, b: LinopyExpression
) -> LinopyExpression:
    """
    Add two linopy-compatible expressions, ensuring linopy types are on the left.

    xarray's DataArray.__add__ doesn't know about linopy types, so mixing
    DataArray + LinearExpression fails unless linopy is the left operand.
    """
    if isinstance(a, xr.DataArray) and not isinstance(b, xr.DataArray):
        return b + a  # type: ignore[operator]  # linopy on left
    return a + b  # type: ignore[operator]


def _involves(cnx: PortsConnection, component_id: str, port_name: str) -> bool:
    """Return True if *cnx* connects *component_id* at *port_name*."""
    return (
        cnx.port1.component.id == component_id and cnx.port1.port_id == port_name
    ) or (cnx.port2.component.id == component_id and cnx.port2.port_id == port_name)


def _make_network_structure_provider(network: Network) -> IndexingStructureProvider:
    """Create an IndexingStructureProvider backed by the network."""

    class _Provider(IndexingStructureProvider):
        def get_component_variable_structure(
            self, component_id: str, name: str
        ) -> IndexingStructure:
            return network.get_component(component_id).model.variables[name].structure

        def get_component_parameter_structure(
            self, component_id: str, name: str
        ) -> IndexingStructure:
            return network.get_component(component_id).model.parameters[name].structure

        def get_parameter_structure(self, name: str) -> IndexingStructure:
            raise RuntimeError(
                "Component context must be set before retrieving parameter structure."
            )

        def get_variable_structure(self, name: str) -> IndexingStructure:
            raise RuntimeError(
                "Component context must be set before retrieving variable structure."
            )

    return _Provider()


def _make_constant_evaluator() -> "Callable[[ExpressionNode], int]":
    """Return an evaluator that resolves only literal constant expressions."""
    ctx = EvaluationContext()
    visitor = EvaluationVisitor(ctx)

    def _evaluate(node: ExpressionNode) -> int:
        result = visit(node, visitor)
        if isinstance(result, float) and result.is_integer():
            return int(result)
        raise ValueError(f"Expected integer literal, got {result!r}")

    return _evaluate


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
    build_strategy: ModelSelectionStrategy = MergedProblemStrategy(),
    risk_strategy: RiskManagementStrategy = UniformRisk(),
    decision_tree_node: str = "",
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
    build_strategy:
        Selects which variables and constraints to include.
    risk_strategy:
        Modifies objective expressions for risk management.
    decision_tree_node:
        Node identifier when operating within a decision tree.
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
        build_strategy=build_strategy,
        risk_strategy=risk_strategy,
        decision_tree_node=decision_tree_node,
    )
    return builder.build()
