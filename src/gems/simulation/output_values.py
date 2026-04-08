# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

"""
Utility classes to obtain solver results from a linopy-based optimization problem.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import xarray as xr

from gems.expression.visitor import visit
from gems.simulation.extra_output import VectorizedExtraOutputBuilder
from gems.simulation.linopy_problem import LinopyOptimizationProblem, build_port_arrays
from gems.simulation.output_values_base import ExtraOutput, OutputVariable

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _scalar(da: xr.DataArray) -> float:
    """Extract a Python float from a scalar (0-d) DataArray."""
    return float(da.item())  # type: ignore[arg-type]


def _da_to_value(
    da: xr.DataArray,
) -> Union[float, List[float], List[List[float]]]:
    """Convert a [time?, scenario?] DataArray (no component dim) to scalar/list/list-of-list.

    Mirrors the old ``BaseOutputValue.value`` getter shape convention:
    - (T=1, S=1) → scalar float
    - (T=1, S>1) → list of floats indexed by scenario
    - (T>1, S>=1) → list of lists indexed by [scenario][time]
    """
    T = da.sizes.get("time", 1)
    S = da.sizes.get("scenario", 1)

    if T == 1 and S == 1:
        return _scalar(da)

    if T == 1:
        return [
            _scalar(da.isel(scenario=s) if "scenario" in da.dims else da)
            for s in range(S)
        ]

    return [
        [
            _scalar(
                da.isel(scenario=s, time=t)
                if "scenario" in da.dims
                else da.isel(time=t)
            )
            for t in range(T)
        ]
        for s in range(S)
    ]


def _value_to_da(
    values: Union[float, List[float], List[List[float]]], comp_id: str
) -> xr.DataArray:
    """Build a [component, time, scenario] DataArray from a scalar/list/list-of-list.

    Mirrors the old ``BaseOutputValue.value`` setter shape convention:
    - float            → (1 comp, 1 time, 1 scenario)
    - list of floats   → (1 comp, 1 time, S scenarios)  — indexed by scenario
    - list of lists    → (1 comp, T times, S scenarios) — indexed by [scenario][time]
    """
    if not isinstance(values, list):
        arr = np.array([[[float(values)]]])
    else:
        n_scenarios = len(values)
        if n_scenarios > 0 and isinstance(values[0], list):
            nested = cast(List[List[float]], values)
            n_times = len(nested[0])
            arr = np.array(
                [[[nested[s][t] for s in range(n_scenarios)] for t in range(n_times)]]
            )
        else:
            flat = cast(List[float], values)
            arr = np.array([[[float(v) for v in flat]]])

    return xr.DataArray(
        arr,
        dims=["component", "time", "scenario"],
        coords={
            "component": [comp_id],
            "time": list(range(arr.shape[1])),
            "scenario": list(range(arr.shape[2])),
        },
    )


def _model_has_data_for_comp(model: "OutputModel", comp_id: str) -> bool:
    """Return True if *model* has any non-None variable data for *comp_id*."""
    for var in model._variables.values():
        if var._data is None:
            continue
        if "component" not in var._data.dims:
            return True
        if comp_id in var._data.coords["component"].values:
            return True
    return False


# ---------------------------------------------------------------------------
# OutputModel
# ---------------------------------------------------------------------------


@dataclass
class OutputModel:
    """Groups all OutputVariable and ExtraOutput DataArrays for one GEMS model.

    One ``OutputModel`` instance covers **all components** of a GEMS model,
    with the component dimension encoded inside each DataArray.
    """

    _model_id: str
    _variables: Dict[str, OutputVariable] = field(init=False, default_factory=dict)
    _extra_outputs: Dict[str, ExtraOutput] = field(init=False, default_factory=dict)
    ignore: bool = field(default=False, init=False)

    def var(self, name: str) -> OutputVariable:
        if name not in self._variables:
            self._variables[name] = OutputVariable(name)
        return self._variables[name]

    def extra_output(self, name: str) -> ExtraOutput:
        if name not in self._extra_outputs:
            self._extra_outputs[name] = ExtraOutput(name)
        return self._extra_outputs[name]


# ---------------------------------------------------------------------------
# Per-component views (backward-compat adapters)
# ---------------------------------------------------------------------------


class VarOutputView:
    """Exposes a component-sliced read/write view of an :class:`OutputVariable`."""

    def __init__(self, var: OutputVariable, comp_id: str) -> None:
        self._var = var
        self._comp_id = comp_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sliced_da(self) -> Optional[xr.DataArray]:
        """Return the DataArray sliced to this component (no component dim), or None."""
        if self._var._data is None:
            return None
        da = self._var._data
        if "component" in da.dims:
            comp_vals = da.coords["component"].values
            if self._comp_id not in comp_vals:
                return None
            return da.sel(component=self._comp_id, drop=True)
        return da

    def _is_close_to(
        self, other: "VarOutputView", rel_tol: float, abs_tol: float
    ) -> bool:
        lhs_da = self._sliced_da()
        rhs_da = other._sliced_da()
        if lhs_da is None and rhs_da is None:
            return True
        if lhs_da is None or rhs_da is None:
            return False
        try:
            lhs_a, rhs_a = xr.align(lhs_da, rhs_da, join="exact")
            return bool(
                np.allclose(lhs_a.values, rhs_a.values, rtol=rel_tol, atol=abs_tol)
            )
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def value(self) -> Union[None, float, List[float], List[List[float]]]:
        da = self._sliced_da()
        if da is None:
            return None
        return _da_to_value(da)

    @value.setter
    def value(self, values: Union[float, List[float], List[List[float]]]) -> None:
        new_da = _value_to_da(values, self._comp_id)
        if self._var._data is None:
            self._var._data = new_da
        else:
            existing = self._var._data
            if "component" in existing.dims:
                comp_vals = existing.coords["component"].values
                if self._comp_id in comp_vals:
                    existing = existing.drop_sel(component=self._comp_id)
            self._var._data = xr.concat([existing, new_da], dim="component")


class ExtraOutputView:
    """Exposes a component-sliced read/write view of an :class:`ExtraOutput`."""

    def __init__(self, eo: ExtraOutput, comp_id: str) -> None:
        self._eo = eo
        self._comp_id = comp_id

    def _sliced_da(self) -> Optional[xr.DataArray]:
        if self._eo._data is None:
            return None
        da = self._eo._data
        if "component" in da.dims:
            comp_vals = da.coords["component"].values
            if self._comp_id not in comp_vals:
                return None
            return da.sel(component=self._comp_id, drop=True)
        return da

    def _is_close_to(
        self, other: "ExtraOutputView", rel_tol: float, abs_tol: float
    ) -> bool:
        lhs_da = self._sliced_da()
        rhs_da = other._sliced_da()
        if lhs_da is None and rhs_da is None:
            return True
        if lhs_da is None or rhs_da is None:
            return False
        try:
            lhs_a, rhs_a = xr.align(lhs_da, rhs_da, join="exact")
            return bool(
                np.allclose(lhs_a.values, rhs_a.values, rtol=rel_tol, atol=abs_tol)
            )
        except ValueError:
            return False

    @property
    def value(self) -> Union[None, float, List[float], List[List[float]]]:
        da = self._sliced_da()
        if da is None:
            return None
        return _da_to_value(da)

    @value.setter
    def value(self, values: Union[float, List[float], List[List[float]]]) -> None:
        new_da = _value_to_da(values, self._comp_id)
        if self._eo._data is None:
            self._eo._data = new_da
        else:
            existing = self._eo._data
            if "component" in existing.dims:
                comp_vals = existing.coords["component"].values
                if self._comp_id in comp_vals:
                    existing = existing.drop_sel(component=self._comp_id)
            self._eo._data = xr.concat([existing, new_da], dim="component")


class ComponentOutputView:
    """Backward-compat adapter: sliced view of an :class:`OutputModel` for one component.

    Returned by :meth:`OutputValues.component`.  Supports the same
    ``var(name).value``, ``extra_output(name).value``, and ``ignore`` API
    that the old ``OutputComponent`` provided.
    """

    def __init__(self, ov: "OutputValues", comp_id: str) -> None:
        self._ov = ov
        self._comp_id = comp_id

    @property
    def _model(self) -> OutputModel:
        return self._ov._models[self._ov._comp_to_model_key[self._comp_id]]

    @property
    def ignore(self) -> bool:
        return self._model.ignore

    @ignore.setter
    def ignore(self, val: bool) -> None:
        self._model.ignore = val

    def var(self, name: str) -> VarOutputView:
        return VarOutputView(self._model.var(name), self._comp_id)

    def extra_output(self, name: str) -> ExtraOutputView:
        return ExtraOutputView(self._model.extra_output(name), self._comp_id)

    def _is_close_to(
        self, other: "ComponentOutputView", rel_tol: float, abs_tol: float
    ) -> bool:
        if self._model.ignore or other._model.ignore:
            return True

        all_var_names = set(self._model._variables) | set(other._model._variables)
        for var_name in all_var_names:
            lv = VarOutputView(self._model.var(var_name), self._comp_id)
            rv = VarOutputView(other._model.var(var_name), other._comp_id)
            if not lv._is_close_to(rv, rel_tol, abs_tol):
                return False

        all_eo_names = set(self._model._extra_outputs) | set(
            other._model._extra_outputs
        )
        for eo_name in all_eo_names:
            lv_eo = ExtraOutputView(self._model.extra_output(eo_name), self._comp_id)
            rv_eo = ExtraOutputView(other._model.extra_output(eo_name), other._comp_id)
            if not lv_eo._is_close_to(rv_eo, rel_tol, abs_tol):
                return False

        return True


# ---------------------------------------------------------------------------
# OutputValues
# ---------------------------------------------------------------------------


@dataclass
class OutputValues:
    """Contains variables and extra outputs after solver completion.

    If constructed with a :class:`~gems.simulation.linopy_problem.LinopyOptimizationProblem`,
    variable solution values are extracted from ``linopy_model.solution`` and extra
    outputs are evaluated vectorized over ``[component, time, scenario]``.

    If constructed without arguments, an empty container is created (useful for
    building expected values in tests via ``component(id).var(name).value = ...``).
    """

    problem: Optional[LinopyOptimizationProblem] = field(default=None)
    _models: Dict[int, OutputModel] = field(init=False, default_factory=dict)
    _comp_to_model_key: Dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._build_models()
        self._fill_models()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputValues):
            return NotImplemented
        return self.is_close(other, rel_tol=0.0, abs_tol=0.0)

    def is_close(
        self, other: "OutputValues", *, rel_tol: float = 1.0e-9, abs_tol: float = 0.0
    ) -> bool:
        lhs_comps = set(self._comp_to_model_key)
        rhs_comps = set(other._comp_to_model_key)

        for comp_id in lhs_comps - rhs_comps:
            lv_model = self._models[self._comp_to_model_key[comp_id]]
            if lv_model.ignore:
                continue
            if _model_has_data_for_comp(lv_model, comp_id):
                return False

        for comp_id in rhs_comps - lhs_comps:
            rv_model = other._models[other._comp_to_model_key[comp_id]]
            if rv_model.ignore:
                continue
            if _model_has_data_for_comp(rv_model, comp_id):
                return False

        for comp_id in lhs_comps & rhs_comps:
            lv = ComponentOutputView(self, comp_id)
            rv = ComponentOutputView(other, comp_id)
            if not lv._is_close_to(rv, rel_tol, abs_tol):
                return False

        return True

    def __str__(self) -> str:
        lines = ["\n"]
        for model in self._models.values():
            for var in model._variables.values():
                lines.append(f"  {var}\n")
            for eo in model._extra_outputs.values():
                lines.append(f"  {eo}\n")
        return "".join(lines)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def component(self, comp_id: str) -> ComponentOutputView:
        """Return a per-component view (backward-compat accessor).

        If *comp_id* is not yet registered (e.g. when building expected values),
        a synthetic :class:`OutputModel` is created automatically.
        """
        if comp_id not in self._comp_to_model_key:
            key = hash(comp_id)
            self._models[key] = OutputModel(comp_id)
            self._comp_to_model_key[comp_id] = key
        return ComponentOutputView(self, comp_id)

    def model(self, model_key: int) -> OutputModel:
        """Return the :class:`OutputModel` for the given model key (``id(Model)``)."""
        return self._models[model_key]

    # ------------------------------------------------------------------
    # Initialisation (called from __post_init__)
    # ------------------------------------------------------------------

    def _build_models(self) -> None:
        if self.problem is None:
            return
        for cmp in self.problem.network.all_components:
            mk = id(cmp.model)
            if mk not in self._models:
                self._models[mk] = OutputModel(cmp.model.id)
            self._comp_to_model_key[cmp.id] = mk

    def _fill_models(self) -> None:
        self._evaluate_variables()
        self._evaluate_extra_outputs()

    # ------------------------------------------------------------------
    # Variable extraction
    # ------------------------------------------------------------------

    def _evaluate_variables(self) -> None:
        """Assign solution DataArrays directly to OutputVariable._data."""
        if self.problem is None:
            return

        solution = self.problem.linopy_model.solution
        if solution is None:
            return

        for (mk, var_name), lv in self.problem._linopy_vars.items():
            lv_name = lv.name
            if lv_name not in solution:
                continue

            sol_da: xr.DataArray = solution[lv_name]

            # Filter to the variable's own components (avoid NaN-padding from
            # the outer-join solution Dataset).
            own_components = list(lv.coords["component"].values)
            filtered_da = sol_da.sel(component=own_components)

            self._models[mk].var(var_name)._data = filtered_da

    # ------------------------------------------------------------------
    # Extra output evaluation
    # ------------------------------------------------------------------

    def _evaluate_extra_outputs(self) -> None:
        """Evaluate model extra outputs and store DataArrays directly."""
        if self.problem is None:
            return
        problem = self.problem  # narrow Optional for mypy + lambda capture

        var_solution_arrays: Dict[Tuple[int, str], xr.DataArray] = {}
        solution = problem.linopy_model.solution
        if solution is not None:
            for (mk, vname), lv in problem._linopy_vars.items():
                if lv.name in solution:
                    var_solution_arrays[(mk, vname)] = solution[lv.name]

        for mk, model in problem.models.items():
            if not model.extra_outputs:
                continue
            components = problem.model_components[mk]

            port_arrays = build_port_arrays(
                model,
                components,
                problem.models,
                problem.model_components,
                problem.network,
                lambda mk_, m: VectorizedExtraOutputBuilder(
                    model_key=mk_,
                    model_name=m.id,
                    param_arrays=problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
                    port_arrays={},
                    block_length=problem.block_length,
                    scenarios_count=problem.scenarios,
                ),
            )

            for out_id, expr_node in model.extra_outputs.items():
                builder = VectorizedExtraOutputBuilder(
                    model_key=mk,
                    model_name=model.id,
                    param_arrays=problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
                    port_arrays=port_arrays,
                    block_length=problem.block_length,
                    scenarios_count=problem.scenarios,
                )
                result_da: xr.DataArray = visit(expr_node, builder)

                # Filter to own component IDs if the component dim is present
                if "component" in result_da.dims:
                    own_ids = [c.id for c in components]
                    present = [
                        c for c in own_ids if c in result_da.coords["component"].values
                    ]
                    result_da = result_da.sel(component=present)

                self._models[mk].extra_output(out_id)._data = result_da
