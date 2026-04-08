# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

"""
Utility classes to obtain solver results from a linopy-based optimization problem.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, TypeVar

import xarray as xr

from gems.expression.visitor import visit
from gems.simulation.extra_output import (
    ExtraOutput,
    VectorizedExtraOutputBuilder,
    _build_port_arrays_xarray,
)
from gems.simulation.linopy_problem import LinopyOptimizationProblem
from gems.simulation.output_values_base import BaseOutputValue
from gems.study.data import TimeScenarioIndex


@dataclass
class OutputVariable(BaseOutputValue):
    """
    Contains a single solver variable's values and status.
    All shared logic is in BaseOutputValue.
    """

    _basis_status: Dict[TimeScenarioIndex, str] = field(
        init=False, default_factory=dict
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputVariable):
            return NotImplemented
        if not super().__eq__(other):
            return False
        return (self.ignore or other.ignore) or (
            self._basis_status == other._basis_status
        )

    def _set(
        self,
        timestep: Optional[int],
        scenario: Optional[int],
        value: float,
        status: Optional[str] = None,
        is_mip: bool = True,
    ) -> None:
        timestep = 0 if timestep is None else timestep
        scenario = 0 if scenario is None else scenario
        key = TimeScenarioIndex(timestep, scenario)
        if key not in self._value:
            size_s = max(self._size[0], scenario + 1)
            size_t = max(self._size[1], timestep + 1)
            self._size = (size_s, size_t)
        self._value[key] = value
        if not is_mip and status is not None:
            self._basis_status[key] = status


@dataclass
class OutputComponent:
    _id: str
    _variables: Dict[str, OutputVariable] = field(init=False, default_factory=dict)
    _extra_outputs: Dict[str, ExtraOutput] = field(init=False, default_factory=dict)
    model: Optional[Any] = field(default=None, init=False)
    ignore: bool = field(default=False, init=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputComponent):
            return NotImplemented
        return self.is_close(other, rel_tol=0.0, abs_tol=0.0)

    def is_close(
        self,
        other: "OutputComponent",
        *,
        rel_tol: float = 1.0e-9,
        abs_tol: float = 0.0,
    ) -> bool:
        return (self.ignore or other.ignore) or (
            self._id == other._id
            and _are_mappings_close(self._variables, other._variables, rel_tol, abs_tol)
            and _are_mappings_close(
                self._extra_outputs, other._extra_outputs, rel_tol, abs_tol
            )
        )

    def __str__(self) -> str:
        string = f"{self._id} : {'(ignored)' if self.ignore else ''}\n"
        for var in self._variables.values():
            string += f"  {str(var)}\n"
        if self._extra_outputs:
            string += "  [Extra Outputs]\n"
            for out in self._extra_outputs.values():
                string += f"    {out._name}: {out._value}\n"
        return string

    def var(self, variable_name: str) -> OutputVariable:
        if variable_name not in self._variables:
            self._variables[variable_name] = OutputVariable(variable_name)
        return self._variables[variable_name]

    def extra_output(self, output_name: str) -> ExtraOutput:
        if output_name not in self._extra_outputs:
            self._extra_outputs[output_name] = ExtraOutput(output_name)
        return self._extra_outputs[output_name]


@dataclass
class OutputValues:
    """
    Contains variables and extra outputs after solver completion.

    If constructed with a LinopyOptimizationProblem, variable solution values
    are extracted from linopy_model.solution.  If constructed without arguments,
    an empty container is created (useful for building expected values in tests).
    """

    problem: Optional[LinopyOptimizationProblem] = field(default=None)
    _components: Dict[str, OutputComponent] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._build_components()
        self._fill_components()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputValues):
            return NotImplemented
        return _are_mappings_close(self._components, other._components, 0.0, 0.0)

    def is_close(
        self, other: "OutputValues", *, rel_tol: float = 1.0e-9, abs_tol: float = 0.0
    ) -> bool:
        return _are_mappings_close(
            self._components, other._components, rel_tol, abs_tol
        )

    def __str__(self) -> str:
        return "\n" + "".join(f"{comp}\n" for comp in self._components.values())

    def _build_components(self) -> None:
        if self.problem is None:
            return
        for cmp in self.problem.network.all_components:
            comp = self.component(cmp.id)
            comp.model = cmp.model

    def _fill_components(self) -> None:
        self._evaluate_variables()
        self._evaluate_extra_outputs()

    def component(self, component_id: str) -> OutputComponent:
        if component_id not in self._components:
            self._components[component_id] = OutputComponent(component_id)
        return self._components[component_id]

    def _evaluate_variables(self) -> None:
        """Extract variable solution values from linopy_model.solution."""
        if self.problem is None:
            return

        solution = self.problem.linopy_model.solution
        if solution is None:
            return

        # Iterate over all linopy variables registered in the problem
        for (_model_key, var_name), lv in self.problem._linopy_vars.items():
            lv_name = lv.name
            if lv_name not in solution:
                continue

            sol_da: xr.DataArray = solution[lv_name]

            # Use the variable's own component coords to avoid iterating over
            # NaN-padded entries from the outer-join solution Dataset.
            own_components = lv.coords["component"].values
            for comp_id in own_components:
                comp_da = sol_da.sel(component=comp_id)

                if "time" in comp_da.dims and "scenario" in comp_da.dims:
                    for t_idx in range(comp_da.sizes["time"]):
                        for s_idx in range(comp_da.sizes["scenario"]):
                            val = float(comp_da.isel(time=t_idx, scenario=s_idx).values)
                            self.component(str(comp_id)).var(var_name)._set(
                                t_idx, s_idx, val
                            )
                elif "time" in comp_da.dims:
                    for t_idx in range(comp_da.sizes["time"]):
                        val = float(comp_da.isel(time=t_idx).values)
                        self.component(str(comp_id)).var(var_name)._set(
                            t_idx, None, val
                        )
                elif "scenario" in comp_da.dims:
                    for s_idx in range(comp_da.sizes["scenario"]):
                        val = float(comp_da.isel(scenario=s_idx).values)
                        self.component(str(comp_id)).var(var_name)._set(
                            None, s_idx, val
                        )
                else:
                    val = float(comp_da.values)
                    self.component(str(comp_id)).var(var_name)._set(None, None, val)

    def _evaluate_extra_outputs(self) -> None:
        """Evaluate all model extra outputs vectorized over [component, time, scenario]."""
        if self.problem is None:
            return

        # Build var_solution_arrays from the linopy solution
        var_solution_arrays: Dict[Tuple[int, str], xr.DataArray] = {}
        solution = self.problem.linopy_model.solution
        if solution is not None:
            for (mk, vname), lv in self.problem._linopy_vars.items():
                if lv.name in solution:
                    var_solution_arrays[(mk, vname)] = solution[lv.name]

        # Process each model that has extra outputs
        for mk, model in self.problem.models.items():
            if not model.extra_outputs:
                continue
            components = self.problem.model_components[mk]

            # Build post-solve xarray port arrays for this model
            port_arrays = _build_port_arrays_xarray(
                model=model,
                components=components,
                model_key=mk,
                all_models=self.problem.models,
                all_model_components=self.problem.model_components,
                var_solution_arrays=var_solution_arrays,
                param_arrays=self.problem.param_arrays,
                network=self.problem.network,
                block_length=self.problem.block_length,
                scenarios_count=self.problem.scenarios,
            )

            for out_id, expr_node in model.extra_outputs.items():
                builder = VectorizedExtraOutputBuilder(
                    model_key=mk,
                    model_name=model.id,
                    param_arrays=self.problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
                    port_arrays=port_arrays,
                    block_length=self.problem.block_length,
                    scenarios_count=self.problem.scenarios,
                )
                result_da: xr.DataArray = visit(expr_node, builder)

                for comp in components:
                    comp_da = (
                        result_da.sel(component=comp.id)
                        if "component" in result_da.dims
                        else result_da
                    )
                    eo = self.component(comp.id).extra_output(out_id)
                    _fill_extra_output_from_da(eo, comp_da)


def _fill_extra_output_from_da(eo: ExtraOutput, da: xr.DataArray) -> None:
    """Unpack a (time?, scenario?) DataArray into ExtraOutput scalar storage."""
    if "time" in da.dims and "scenario" in da.dims:
        for t in range(da.sizes["time"]):
            for s in range(da.sizes["scenario"]):
                eo._set(t, s, float(da.isel(time=t, scenario=s).values))
    elif "time" in da.dims:
        for t in range(da.sizes["time"]):
            eo._set(t, None, float(da.isel(time=t).values))
    elif "scenario" in da.dims:
        for s in range(da.sizes["scenario"]):
            eo._set(None, s, float(da.isel(scenario=s).values))
    else:
        eo._set(None, None, float(da.values))


Comparable = TypeVar("Comparable", OutputComponent, OutputVariable, ExtraOutput)


def _are_mappings_close(
    lhs: Mapping[str, Comparable],
    rhs: Mapping[str, Comparable],
    rel_tol: float,
    abs_tol: float,
) -> bool:
    lhs_keys = lhs.keys()
    rhs_keys = rhs.keys()

    for key in lhs_keys - rhs_keys:
        if not lhs[key].ignore:
            return False

    for key in rhs_keys - lhs_keys:
        if not rhs[key].ignore:
            return False

    for key in lhs_keys & rhs_keys:
        left_item = lhs[key]
        right_item = rhs[key]
        if not left_item.ignore and not left_item.is_close(
            right_item, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return False

    return True
