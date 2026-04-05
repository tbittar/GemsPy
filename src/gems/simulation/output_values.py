# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.

"""
Utility classes to obtain solver results from a linopy-based optimization problem.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Set, TypeVar

import xarray as xr

from gems.expression import evaluate
from gems.expression.evaluate import EvaluationError
from gems.simulation.extra_output import ExtraOutput, ExtraOutputValueProvider
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

    def evaluate_extra_outputs(self, problem: LinopyOptimizationProblem) -> None:
        """Evaluate all model-defined extra outputs for this component."""
        if self.model is None or self.model.extra_outputs is None:
            return

        self._extra_outputs = {}

        for out_id, expr_node in self.model.extra_outputs.items():
            if out_id not in self._extra_outputs:
                self._extra_outputs[out_id] = ExtraOutput(out_id)
            self._evaluate_single_extra_output(
                self._extra_outputs[out_id], problem, expr_node
            )

    def _evaluate_single_extra_output(
        self,
        extra_output: ExtraOutput,
        problem: LinopyOptimizationProblem,
        expr_node: Any,
    ) -> None:
        all_indices: Set[TimeScenarioIndex] = set()
        for var in self._variables.values():
            all_indices.update(var._value.keys())
        if not all_indices:
            all_indices = {TimeScenarioIndex(0, 0)}

        sorted_indices = sorted(all_indices, key=lambda k: (k.time, k.scenario))

        for idx in sorted_indices:
            try:
                expanded_expr = problem.expand_operators_for_extra_output(
                    expr_node, self._id
                )
                provider = ExtraOutputValueProvider(self, problem, idx)
                val = float(evaluate(expanded_expr, provider))
            except EvaluationError as e:
                print(
                    f"[ERROR] Eval failed for '{extra_output._name}' in {self._id} "
                    f"at t={idx.time}, s={idx.scenario}: {e}"
                )
                val = float("nan")
            except Exception as e:
                print(
                    f"[ERROR] Unexpected error for '{extra_output._name}' in {self._id}: {e}"
                )
                val = float("nan")

            extra_output._set(idx.time, idx.scenario, val)


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
        if self.problem is None:
            return
        for comp in self._components.values():
            comp.evaluate_extra_outputs(self.problem)


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


@dataclass(frozen=True)
class BendersSolution:
    data: Dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BendersSolution):
            return NotImplemented
        return (
            self.overall_cost == other.overall_cost
            and self.candidates == other.candidates
        )

    def is_close(
        self,
        other: "BendersSolution",
        *,
        rel_tol: float = 1.0e-9,
        abs_tol: float = 0.0,
    ) -> bool:
        return (
            math.isclose(
                self.overall_cost, other.overall_cost, abs_tol=abs_tol, rel_tol=rel_tol
            )
            and self.candidates.keys() == other.candidates.keys()
            and all(
                math.isclose(
                    self.candidates[key],
                    other.candidates[key],
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
                for key in self.candidates
            )
        )

    def __str__(self) -> str:
        lpad = 30
        rpad = 12

        string = "Benders' solution:\n"
        string += f"{'Overall cost':<{lpad}} : {self.overall_cost:>{rpad}}\n"
        string += f"{'Investment cost':<{lpad}} : {self.investment_cost:>{rpad}}\n"
        string += f"{'Operational cost':<{lpad}} : {self.operational_cost:>{rpad}}\n"
        string += "-" * (lpad + rpad + 3) + "\n"
        for candidate, investment in self.candidates.items():
            string += f"{candidate:<{lpad}} : {investment:>{rpad}}\n"

        return string

    @property
    def investment_cost(self) -> float:
        return self.data["solution"]["investment_cost"]

    @property
    def operational_cost(self) -> float:
        return self.data["solution"]["operational_cost"]

    @property
    def overall_cost(self) -> float:
        return self.data["solution"]["overall_cost"]

    @property
    def candidates(self) -> Dict[str, float]:
        return self.data["solution"]["values"]

    @property
    def status(self) -> str:
        return self.data["solution"]["problem_status"]

    @property
    def absolute_gap(self) -> float:
        return self.data["solution"]["optimality_gap"]

    @property
    def relative_gap(self) -> float:
        return self.data["solution"]["relative_gap"]

    @property
    def stopping_criterion(self) -> str:
        return self.data["solution"]["stopping_criterion"]


@dataclass(frozen=True, eq=False)
class BendersMergedSolution(BendersSolution):
    @property
    def lower_bound(self) -> float:
        return self.data["solution"]["lb"]

    @property
    def upper_bound(self) -> float:
        return self.data["solution"]["ub"]


@dataclass(frozen=True, eq=False)
class BendersDecomposedSolution(BendersSolution):
    @property
    def nb_iterations(self) -> int:
        return self.data["solution"]["iteration"]

    @property
    def duration(self) -> float:
        return self.data["solution"]["run_duration"]
