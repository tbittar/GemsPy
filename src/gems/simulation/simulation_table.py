from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import xarray as xr

from gems.expression.visitor import visit
from gems.simulation.extra_output import VectorizedExtraOutputBuilder
from gems.simulation.linopy_problem import LinopyOptimizationProblem, build_port_arrays


class SimulationColumns(str, Enum):
    BLOCK = "block"
    COMPONENT = "component"
    OUTPUT = "output"
    ABSOLUTE_TIME_INDEX = "absolute-time-index"
    BLOCK_TIME_INDEX = "block-time-index"
    SCENARIO_INDEX = "scenario-index"
    VALUE = "value"
    BASIS_STATUS = "basis-status"


class SimulationTableBuilder:
    """Builds simulation tables directly from a LinopyOptimizationProblem."""

    def __init__(self, simulation_id: Optional[str] = None) -> None:
        self.simulation_id: str = simulation_id or datetime.now().strftime(
            "%Y%m%d-%H%M"
        )

    def build(
        self,
        problem: LinopyOptimizationProblem,
        absolute_time_offset: Optional[int] = None,
    ) -> pd.DataFrame:
        block = problem.block.id
        block_size = problem.block_length

        if absolute_time_offset is None:
            absolute_time_offset = (block - 1) * block_size

        rows: list[dict[str, Any]] = []
        rows += self._collect_solver_outputs(problem, block, absolute_time_offset)
        rows += self._collect_extra_outputs(problem, block, absolute_time_offset)
        rows.append(self._collect_objective_value(problem, block))

        return pd.DataFrame(rows, columns=[c.value for c in SimulationColumns])

    # -------------------------------------------------------------------------
    # Solver outputs
    # -------------------------------------------------------------------------

    def _collect_solver_outputs(
        self,
        problem: LinopyOptimizationProblem,
        block: int,
        abs_offset: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        solution = problem.linopy_model.solution
        if solution is None:
            return rows

        for (_, var_name), lv in problem._linopy_vars.items():
            if lv.name not in solution:
                continue

            sol_da: xr.DataArray = solution[lv.name]
            own_components = list(lv.coords["component"].values)
            sol_da = sol_da.sel(component=own_components)

            rows += self._da_to_rows(
                sol_da, var_name, block, abs_offset, basis_status=None
            )

        return rows

    # -------------------------------------------------------------------------
    # Extra outputs
    # -------------------------------------------------------------------------

    def _collect_extra_outputs(
        self,
        problem: LinopyOptimizationProblem,
        block: int,
        abs_offset: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        var_solution_arrays: Dict[Tuple[str, str], xr.DataArray] = {}
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
                    model_id=mk_,
                    param_arrays=problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
                    port_arrays={},
                    block_length=problem.block_length,
                    scenarios_count=problem.scenarios,
                ),
            )

            for out_id, expr_node in model.extra_outputs.items():
                builder = VectorizedExtraOutputBuilder(
                    model_id=mk,
                    param_arrays=problem.param_arrays,
                    var_solution_arrays=var_solution_arrays,
                    port_arrays=port_arrays,
                    block_length=problem.block_length,
                    scenarios_count=problem.scenarios,
                )
                result_da: xr.DataArray = cast(xr.DataArray, visit(expr_node, builder))

                if "component" in result_da.dims:
                    own_ids = [c.id for c in components]
                    present = [
                        c for c in own_ids if c in result_da.coords["component"].values
                    ]
                    result_da = result_da.sel(component=present)

                rows += self._da_to_rows(
                    result_da, out_id, block, abs_offset, basis_status=None
                )

        return rows

    # -------------------------------------------------------------------------
    # Objective value
    # -------------------------------------------------------------------------

    def _collect_objective_value(
        self, problem: LinopyOptimizationProblem, block: int
    ) -> dict[str, Any]:
        return {
            SimulationColumns.BLOCK.value: block,
            SimulationColumns.COMPONENT.value: None,
            SimulationColumns.OUTPUT.value: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX.value: None,
            SimulationColumns.BLOCK_TIME_INDEX.value: None,
            SimulationColumns.SCENARIO_INDEX.value: None,
            SimulationColumns.VALUE.value: problem.objective_value,
            SimulationColumns.BASIS_STATUS.value: None,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _da_to_rows(
        da: xr.DataArray,
        output_name: str,
        block: int,
        abs_offset: int,
        basis_status: Optional[str],
    ) -> list[dict[str, Any]]:
        """Flatten a [component?, time?, scenario?] DataArray into table rows."""
        # Normalize to a uniform [component, time, scenario] shape so that
        # the iteration below never needs to branch on which dims are present.
        if "component" not in da.dims:
            da = da.expand_dims(component=[None])
        if "time" not in da.dims:
            da = da.expand_dims(time=[0])
        if "scenario" not in da.dims:
            da = da.expand_dims(scenario=[0])

        da = da.transpose("component", "time", "scenario")
        comp_vals: List[Any] = list(da.coords["component"].values)
        n_time = da.sizes["time"]
        n_scen = da.sizes["scenario"]
        arr = da.values  # shape [C, T, S]

        return [
            {
                SimulationColumns.BLOCK.value: block,
                SimulationColumns.COMPONENT.value: str(c) if c is not None else None,
                SimulationColumns.OUTPUT.value: output_name,
                SimulationColumns.ABSOLUTE_TIME_INDEX.value: abs_offset + t,
                SimulationColumns.BLOCK_TIME_INDEX.value: t,
                SimulationColumns.SCENARIO_INDEX.value: s,
                SimulationColumns.VALUE.value: float(arr[ci, t, s]),
                SimulationColumns.BASIS_STATUS.value: basis_status,
            }
            for ci, c in enumerate(comp_vals)
            for t in range(n_time)
            for s in range(n_scen)
        ]


@dataclass
class SimulationTableWriter:
    """Handles writing simulation tables to CSV."""

    simulation_table: pd.DataFrame

    def write_csv(
        self,
        output_dir: Union[str, Path],
        simulation_id: str,
        optim_nb: int,
    ) -> Path:
        """Write the simulation table to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"simulation_table_{simulation_id}_{optim_nb}.csv"
        filepath = output_dir / filename
        self.simulation_table.to_csv(filepath, index=False)
        return filepath
