from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from attr import dataclass

# from gems.simulation.extra_output import ExtraOutput
# from optimization import OptimizationProblem  # Adjust import as needed
from gems.simulation.output_values import OutputValues


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
    """Builds simulation tables from solver output values."""

    def __init__(self, simulation_id: Optional[str] = None) -> None:
        self.simulation_id: str = simulation_id or datetime.now().strftime(
            "%Y%m%d-%H%M"
        )

    def build(
        self, output_values: OutputValues, absolute_time_offset: Optional[int] = None
    ) -> pd.DataFrame:
        if output_values.problem is None:
            raise ValueError("OutputValues problem is not set.")

        problem = output_values.problem
        block = problem.block.id
        block_size = problem.block_length

        absolute_time_offset = absolute_time_offset or (block - 1) * block_size
        assert absolute_time_offset is not None

        rows: list[dict[str, Any]] = []
        rows += self._collect_solver_outputs(output_values, block, absolute_time_offset)
        rows += self._collect_extra_outputs(output_values, block, absolute_time_offset)
        rows.append(self._collect_objective_value(output_values, block))

        return pd.DataFrame(rows, columns=[c.value for c in SimulationColumns])

    # -------------------------------------------------------------------------
    # Solver outputs
    # -------------------------------------------------------------------------
    def _collect_solver_outputs(
        self, output_values: OutputValues, block: int, abs_offset: int
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for var in output_values._variables.values():
            if var._data is None:
                continue
            da = var._data
            comp_dim = "component" in da.dims
            time_dim = "time" in da.dims
            scen_dim = "scenario" in da.dims
            comp_ids = list(da.coords["component"].values) if comp_dim else [None]

            for i_comp, comp_id in enumerate(comp_ids):
                n_time = da.sizes.get("time", 1) if time_dim else 1
                n_scen = da.sizes.get("scenario", 1) if scen_dim else 1
                for t in range(n_time):
                    for s in range(n_scen):
                        isel: Dict[str, Any] = {}
                        if comp_dim:
                            isel["component"] = i_comp
                        if time_dim:
                            isel["time"] = t
                        if scen_dim:
                            isel["scenario"] = s
                        val = float(da.isel(**isel).item()) if isel else float(da.item())  # type: ignore[arg-type]

                        basis_status = None
                        if var._basis_status is not None:
                            bs_isel: Dict[str, Any] = {
                                k: v
                                for k, v in isel.items()
                                if k in var._basis_status.dims
                            }
                            bs_val = (
                                var._basis_status.isel(**bs_isel)
                                if bs_isel
                                else var._basis_status
                            )
                            basis_status = str(bs_val.item())  # type: ignore[arg-type]

                        rows.append(
                            {
                                SimulationColumns.BLOCK.value: block,
                                SimulationColumns.COMPONENT.value: str(comp_id)
                                if comp_id is not None
                                else None,
                                SimulationColumns.OUTPUT.value: var._name,
                                SimulationColumns.ABSOLUTE_TIME_INDEX.value: abs_offset
                                + t,
                                SimulationColumns.BLOCK_TIME_INDEX.value: t,
                                SimulationColumns.SCENARIO_INDEX.value: s,
                                SimulationColumns.VALUE.value: val,
                                SimulationColumns.BASIS_STATUS.value: basis_status,
                            }
                        )
        return rows

    # -------------------------------------------------------------------------
    # Extra outputs
    # -------------------------------------------------------------------------
    def _collect_extra_outputs(
        self, output_values: OutputValues, block: int, abs_offset: int
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for eo in output_values._extra_outputs.values():
            if eo._data is None:
                continue
            da = eo._data
            comp_dim = "component" in da.dims
            time_dim = "time" in da.dims
            scen_dim = "scenario" in da.dims
            comp_ids = list(da.coords["component"].values) if comp_dim else [None]

            for i_comp, comp_id in enumerate(comp_ids):
                n_time = da.sizes.get("time", 1) if time_dim else 1
                n_scen = da.sizes.get("scenario", 1) if scen_dim else 1
                for t in range(n_time):
                    for s in range(n_scen):
                        isel: Dict[str, Any] = {}
                        if comp_dim:
                            isel["component"] = i_comp
                        if time_dim:
                            isel["time"] = t
                        if scen_dim:
                            isel["scenario"] = s
                        val = float(da.isel(**isel).item()) if isel else float(da.item())  # type: ignore[arg-type]

                        rows.append(
                            {
                                SimulationColumns.BLOCK.value: block,
                                SimulationColumns.COMPONENT.value: str(comp_id)
                                if comp_id is not None
                                else None,
                                SimulationColumns.OUTPUT.value: eo._name,
                                SimulationColumns.ABSOLUTE_TIME_INDEX.value: abs_offset
                                + t,
                                SimulationColumns.BLOCK_TIME_INDEX.value: t,
                                SimulationColumns.SCENARIO_INDEX.value: s,
                                SimulationColumns.VALUE.value: val,
                                SimulationColumns.BASIS_STATUS.value: None,
                            }
                        )
        return rows

    # -------------------------------------------------------------------------
    # Objective value
    # -------------------------------------------------------------------------
    def _collect_objective_value(
        self, output_values: OutputValues, block: int
    ) -> dict[str, Any]:
        assert output_values.problem is not None, "OutputValues problem is not set"
        return {
            SimulationColumns.BLOCK.value: block,
            SimulationColumns.COMPONENT.value: None,
            SimulationColumns.OUTPUT.value: "objective-value",
            SimulationColumns.ABSOLUTE_TIME_INDEX.value: None,
            SimulationColumns.BLOCK_TIME_INDEX.value: None,
            SimulationColumns.SCENARIO_INDEX.value: None,
            SimulationColumns.VALUE.value: output_values.problem.objective_value,
            SimulationColumns.BASIS_STATUS.value: None,
        }


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
