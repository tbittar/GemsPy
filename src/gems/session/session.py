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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import xarray as xr

from gems.optim_config.parsing import (
    OptimConfig,
    ResolutionMode,
    load_optim_config,
)
from gems.simulation.optimization import OptimizationProblem, build_problem
from gems.simulation.simulation_table import (
    SimulationTable,
    SimulationTableBuilder,
    merge_simulation_tables,
)
from gems.simulation.time_block import TimeBlock
from gems.study.folder import load_study
from gems.study.study import Study


@dataclass
class SimulationSession:
    study: Study
    optim_config: OptimConfig
    total_timesteps: int
    scenario_ids: List[int]
    run_id: str = field(default_factory=lambda: str(uuid4()))
    output_dir: Optional[Path] = None

    def run(self) -> SimulationTable:
        """Entry point. Dispatches to the appropriate resolution strategy."""
        mode = self.optim_config.resolution.mode
        if mode == ResolutionMode.FRONTAL:
            return self._run_frontal()
        elif mode == ResolutionMode.SEQUENTIAL_SUBPROBLEMS:
            return self._run_sequential()
        elif mode == ResolutionMode.PARALLEL_SUBPROBLEMS:
            return self._run_parallel()
        elif mode == ResolutionMode.BENDERS_DECOMPOSITION:
            return self._run_benders()
        raise ValueError(f"Unknown resolution mode: {mode}")

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _run_frontal(self) -> SimulationTable:
        block = TimeBlock(0, list(range(self.total_timesteps)))
        problem = self._run_block(block, scenario_ids=self.scenario_ids)
        return self._reduce([problem], scenario_ids_remap=self.scenario_ids)

    def _run_sequential(self) -> SimulationTable:
        cfg = self.optim_config.resolution
        horizon: int = cfg.horizon  # type: ignore[assignment]
        overlap: int = cfg.overlap

        all_tables: List[SimulationTable] = []
        for scenario_id in self.scenario_ids:
            t_start = 0
            block_id = 0
            carry_over: Dict[Tuple[str, str], xr.DataArray] = {}
            problems: List[OptimizationProblem] = []

            while t_start < self.total_timesteps:
                end = min(t_start + horizon, self.total_timesteps)
                timesteps = list(range(t_start, end))
                block = TimeBlock(block_id, timesteps)
                problem = self._run_block(
                    block,
                    scenario_ids=[scenario_id],
                    initial_values=carry_over or None,
                )
                problems.append(problem)
                carry_over = self._extract_carry_over(
                    problem, local_index=len(timesteps) - 1
                )
                t_start += horizon - overlap
                block_id += 1

            all_tables.append(self._reduce(problems, scenario_ids_remap=[scenario_id]))

        return merge_simulation_tables(all_tables)

    def _run_parallel(self) -> SimulationTable:
        cfg = self.optim_config.resolution
        horizon: int = cfg.horizon  # type: ignore[assignment]

        all_tables: List[SimulationTable] = []
        for scenario_id in self.scenario_ids:
            starts = range(0, self.total_timesteps, horizon)
            blocks = [
                TimeBlock(i, list(range(t, min(t + horizon, self.total_timesteps))))
                for i, t in enumerate(starts)
            ]
            problems = [self._run_block(b, scenario_ids=[scenario_id]) for b in blocks]
            all_tables.append(self._reduce(problems, scenario_ids_remap=[scenario_id]))

        return merge_simulation_tables(all_tables)

    def _run_benders(self) -> SimulationTable:
        import pandas as pd

        from gems.simulation import (
            BendersRunner,
            build_couplings,
            build_decomposed_problems,
            dump_couplings,
        )

        block = TimeBlock(1, list(range(self.total_timesteps)))
        decomposed = build_decomposed_problems(
            self.study, block, self.scenario_ids, self.optim_config
        )
        if decomposed.master is not None and self.output_dir is not None:
            dump_couplings(
                build_couplings(decomposed, self.optim_config), self.output_dir
            )
        if self.output_dir is not None:
            BendersRunner(emplacement=self.output_dir).run()
        return SimulationTable(pd.DataFrame())

    # ------------------------------------------------------------------
    # Map / reduce helpers
    # ------------------------------------------------------------------

    def _run_block(
        self,
        block: TimeBlock,
        scenario_ids: List[int],
        initial_values: Optional[Dict[Tuple[str, str], xr.DataArray]] = None,
    ) -> OptimizationProblem:
        """MAP: build and solve one time block."""
        problem = build_problem(
            self.study,
            block,
            scenario_ids,
            initial_values=initial_values,
        )
        problem.solve(solver_name="highs")
        return problem

    def _reduce(
        self,
        problems: List[OptimizationProblem],
        scenario_ids_remap: List[int],
    ) -> SimulationTable:
        """REDUCE: merge block results, remapping 0-based scenario indices to actual IDs."""
        builder = SimulationTableBuilder()
        tables = [
            builder.build(p, scenario_ids_remap=scenario_ids_remap) for p in problems
        ]
        return merge_simulation_tables(tables)

    @staticmethod
    def _extract_carry_over(
        problem: OptimizationProblem,
        local_index: int,
    ) -> Dict[Tuple[str, str], xr.DataArray]:
        """Extract variable values at *local_index* for use as initial values in the next block."""
        carry_over: Dict[Tuple[str, str], xr.DataArray] = {}
        solution = problem.linopy_model.solution
        if solution is None:
            return carry_over
        for (mk, var_name), lv in problem._linopy_vars.items():
            if "time" in lv.dims and lv.name in solution:
                sol_da: xr.DataArray = solution[lv.name]
                carry_over[(mk, var_name)] = sol_da.isel(time=local_index, drop=True)
        return carry_over


def load_session(
    study_dir: Path,
    total_timesteps: int,
    scenario_ids: List[int],
    run_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> SimulationSession:
    """Factory: load a study from disk and build a SimulationSession."""
    study = load_study(study_dir)
    config_path = study_dir / "input" / "optim-config.yml"
    optim_config = load_optim_config(config_path) or OptimConfig()
    return SimulationSession(
        study=study,
        optim_config=optim_config,
        total_timesteps=total_timesteps,
        scenario_ids=scenario_ids,
        run_id=run_id or str(uuid4()),
        output_dir=output_dir,
    )
