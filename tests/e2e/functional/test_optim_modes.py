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
E2E consistency test: frontal, parallel-subproblems, and sequential-subproblems
resolution modes must produce identical per-timestep results for a fully
time-separable problem (all thermal clusters have d_min_up=d_min_down=1).

All three modes run on the andromede_v1 DSR study (base028) over 504 timesteps
(end-timestep=503). The sequential config uses block-length=168 and block-overlap=1,
which produces 4 blocks (3 full + 1 partial) over the shared time scope.
Per-timestep values are compared after deduplicating overlap rows.

Integer decision variables (nb_units_on, nb_starting, nb_stopping) are excluded
from the per-timestep comparison: with zero startup and fixed costs those quantities
have multiple equivalent optimal values in the MIP, so the three modes can produce
different integer solutions with the same objective. The dispatch (generation) and
energy balance outputs are uniquely determined by the merit order and are compared.
"""

import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from gems.study.folder import run_study

_STUDY_SRC = Path(__file__).parent / "studies" / "dsr_3_blocks"

_FRONTAL_CONFIG = textwrap.dedent("""\
    time-scope:
      start-timestep: 0
      end-timestep: 503
    solver-options:
      solver: highs
      solver-logs: false
      solver-parameters: ""
    scenario-scope:
      nb-scenarios: 1
    resolution:
      mode: frontal
""")

_PARALLEL_CONFIG = textwrap.dedent("""\
    time-scope:
      start-timestep: 0
      end-timestep: 503
    solver-options:
      solver: highs
      solver-logs: false
      solver-parameters: ""
    scenario-scope:
      nb-scenarios: 1
    resolution:
      mode: parallel-subproblems
      block-length: 168
""")

_SEQUENTIAL_CONFIG = textwrap.dedent("""\
    time-scope:
      start-timestep: 0
      end-timestep: 503
    solver-options:
      solver: highs
      solver-logs: false
      solver-parameters: ""
    scenario-scope:
      nb-scenarios: 1
    resolution:
      mode: sequential-subproblems
      block-length: 168
      block-overlap: 1
""")

_KEY_COLS = [
    "component",
    "output",
    "absolute-time-index",
    "scenario-index",
]

# Integer decision variables that are not uniquely determined when startup and
# fixed costs are zero: multiple nb_units_on patterns achieve the same objective.
_DEGENERATE_OUTPUTS = {"nb_units_on", "nb_starting", "nb_stopping"}


def _load_csv(study_dir: Path) -> pd.DataFrame:
    output_files = list((study_dir / "output").glob("simulation_table_*.csv"))
    assert len(output_files) == 1, f"Expected 1 output file, got {len(output_files)}"
    return pd.read_csv(output_files[0])


def _run_with_config(study_dir: Path, config_yaml: str) -> pd.DataFrame:
    """Write optim-config.yml, run the study, return the raw simulation table."""
    config_path = study_dir / "input" / "optim-config.yml"
    config_path.write_text(config_yaml)
    run_study(study_dir)
    return _load_csv(study_dir)


def _per_timestep_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract uniquely-determined per-timestep rows from a simulation table.

    Drops objective-value (aggregate) rows, degenerate integer-variable rows,
    and duplicate rows caused by overlapping blocks in sequential mode.
    """
    df = raw[
        (raw["output"] != "objective-value")
        & (~raw["output"].isin(_DEGENERATE_OUTPUTS))
    ].copy()
    df = df.drop_duplicates(subset=_KEY_COLS)
    df = df.sort_values(_KEY_COLS).reset_index(drop=True)
    return df


def _total_objective(raw: pd.DataFrame) -> float:
    """Sum all per-block objective-value entries."""
    return float(raw.loc[raw["output"] == "objective-value", "value"].sum())


@pytest.mark.parametrize(
    "other_config,label,check_objective",
    [
        (_PARALLEL_CONFIG, "parallel", True),
        (_SEQUENTIAL_CONFIG, "sequential", False),
    ],
)
def test_optim_modes_produce_identical_results(
    tmp_path: Path,
    other_config: str,
    label: str,
    check_objective: bool,
) -> None:
    """Frontal mode and *label* mode yield the same per-timestep dispatch values.

    For parallel mode (no overlap) the summed block objectives are also asserted
    equal to the frontal objective. For sequential mode the summed block objectives
    are not directly comparable to frontal because overlapping timesteps contribute
    to two blocks' objective values each.
    """
    frontal_dir = tmp_path / "frontal"
    other_dir = tmp_path / label
    shutil.copytree(_STUDY_SRC, frontal_dir)
    shutil.copytree(_STUDY_SRC, other_dir)

    frontal_raw = _run_with_config(frontal_dir, _FRONTAL_CONFIG)
    other_raw = _run_with_config(other_dir, other_config)

    if check_objective:
        assert _total_objective(frontal_raw) == pytest.approx(
            _total_objective(other_raw), rel=1e-6
        ), (
            f"Total objective differs: frontal={_total_objective(frontal_raw)}, "
            f"{label}={_total_objective(other_raw)}"
        )

    frontal_df = _per_timestep_df(frontal_raw)
    other_df = _per_timestep_df(other_raw)

    assert set(frontal_df[_KEY_COLS].itertuples(index=False)) == set(
        other_df[_KEY_COLS].itertuples(index=False)
    ), (
        f"Frontal and {label} modes cover different "
        "(component, output, timestep, scenario) sets"
    )

    merged = frontal_df[_KEY_COLS + ["value"]].merge(
        other_df[_KEY_COLS + ["value"]],
        on=_KEY_COLS,
        suffixes=("_frontal", f"_{label}"),
    )

    mismatches = merged[
        (merged["value_frontal"] - merged[f"value_{label}"]).abs()
        > 1e-6 * merged["value_frontal"].abs().clip(lower=1e-12)
    ]

    assert mismatches.empty, (
        f"Per-timestep value mismatches between frontal and {label} "
        f"({len(mismatches)} rows):\n{mismatches[_KEY_COLS + ['value_frontal', f'value_{label}']].to_string()}"
    )
