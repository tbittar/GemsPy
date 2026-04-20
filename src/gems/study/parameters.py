from pathlib import Path
from typing import Any, Dict, List

from pydantic import Field
from yaml import safe_load

from gems.utils import ModifiedBaseModel


class StudyParameters(ModifiedBaseModel):
    solver: str = "highs"
    solver_logs: bool = Field(False, alias="solver-logs")
    solver_parameters: str = Field("", alias="solver-parameters")
    first_time_step: int = Field(0, alias="first-time-step")
    last_time_step: int = Field(0, alias="last-time-step")
    nb_scenarios: int = Field(1, alias="nb-scenarios")

    @property
    def scenario_ids(self) -> List[int]:
        return list(range(self.nb_scenarios))

    @property
    def total_timesteps(self) -> int:
        return self.last_time_step - self.first_time_step + 1

    def parsed_solver_options(self) -> Dict[str, Any]:
        """Parse 'KEY VALUE KEY2 VALUE2 ...' into a dict with numeric coercion."""
        if not self.solver_parameters.strip():
            return {}
        tokens = self.solver_parameters.split()
        if len(tokens) % 2 != 0:
            raise ValueError(
                f"solver-parameters must be space-separated key-value pairs, got: {self.solver_parameters!r}"
            )
        result: Dict[str, Any] = {}
        for i in range(0, len(tokens), 2):
            key, raw = tokens[i], tokens[i + 1]
            try:
                result[key] = int(raw)
            except ValueError:
                try:
                    result[key] = float(raw)
                except ValueError:
                    result[key] = raw
        return result


def load_parameters(study_dir: Path) -> StudyParameters:
    """Load parameters.yml from the study root. Returns defaults if the file is absent."""
    path = study_dir / "parameters.yml"
    if not path.exists():
        return StudyParameters()
    with path.open() as f:
        return StudyParameters.model_validate(safe_load(f))
