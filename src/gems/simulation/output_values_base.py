# output_values_base.py

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import xarray as xr


@dataclass
class OutputVariable:
    """
    Stores a solver variable's solution as a vectorized xr.DataArray
    with dims ⊆ {component, time, scenario}.
    """

    _name: str
    _data: Optional[xr.DataArray] = field(init=False, default=None)
    _basis_status: Optional[xr.DataArray] = field(init=False, default=None)
    ignore: bool = field(default=False, init=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputVariable):
            return NotImplemented
        return self.is_close(other, rel_tol=0.0, abs_tol=0.0)

    def is_close(
        self,
        other: "OutputVariable",
        *,
        rel_tol: float = 1.0e-9,
        abs_tol: float = 0.0,
    ) -> bool:
        if not isinstance(other, OutputVariable):
            return NotImplemented  # type: ignore[return-value]
        if self.ignore or other.ignore:
            return True
        if (self._data is None) != (other._data is None):
            return False
        if self._data is None:
            return True
        assert other._data is not None  # narrowing: both non-None by checks above
        try:
            lhs, rhs = xr.align(self._data, other._data, join="exact")
        except ValueError:
            return False
        return bool(np.allclose(lhs.values, rhs.values, rtol=rel_tol, atol=abs_tol))

    def __str__(self) -> str:
        return f"{self._name} : {self._data!r} {'(ignored)' if self.ignore else ''}"


@dataclass
class ExtraOutput:
    """
    Stores a post-solve extra output expression as a vectorized xr.DataArray
    with dims ⊆ {component, time, scenario}.
    """

    _name: str
    _data: Optional[xr.DataArray] = field(init=False, default=None)
    ignore: bool = field(default=False, init=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtraOutput):
            return NotImplemented
        return self.is_close(other, rel_tol=0.0, abs_tol=0.0)

    def is_close(
        self,
        other: "ExtraOutput",
        *,
        rel_tol: float = 1.0e-9,
        abs_tol: float = 0.0,
    ) -> bool:
        if not isinstance(other, ExtraOutput):
            return NotImplemented  # type: ignore[return-value]
        if self.ignore or other.ignore:
            return True
        if (self._data is None) != (other._data is None):
            return False
        if self._data is None:
            return True
        assert other._data is not None  # narrowing: both non-None by checks above
        try:
            lhs, rhs = xr.align(self._data, other._data, join="exact")
        except ValueError:
            return False
        return bool(np.allclose(lhs.values, rhs.values, rtol=rel_tol, atol=abs_tol))

    def __str__(self) -> str:
        return f"{self._name} : {self._data!r} {'(ignored)' if self.ignore else ''}"
