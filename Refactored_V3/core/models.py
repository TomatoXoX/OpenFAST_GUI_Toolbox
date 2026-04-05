"""
core/models.py
==============
Pure data-transfer objects and domain models.

Zero external dependencies beyond the standard library.
No tkinter, no numpy, no UI state.  All fields are typed explicitly so
that the ViewModel and the View can reason about data shapes without
importing from each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DistributionType(str, Enum):
    """Supported parameter-sampling strategies."""
    GRID_SEARCH = "grid_search"
    CSV_COLUMNWISE = "csv_columnwise"
    LATIN_HYPERCUBE = "latin_hypercube"
    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def sampling_types(cls) -> frozenset["DistributionType"]:
        return frozenset({cls.LATIN_HYPERCUBE, cls.UNIFORM, cls.NORMAL})


class ParameterType(str, Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    OPTION = "option"


class CaseStatus(str, Enum):
    READY = "Ready"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    PROCESSING = "Processing"


# ---------------------------------------------------------------------------
# Parameter-related models
# ---------------------------------------------------------------------------

@dataclass
class ParameterInfo:
    """Metadata discovered for a single OpenFAST file parameter."""
    name: str
    file_key: str
    line_number: int
    original_value: Union[float, int, bool, str]
    type: ParameterType
    description: str = ""
    unit: str = ""


@dataclass
class ParameterVariation:
    """
    Specification for how a single parameter should be varied across cases.

    All range / list / option fields use plain Python types — the ViewModel
    translates from presentation-layer (e.g. tk.StringVar) before
    constructing this object.
    """
    param_info: ParameterInfo

    # Grid / range fields (float & int)
    start: float = 0.0
    end: float = 1.0
    steps: int = 5

    # int-specific
    int_mode: str = "Range"          # "Range" | "List"
    int_list: List[int] = field(default_factory=list)

    # bool-specific
    bool_choice: str = "Vary (True & False)"  # "Vary (True & False)" | "True" | "False"

    # option-specific
    options_list: List[str] = field(default_factory=list)

    # csv_columnwise
    csv_values: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# File-structure models
# ---------------------------------------------------------------------------

@dataclass
class FileInfo:
    """Represents a single OpenFAST input file discovered during scanning."""
    key: str
    path: Path
    original_strings: set = field(default_factory=set)
    parameters: Dict[str, ParameterInfo] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Case models
# ---------------------------------------------------------------------------

@dataclass
class GeometryCase:
    """
    A single row from the geometry CSV.

    The `data` dict holds raw column values exactly as read from the CSV.
    The `id` field is the value in the 'ID' column (or 'base' for dummy).
    """
    id: Union[str, int]
    data: Dict[str, Any]
    is_dummy: bool = False


@dataclass
class CaseInfo:
    """
    Summary record written to test_cases_summary.json and used at runtime.
    """
    case_name: str
    fst_file: str
    geometry_id: Union[str, int, None]
    parameters: Dict[str, Any]
    path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "fst_file": self.fst_file,
            "geometry_id": self.geometry_id,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], base_dir: Optional[Path] = None) -> "CaseInfo":
        obj = cls(
            case_name=d["case_name"],
            fst_file=d["fst_file"],
            geometry_id=d.get("geometry_id"),
            parameters=d.get("parameters", {}),
        )
        if base_dir is not None:
            obj.path = base_dir / obj.case_name
        return obj


@dataclass
class RunResult:
    """Outcome of a single simulation run or post-processing task."""
    case_name: str
    status: CaseStatus
    result_message: str = ""
    runtime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Application state snapshot (used by ViewModel for serialisation)
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """
    Persisted configuration that can be saved/loaded as JSON.
    Holds only primitive-serialisable values.
    """
    base_fst_path: str = ""
    output_dir: str = ""
    num_cases: int = 10
    distribution: str = DistributionType.GRID_SEARCH.value
    geometry_csv_path: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
