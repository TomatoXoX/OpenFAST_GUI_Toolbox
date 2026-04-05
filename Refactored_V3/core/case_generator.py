"""
core/case_generator.py
======================
Pure logic for generating OpenFAST test-case directories.

Responsibilities
----------------
* Build parameter-value combinations (grid search, CSV-columnwise, sampling).
* For each (geometry, parameter-combo) pair: create a case directory, copy
  files, apply parameter modifications, apply geometry modifications.
* Write ``case_info.json`` per case and ``test_cases_summary.json`` globally.

No tkinter.  Progress is communicated through a plain callable ``log(str)``.
"""
from __future__ import annotations

import itertools
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from core.models import (
    CaseInfo,
    DistributionType,
    FileInfo,
    GeometryCase,
    ParameterInfo,
    ParameterVariation,
)
from core.file_modifier import (
    copy_and_rewrite_paths,
    modify_elastodyn_dat,
    modify_hydrodyn_dat,
    modify_moordyn_dat,
    modify_parameter_in_file,
)
from core.geometry_service import calculate_platform_properties

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_test_cases(
    base_fst_path: Path,
    output_path: Path,
    file_structure: Dict[str, FileInfo],
    discovered_parameters: Dict[str, Dict[str, ParameterInfo]],
    parameter_variations: List[ParameterVariation],
    geometry_cases: List[GeometryCase],
    distribution: DistributionType,
    num_samples: int,
    log: Optional[Callable[[str], None]] = None,
    confirm_large: Optional[Callable[[int], bool]] = None,
) -> List[CaseInfo]:
    """
    Generate all test-case directories under *output_path*.

    Parameters
    ----------
    base_fst_path:
        Root ``.fst`` file (used only for metadata).
    output_path:
        Root output directory.  Will be created; existing content removed.
    file_structure:
        Mapping from file_key → :class:`FileInfo` (from parameter discovery).
    discovered_parameters:
        Mapping from file_key → {param_name → :class:`ParameterInfo`}.
    parameter_variations:
        List of :class:`ParameterVariation` objects (may be empty).
    geometry_cases:
        List of :class:`GeometryCase` objects (may be empty).
    distribution:
        How to sample parameter values.
    num_samples:
        Number of samples for stochastic distributions.
    log:
        Optional ``(message: str) -> None`` progress callback.
    confirm_large:
        Optional ``(total_cases: int) -> bool``.  Called when total > 10 000.
        Return ``False`` to abort.

    Returns
    -------
    list of :class:`CaseInfo`
        One entry per successfully created case.
    """
    _log(log, "Starting test case generation…")

    # -- Geometry defaults --
    geo_list = geometry_cases if geometry_cases else [GeometryCase(id="base", data={}, is_dummy=True)]

    # -- Parameter combinations --
    param_combos: List[Tuple[Any, ...]] = _build_param_combos(
        parameter_variations, distribution, num_samples, log
    )
    if not param_combos:
        param_combos = [()]  # Single run with no variation

    total = len(geo_list) * len(param_combos)
    _log(log, f"Total cases: {total} ({len(geo_list)} geometries × {len(param_combos)} param sets)")

    if total > 10_000 and confirm_large is not None:
        if not confirm_large(total):
            return []

    # -- Create output directory --
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: List[CaseInfo] = []
    overall_idx = 0

    for geo_case in geo_list:
        for combo in param_combos:
            overall_idx += 1
            geom_id = geo_case.id
            case_name = f"case_{overall_idx:04d}_geom_{geom_id}"
            case_dir = output_path / case_name

            _log(log, f"Creating case {overall_idx}/{total}: {case_name}")
            case_dir.mkdir()

            # 1. Copy all base files
            for file_info in file_structure.values():
                copy_and_rewrite_paths(
                    file_info.path,
                    case_dir / file_info.path.name,
                    log,
                )

            # 2. Apply geometry modifications
            if not geo_case.is_dummy:
                success = _apply_geometry(
                    case_dir, geo_case, file_structure, discovered_parameters, log
                )
                if not success:
                    _log(log, f"  Skipping {case_name} due to geometry error.")
                    shutil.rmtree(case_dir)
                    continue

            # 3. Apply parameter variations
            case_params: Dict[str, Any] = {}
            for idx, value in enumerate(combo):
                variation = parameter_variations[idx]
                pinfo = variation.param_info
                file_key = pinfo.file_key

                # Normalise numpy scalars
                if isinstance(value, np.integer):
                    value = int(value)
                elif isinstance(value, np.floating):
                    value = float(value)

                case_params[f"{file_key}/{pinfo.name}"] = value
                modify_parameter_in_file(
                    case_dir,
                    file_key,
                    file_structure,
                    pinfo.name,
                    value,
                    pinfo,
                    log,
                )

            # 4. Save per-case metadata
            case_info = CaseInfo(
                case_name=case_name,
                fst_file=base_fst_path.name,
                geometry_id=geom_id,
                parameters=case_params,
                path=case_dir,
            )
            (case_dir / "case_info.json").write_text(
                json.dumps(case_info.to_dict(), indent=2), encoding="utf-8"
            )
            summary.append(case_info)

    # 5. Write global summary
    summary_data = {
        "generation_date": datetime.now().isoformat(),
        "base_fst_file": str(base_fst_path),
        "num_cases": len(summary),
        "test_cases": [c.to_dict() for c in summary],
    }
    (output_path / "test_cases_summary.json").write_text(
        json.dumps(summary_data, indent=4), encoding="utf-8"
    )

    _log(log, f"Successfully generated {len(summary)} test cases in '{output_path}'")
    return summary


# ---------------------------------------------------------------------------
# Parameter combination builders
# ---------------------------------------------------------------------------

def _build_param_combos(
    variations: List[ParameterVariation],
    distribution: DistributionType,
    num_samples: int,
    log: Optional[Callable[[str], None]],
) -> List[Tuple[Any, ...]]:
    if not variations:
        return []

    if distribution == DistributionType.GRID_SEARCH:
        return _grid_combos(variations)

    if distribution == DistributionType.CSV_COLUMNWISE:
        return _csv_combos(variations)

    # Sampling distributions
    return _sample_combos(variations, distribution, num_samples, log)


def _grid_combos(variations: List[ParameterVariation]) -> List[Tuple[Any, ...]]:
    """Cartesian product across all parameter ranges."""
    value_lists: List[List[Any]] = []
    for v in variations:
        ptype = v.param_info.type.value
        values: List[Any] = []
        if ptype == "float":
            values = np.linspace(v.start, v.end, v.steps).tolist() if v.steps > 1 else [v.start]
        elif ptype == "int":
            if v.int_mode == "Range":
                values = (
                    np.round(np.linspace(v.start, v.end, v.steps)).astype(int).tolist()
                    if v.steps > 1 else [int(round(v.start))]
                )
            else:
                values = v.int_list or [int(round(v.start))]
        elif ptype == "bool":
            values = [True, False] if "Vary" in v.bool_choice else [v.bool_choice == "True"]
        elif ptype == "option":
            values = v.options_list or [str(v.param_info.original_value)]
        value_lists.append(values or [v.param_info.original_value])
    return list(itertools.product(*value_lists))


def _csv_combos(variations: List[ParameterVariation]) -> List[Tuple[Any, ...]]:
    """Zip matching CSV columns together."""
    all_lists: List[List[Any]] = []
    for v in variations:
        all_lists.append(v.csv_values or [v.param_info.original_value])
    if not all_lists or not all_lists[0]:
        return []
    if not all(len(lst) == len(all_lists[0]) for lst in all_lists):
        raise ValueError("All CSV value lists must have the same length.")
    return list(zip(*all_lists))


def _sample_combos(
    variations: List[ParameterVariation],
    distribution: DistributionType,
    num_samples: int,
    log: Optional[Callable[[str], None]],
) -> List[Tuple[Any, ...]]:
    """Latin-hypercube / uniform / normal sampling."""
    numeric = [v for v in variations if v.param_info.type.value in {"float", "int"}]
    if not numeric:
        raise ValueError("Sampling distributions require numeric parameters.")

    d = len(numeric)
    try:
        from scipy.stats import qmc  # type: ignore
        sample = qmc.LatinHypercube(d=d).random(n=num_samples)
    except (ImportError, AttributeError):
        _log(log, "Warning: 'scipy' not found. Falling back to uniform random.")
        sample = np.random.rand(num_samples, d)

    columns = [
        v.start + (v.end - v.start) * sample[:, i]
        for i, v in enumerate(numeric)
    ]
    return list(zip(*columns))


# ---------------------------------------------------------------------------
# Geometry application
# ---------------------------------------------------------------------------

def _apply_geometry(
    case_dir: Path,
    geo_case: GeometryCase,
    file_structure: Dict[str, FileInfo],
    discovered_parameters: Dict[str, Dict[str, ParameterInfo]],
    log: Optional[Callable[[str], None]],
) -> bool:
    try:
        _log(log, f"  Applying geometry for ID={geo_case.id}")
        results = calculate_platform_properties(geo_case.data, log)
        modify_elastodyn_dat(case_dir, results, file_structure, discovered_parameters, log)
        modify_moordyn_dat(case_dir, results, file_structure, log)
        modify_hydrodyn_dat(case_dir, results, file_structure, log)
        return True
    except Exception as exc:
        _log(log, f"  ERROR in geometry for ID={geo_case.id}: {exc}")
        logger.exception("Geometry application failed for case %s", case_dir.name)
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        log(msg)
    logger.info(msg)
