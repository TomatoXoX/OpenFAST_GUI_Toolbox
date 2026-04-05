"""
core/geometry_service.py
========================
Thin adapter between the raw ``geometry.py`` engine and the rest of the
application.

Responsibilities
----------------
* Call ``geometry.calculate_semisub_properties`` with raw geometry CSV data.
* Re-shape the result into the canonical dict format that :mod:`core.file_modifier`
  expects.
* Validate inputs and raise meaningful exceptions.

No UI imports.  Progress / errors are communicated via plain exceptions and
an optional ``logging.Logger``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required columns in a geometry CSV row
# ---------------------------------------------------------------------------

REQUIRED_GEOMETRY_COLUMNS = frozenset({
    "ID",
    "MC_radius", "MC_height_above_SWL", "MC_height_below_SWL", "MC_thickness",
    "distance",
    "UC_radius", "UC_height_above_SWL", "UC_height_below_SWL", "UC_thickness",
    "BC_radius", "BC_height", "BC_thickness",
})


def validate_geometry_csv_columns(columns: set) -> None:
    """
    Raise :class:`ValueError` if *columns* is missing any required geometry column.
    """
    missing = REQUIRED_GEOMETRY_COLUMNS - columns
    if missing:
        raise ValueError(
            f"Geometry CSV is missing required columns: {', '.join(sorted(missing))}"
        )


def calculate_platform_properties(
    geo_case_data: Dict[str, Any],
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Calculate platform structural properties from a single geometry row.

    Parameters
    ----------
    geo_case_data:
        A dict with all required geometry columns (e.g. one row from a CSV).
        Extra columns are silently ignored via ``**kwargs``.
    log:
        Optional progress callback ``(message: str) -> None``.

    Returns
    -------
    dict with keys:
        ``total_properties_no_ballast``, ``mooring_points``,
        ``total_inertia_about_cm``, ``column_properties``

    Raises
    ------
    ImportError
        If ``geometry`` module cannot be imported.
    Exception
        Any error from the underlying geometry engine is re-raised.
    """
    try:
        import geometry as calc_geo  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import 'geometry' module.  "
            "Ensure geometry.py is on the Python path."
        ) from exc

    # Extract the scalar geometry parameters (ignore 'ID' and any extra cols)
    geom_params = {
        k: v for k, v in geo_case_data.items()
        if k in REQUIRED_GEOMETRY_COLUMNS and k != "ID"
    }

    geom_id = geo_case_data.get("ID", "?")
    _info(log, f"  Calculating geometry for ID={geom_id}, MC_r={geom_params.get('MC_radius')}")

    raw = calc_geo.calculate_semisub_properties(**geom_params, print_results=False)

    # --- re-shape to canonical output ---
    structural = raw["structural_properties"]
    mooring    = raw["mooring_points"]
    inertia    = raw["total_inertia_about_cm"]

    return {
        "total_properties_no_ballast": {
            "weight": structural["weight"],
            "cg": (0.0, 0.0, structural["cg"][2]),
        },
        "mooring_points": [
            {"x": p["x"], "y": p["y"], "z": p["z"]} for p in mooring
        ],
        "total_inertia_about_cm": {
            "roll":  inertia["roll"],
            "pitch": inertia["pitch"],
            "yaw":   inertia["yaw"],
        },
        "column_properties": {
            "main":  {"radius": geom_params["MC_radius"], "thickness": geom_params["MC_thickness"]},
            "upper": {"radius": geom_params["UC_radius"], "thickness": geom_params["UC_thickness"]},
            "base":  {"radius": geom_params["BC_radius"], "thickness": geom_params["BC_thickness"]},
        },
    }


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------

def _info(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        log(msg)
    logger.info(msg)
