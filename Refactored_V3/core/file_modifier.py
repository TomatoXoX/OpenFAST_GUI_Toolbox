"""
core/file_modifier.py
=====================
All logic for in-place modification of OpenFAST input files.

Responsibilities
----------------
* Copy a source OpenFAST file to a case directory, rewriting internal
  path references to use basenames only.
* Overwrite a specific numeric / bool / option parameter in a file.
* Apply geometry-derived modifications to ElastoDyn, MoorDyn, and
  HydroDyn input files.

This module is **pure I/O logic** — no UI, no tkinter, no queues.
Communication is via a plain logging.Logger and return values / exceptions.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.models import FileInfo, ParameterInfo, ParameterType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-copy helper
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {".fst", ".dat", ".twr", ".bld", ".ipt", ".txt", ".in"}


def copy_and_rewrite_paths(
    source_path: Path,
    dest_path: Path,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Copy *source_path* → *dest_path*, stripping directory prefixes from
    all quoted path strings found inside the file so that the copy can live
    flat in a new directory.

    Binary / non-text files are copied as-is.
    """
    if source_path.suffix.lower() not in _TEXT_EXTENSIONS:
        shutil.copy2(source_path, dest_path)
        return

    try:
        content = source_path.read_text(encoding="utf-8", errors="ignore")
        pattern = re.compile(r'(["\'])((?:\.\.[\\/])*[a-zA-Z0-9_.\-\s\\/]+)\1')

        def _replacer(match: re.Match) -> str:
            quote = match.group(1)
            path_str = match.group(2)
            if path_str.lower() in {"default", "unused", "none"}:
                return match.group(0)
            return f"{quote}{Path(path_str).name}{quote}"

        new_content = pattern.sub(_replacer, content)
        if new_content != content and log:
            log(f"    Rewrote internal paths in {dest_path.name}")
        dest_path.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        _warn(log, f"Error rewriting {source_path.name}: {exc}. Copying as-is.")
        shutil.copy2(source_path, dest_path)


# ---------------------------------------------------------------------------
# Parameter line formatting
# ---------------------------------------------------------------------------

def format_parameter_line(
    line: str,
    new_value: Any,
    param_type: ParameterType,
) -> str:
    """
    Return *line* with its leading value token replaced by *new_value*,
    preserving all trailing whitespace and comments.
    """
    if param_type == ParameterType.FLOAT:
        value_str = f"{float(new_value):.7G}"
    elif param_type == ParameterType.BOOL:
        value_str = str(bool(new_value)).upper()
    elif param_type == ParameterType.OPTION:
        value_str = f'"{new_value}"' if " " in str(new_value) else str(new_value)
    else:
        value_str = str(new_value)

    parts = line.split()
    if not parts:
        return line
    return re.sub(r"^\s*[^\s]+", f"{value_str:>{len(parts[0])}}", line, count=1)


# ---------------------------------------------------------------------------
# Parameter modification
# ---------------------------------------------------------------------------

def modify_parameter_in_file(
    case_dir: Path,
    file_key: str,
    file_structure: Dict[str, FileInfo],
    param_name: str,
    value: Any,
    param_info: ParameterInfo,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Overwrite the value of *param_name* inside the copy of the file
    identified by *file_key* that lives in *case_dir*.
    """
    info = file_structure.get(file_key)
    if info is None:
        _warn(log, f"Unknown file_key '{file_key}' — skipping '{param_name}'")
        return

    file_path = case_dir / info.path.name
    if not file_path.exists():
        _warn(log, f"File not found: {file_path} (param '{param_name}')")
        return

    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    line_num = param_info.line_number

    if 0 <= line_num < len(lines) and param_name in lines[line_num]:
        lines[line_num] = format_parameter_line(lines[line_num], value, param_info.type)
        file_path.write_text("".join(lines), encoding="utf-8")
    else:
        # Fall back to full-file search
        _warn(
            log,
            f"Parameter '{param_name}' not at expected line in {file_path.name}; searching…",
        )
        for idx, line in enumerate(lines):
            if (
                re.search(rf"\b{re.escape(param_name)}\b", line)
                and not line.strip().startswith(("!", "#"))
            ):
                lines[idx] = format_parameter_line(line, value, param_info.type)
                file_path.write_text("".join(lines), encoding="utf-8")
                return
        _warn(log, f"ERROR: Could not find '{param_name}' in {file_path.name}")


# ---------------------------------------------------------------------------
# Geometry-specific modifiers
# ---------------------------------------------------------------------------

def modify_elastodyn_dat(
    case_dir: Path,
    platform_results: Dict,
    file_structure: Dict[str, FileInfo],
    discovered_parameters: Dict[str, Dict[str, ParameterInfo]],
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Write platform mass/inertia values into the ElastoDyn file."""
    _info(log, "    Modifying ElastoDyn.dat…")
    total_props = platform_results["total_properties_no_ballast"]
    total_inertia = platform_results["total_inertia_about_cm"]

    params_to_set = {
        "PtfmMass":  total_props["weight"],
        "PtfmCMzt":  total_props["cg"][2],
        "PtfmRIner": total_inertia["roll"],
        "PtfmPIner": total_inertia["pitch"],
        "PtfmYIner": total_inertia["yaw"],
    }

    elastodyn_key = _find_file_key(file_structure, "elastodyn")
    if elastodyn_key is None:
        _warn(log, "    WARNING: ElastoDyn file not found in structure. Skipping.")
        return

    for param_name, value in params_to_set.items():
        file_params = discovered_parameters.get(elastodyn_key, {})
        if param_name in file_params:
            modify_parameter_in_file(
                case_dir,
                elastodyn_key,
                file_structure,
                param_name,
                value,
                file_params[param_name],
                log,
            )
            _info(log, f"      Set {param_name} = {value:.4f}")
        else:
            _warn(log, f"    WARNING: '{param_name}' not in discovered params. Skipped.")


def modify_moordyn_dat(
    case_dir: Path,
    platform_results: Dict,
    file_structure: Dict[str, FileInfo],
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Update POINTS table fairlead coordinates in the MoorDyn file."""
    _info(log, "    Modifying MoorDyn.dat…")
    moordyn_path = _resolve_file(case_dir, file_structure, "moordyn")
    if moordyn_path is None:
        _warn(log, "    WARNING: MoorDyn.dat not found. Skipping.")
        return

    fairleads = platform_results["mooring_points"]
    if len(fairleads) != 3:
        _warn(log, f"    WARNING: Expected 3 mooring points, got {len(fairleads)}. Skipping.")
        return

    lines = moordyn_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines: list = []
    in_points = False
    header_passed = False

    for line in lines:
        if "POINTS" in line.upper():
            in_points = True
            new_lines.append(line)
            continue

        if not in_points:
            new_lines.append(line)
            continue

        if line.strip().startswith(("---", "LINES")):
            in_points = False
            new_lines.append(line)
            continue

        if not header_passed and ("ID" in line or "(-)" in line):
            new_lines.append(line)
            continue
        header_passed = True

        parts = line.strip().split()
        if not parts or not parts[0].isdigit():
            new_lines.append(line)
            continue

        point_id = int(parts[0])
        if 4 <= point_id <= 6:
            fl = fairleads[point_id - 4]
            new_line = (
                f"{point_id:<5d} {'Vessel':<10s} "
                f"{fl['x']:<12.5f} {fl['y']:<12.5f} {fl['z']:<12.5f} "
                f"{parts[5]:<8s} {parts[6]:<8s} {parts[7]:<8s} {parts[8]:<5s}"
            )
            new_lines.append(new_line)
            _info(log, f"      Updated MoorDyn POINT {point_id} → ({fl['x']:.2f}, {fl['y']:.2f}, {fl['z']:.2f})")
        else:
            new_lines.append(line)

    moordyn_path.write_text("\n".join(new_lines), encoding="utf-8")


def modify_hydrodyn_dat(
    case_dir: Path,
    platform_results: Dict,
    file_structure: Dict[str, FileInfo],
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Update cylindrical member cross-section table in HydroDyn."""
    _info(log, "    Modifying HydroDyn.dat…")
    hydrodyn_path = _resolve_file(case_dir, file_structure, "hydrodyn")
    if hydrodyn_path is None:
        _warn(log, "    WARNING: HydroDyn.dat not found. Skipping.")
        return

    col_props = platform_results.get("column_properties")
    if not col_props:
        _warn(log, "    WARNING: 'column_properties' missing from results. Skipping.")
        return

    lines = hydrodyn_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    new_lines: list = []
    in_cyl = False
    header_passed = False

    for line in lines:
        if "CYLINDRICAL MEMBER CROSS-SECTION PROPERTIES" in line.upper():
            in_cyl = True
            new_lines.append(line)
            continue

        if not in_cyl:
            new_lines.append(line)
            continue

        if line.strip().startswith(("---", "RECTANGULAR")):
            in_cyl = False
            new_lines.append(line)
            continue

        if not header_passed and ("PropSetID" in line or "(-)" in line):
            new_lines.append(line)
            continue
        header_passed = True

        parts = line.strip().split()
        comment = " ".join(re.findall(r"!\s*(.*)", line)).lower()

        prop_to_update = None
        if "main column" in comment:
            prop_to_update = col_props.get("main")
        elif "upper column" in comment:
            prop_to_update = col_props.get("upper")
        elif "base column" in comment:
            prop_to_update = col_props.get("base")

        if prop_to_update and len(parts) >= 3:
            new_d = prop_to_update["radius"] * 2.0
            new_thck = prop_to_update["thickness"]
            comment_part = f"! {comment.title()}" if comment else ""
            new_line = f"{parts[0]:<12s} {new_d:<10.5f} {new_thck:<12.5f} {comment_part}"
            new_lines.append(new_line)
            _info(log, f"      Updated HydroDyn '{comment.title()}' → D={new_d:.3f}, Thck={new_thck:.4f}")
        else:
            new_lines.append(line)

    hydrodyn_path.write_text("\n".join(new_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _find_file_key(file_structure: Dict[str, FileInfo], substring: str) -> Optional[str]:
    for key, info in file_structure.items():
        if substring in info.path.name.lower():
            return key
    return None


def _resolve_file(
    case_dir: Path,
    file_structure: Dict[str, FileInfo],
    substring: str,
) -> Optional[Path]:
    key = _find_file_key(file_structure, substring)
    if key is None:
        return None
    candidate = case_dir / file_structure[key].path.name
    return candidate if candidate.exists() else None


def _warn(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        log(msg)
    logger.warning(msg)


def _info(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        log(msg)
    logger.info(msg)
