"""
core/parameter_engine.py
========================
Pure-logic parameter discovery and parsing engine.

Responsibilities
----------------
* Recursively scan an OpenFAST FST file and all referenced sub-files.
* Extract numeric / bool / option parameters from each file.
* Return strongly-typed :class:`ParameterInfo` objects.

This module has **no** tkinter or UI dependency.  It communicates
progress / warnings only through a plain :class:`logging.Logger`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from core.models import FileInfo, ParameterInfo, ParameterType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_all_files(
    fst_path: Path,
    progress_callback=None,
) -> Dict[str, FileInfo]:
    """
    Entry point: recursively discover every OpenFAST input file reachable
    from *fst_path*, parse parameters in each file, and return a dict
    keyed by a unique file-key string.

    Parameters
    ----------
    fst_path:
        Absolute path to the root ``.fst`` file.
    progress_callback:
        Optional callable ``(message: str) -> None`` for progress reporting.

    Returns
    -------
    dict
        ``{file_key: FileInfo}`` for every discovered file.
    """
    raw_map: Dict[Path, _RawFileInfo] = {}
    processed: Set[Path] = set()
    _scan_recursive(fst_path.resolve(), raw_map, processed, progress_callback)

    result: Dict[str, FileInfo] = {}
    used_keys: Set[str] = set()

    for path, raw in raw_map.items():
        key = _unique_key(path, used_keys)
        used_keys.add(key)
        file_info = FileInfo(
            key=key,
            path=path,
            original_strings=raw.original_strings,
            parameters=raw.params,
        )
        result[key] = file_info

    return result


def extract_parameters(lines: List[str]) -> Dict[str, ParameterInfo]:
    """
    Parse a list of text lines from an OpenFAST input file and return a
    dict of ``{param_name: ParameterInfo}`` for every recognised parameter.

    Parameters are identified by the OpenFAST convention::

        <value>   <ParameterName>   - description (unit)

    This function is stateless and fully testable in isolation.
    """
    parameters: Dict[str, ParameterInfo] = {}
    pattern = re.compile(
        r"^\s*([^\s!#]+)\s+([a-zA-Z_][a-zA-Z0-9_()]*)", re.IGNORECASE
    )

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(("!", "#"))
            or all(ch in "-=_ " for ch in stripped)
        ):
            continue

        match = pattern.match(stripped)
        if not match:
            continue

        value_str, param_name = match.groups()

        # Skip keywords that look like parameter names but aren't
        if param_name.lower() in {"true", "false", "default", "unused", "none", "end"}:
            continue
        # Skip lines whose "value" is actually a file path
        if any(ext in value_str.lower() for ext in [".dat", ".txt", ".csv", ".twr", ".bld", ".fst"]):
            continue

        try:
            param_info = _parse_value(value_str, line)
            if param_info is None:
                continue

            description = ""
            comment_match = re.search(r"[-!]\s*(.+)$", line)
            if comment_match:
                description = comment_match.group(1).strip()

            parameters[param_name] = ParameterInfo(
                name=param_name,
                file_key="",          # filled in by caller
                line_number=idx,
                original_value=param_info["value"],
                type=ParameterType(param_info["type"]),
                description=description,
                unit=_extract_unit(line),
            )
        except Exception:
            continue

    return parameters


def extract_unit(line: str) -> str:
    """Extract the first short parenthesised unit string from *line*."""
    return _extract_unit(line)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _RawFileInfo:
    __slots__ = ("key", "original_strings", "params")

    def __init__(self, key: str) -> None:
        self.key = key
        self.original_strings: set = set()
        self.params: Dict[str, ParameterInfo] = {}


def _scan_recursive(
    file_path: Path,
    file_info_map: Dict[Path, _RawFileInfo],
    processed: Set[Path],
    progress_callback,
) -> None:
    """Recursively scan *file_path* and all referenced children."""
    if not file_path or not file_path.exists() or file_path in processed:
        return

    if progress_callback:
        progress_callback(f"  Scanning: {file_path.name}")
    processed.add(file_path)

    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("Could not read %s: %s", file_path, exc)
        return

    if file_path not in file_info_map:
        file_info_map[file_path] = _RawFileInfo(file_path.name)

    raw = file_info_map[file_path]
    raw.params = extract_parameters(content.splitlines())

    # Wire file_key into each ParameterInfo now (will be refined later)
    for pinfo in raw.params.values():
        pinfo.file_key = file_path.name

    # Find child file references (quoted strings inside the file)
    ref_pattern = re.compile(
        r'(["\'])((?:[a-zA-Z]:)?[a-zA-Z0-9_.\-\s\\/]+)\1'
    )
    for match in ref_pattern.finditer(content):
        path_inside = match.group(2)
        if not path_inside or path_inside.lower() in {"default", "unused", "none"}:
            continue

        resolved = (file_path.parent / path_inside).resolve()

        if resolved.is_file():
            _scan_recursive(resolved, file_info_map, processed, progress_callback)
        else:
            parent_dir = resolved.parent
            root_name = resolved.name
            if parent_dir.is_dir():
                for item in parent_dir.glob(f"{root_name}.*"):
                    if item.is_file():
                        if progress_callback:
                            progress_callback(
                                f"  [Discovery] Found family member: {item.name}"
                            )
                        _scan_recursive(
                            item, file_info_map, processed, progress_callback
                        )


def _unique_key(path: Path, used_keys: Set[str]) -> str:
    key = path.name
    if key not in used_keys:
        return key
    counter = 2
    stem = path.stem
    suffix = path.suffix
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used_keys:
            return candidate
        counter += 1


def _parse_value(value_str: str, description: str) -> Optional[Dict]:
    """
    Try to convert *value_str* to a typed Python value.

    Returns ``None`` if the token cannot be recognised as a parameter value.
    """
    value_str = value_str.strip().strip('"\'')
    if value_str.upper() == "DEFAULT":
        return None

    try:
        numeric = float(value_str)
        keywords = ["switch", "flag", "mode", "method", "order", "num", "index"]
        if any(kw in description.lower() for kw in keywords):
            if numeric == int(numeric) and "." not in value_str and "e" not in value_str.lower():
                return {"value": int(numeric), "type": ParameterType.INT.value}
        return {"value": numeric, "type": ParameterType.FLOAT.value}
    except ValueError:
        pass

    if value_str.lower() in {"true", "false"}:
        return {"value": value_str.lower() == "true", "type": ParameterType.BOOL.value}

    if any(kw in description.lower() for kw in ["option", "name", "file", "type"]):
        return {"value": value_str, "type": ParameterType.OPTION.value}

    return None


def _extract_unit(line: str) -> str:
    matches = re.findall(r"\(([^)]+)\)", line)
    for match in matches:
        if len(match) < 10 and not any(
            word in match.lower() for word in ["flag", "switch", "see"]
        ):
            return match
    return ""
