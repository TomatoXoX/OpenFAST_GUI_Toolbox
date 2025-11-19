import logging
import os
import re
from pathlib import Path
from typing import Dict, List


def _strip_quotes(s: str) -> str:
    """Removes leading/trailing single or double quotes from the string."""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _read_lines(p: str, logger: logging.Logger) -> List[str]:
    """Reads all lines from a text file with error handling and logging."""
    logger.debug(f"Reading file: {p}")
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except Exception as exc:  # pragma: no cover - logging driven
        logger.error(f"Failed to read file {p}: {exc}")
        return []


def _find_fst_refs(fst: str, logger: logging.Logger) -> Dict[str, str]:
    """Parses an FST file to find references to other module files."""
    logger.info(f"Parsing FST for module references: {fst}")
    base = os.path.dirname(os.path.abspath(fst))
    refs: Dict[str, str] = {}
    pattern = re.compile(r'^\s*"?([^"\s]+)"?\s+([A-Za-z0-9_()]+)')

    for line in _read_lines(fst, logger):
        match = pattern.match(line.strip())
        if match:
            refs[match.group(2)] = _strip_quotes(match.group(1))

    for key, value in list(refs.items()):
        if value.lower() in {"unused", "none"}:
            continue
        if not os.path.isabs(value):
            refs[key] = os.path.normpath(os.path.join(base, value))

    return refs