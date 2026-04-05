"""
services/execution_service.py
==============================
Thread-based orchestration for running OpenFAST simulations.

Responsibilities
----------------
* Accept a list of :class:`~core.models.CaseInfo` objects and an OpenFAST
  executable path.
* Spawn a configurable number of worker threads that each run one simulation
  at a time via :mod:`subprocess`.
* Post progress messages to a generic *message_bus* callback so the caller
  (ViewModel) can forward them to the UI or a queue without this service
  knowing anything about tkinter.

Design contract
---------------
The *message_bus* callable has the signature::

    message_bus(channel: str, payload: Any) -> None

Channels used by this service:
    ``"run_log"``            — plain ``str`` log line
    ``"run_tree_update"``    — ``(item_id, column, value)``
    ``"run_progress"``       — ``float`` percentage 0-100
    ``"enable_run_button"``  — ``None`` (signals completion)
"""
from __future__ import annotations

import gc
import logging
import os
import queue
import subprocess
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.models import CaseInfo, CaseStatus

logger = logging.getLogger(__name__)

# Type alias for a callable message bus
MessageBus = Callable[[str, Any], None]


class SimulationRunService:
    """
    Runs a batch of OpenFAST simulations in parallel worker threads.

    Usage::

        service = SimulationRunService(
            message_bus=my_bus,
            openfast_exe="/path/to/openfast",
            num_workers=4,
        )
        service.start(cases)

    """

    def __init__(
        self,
        message_bus: MessageBus,
        openfast_exe: str,
        num_workers: int = 1,
    ) -> None:
        self._bus = message_bus
        self._exe = openfast_exe
        self._num_workers = max(1, num_workers)

        self._job_queue: queue.Queue = queue.Queue()
        self._progress_lock = threading.Lock()
        self._completed = 0
        self._total = 0

    def start(self, cases: Dict[str, CaseInfo]) -> None:
        """
        Begin running all cases.  Returns immediately; work happens in daemon
        threads.  The caller is notified via *message_bus* when complete.

        Parameters
        ----------
        cases:
            Mapping from *item_id* (the tree-widget row key) → :class:`CaseInfo`.
            All entries will be submitted to the work queue.
        """
        if not cases:
            return

        # Reset progress counters
        self._completed = 0
        self._total = len(cases)

        # Drain any leftover items
        while not self._job_queue.empty():
            try:
                self._job_queue.get_nowait()
            except queue.Empty:
                break

        for item_id, case_info in cases.items():
            self._job_queue.put((item_id, case_info))

        self._log(
            "run_log",
            f"Starting {self._total} simulation(s) with {self._num_workers} worker(s)…",
        )

        threads = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(self._num_workers)
        ]
        manager = threading.Thread(
            target=self._manager, args=(threads,), daemon=True
        )
        manager.start()

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _manager(self, threads: List[threading.Thread]) -> None:
        for t in threads:
            t.start()
        self._job_queue.join()
        self._bus("run_log", "\n--- All simulations completed. ---")
        self._bus("enable_run_button", None)

    def _worker(self) -> None:
        while True:
            try:
                item_id, case_info = self._job_queue.get_nowait()
            except queue.Empty:
                return

            self._run_single(item_id, case_info)
            self._job_queue.task_done()
            gc.collect()

    def _run_single(self, item_id: str, case_info: CaseInfo) -> None:
        case_path = case_info.path
        case_name = case_info.case_name

        self._bus("run_tree_update", (item_id, "Status", CaseStatus.RUNNING.value))
        self._bus("run_log", f"--- Running {case_name} ---")
        start_time = datetime.now()

        log_file_path = case_path / f"{case_name}_openfast.log"
        process = None
        log_handle = None

        try:
            log_handle = log_file_path.open("w", encoding="utf-8")
            cmd = [self._exe, case_info.fst_file]
            process = subprocess.Popen(
                cmd,
                cwd=str(case_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                bufsize=1,
            )

            has_error = False
            error_keywords = [
                "error:", "error ", "aborting", "failed", "fortran runtime error"
            ]
            for line in iter(process.stdout.readline, ""):
                log_handle.write(line)
                self._bus("run_log", f"[{case_name}] {line.rstrip()}")
                if any(kw in line.lower() for kw in error_keywords):
                    has_error = True

            process.wait()
            runtime = (datetime.now() - start_time).total_seconds()

            if process.returncode != 0 or has_error:
                result_msg = (
                    f"Error (code {process.returncode})"
                    if not has_error
                    else "Error (in output)"
                )
                status = CaseStatus.FAILED.value
            else:
                result_msg = "Success"
                status = CaseStatus.COMPLETED.value

            self._bus("run_log", f"[{case_name}] Log saved to '{log_file_path.name}'")

        except Exception as exc:
            runtime = (datetime.now() - start_time).total_seconds()
            result_msg = f"Exception: {exc}"
            status = CaseStatus.FAILED.value
            self._bus(
                "run_log",
                f"FATAL ERROR launching {case_name}: {exc}\n{traceback.format_exc()}",
            )
        finally:
            if process and process.stdout:
                process.stdout.close()
            if log_handle:
                log_handle.close()

        self._bus("run_tree_update", (item_id, "Status", status))
        self._bus("run_tree_update", (item_id, "Result", result_msg))
        self._bus("run_tree_update", (item_id, "Runtime", f"{runtime:.1f}s"))

        with self._progress_lock:
            self._completed += 1
            pct = (self._completed / self._total) * 100
            self._bus("run_progress", pct)

    def _log(self, channel: str, msg: str) -> None:
        self._bus(channel, msg)
        logger.info(msg)
