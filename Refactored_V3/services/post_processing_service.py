"""
services/post_processing_service.py
=====================================
Thread-based orchestration for post-processing OpenFAST outputs.

Responsibilities
----------------
* Accept a list of cases and a set of task flags.
* Spawn worker threads that each run the full post-processing pipeline
  (convert → d'Alembert → plot → frequency analysis) for a single case.
* Post progress and results back through a *message_bus* callback.

Message-bus channels used:
    ``"post_proc_log"``            — plain ``str``
    ``"post_proc_tree_update"``    — ``(item_id, column, value)``
    ``"post_proc_progress"``       — ``float`` 0-100
    ``"enable_post_proc_button"``  — ``None``
"""
from __future__ import annotations

import gc
import logging
import queue
import re
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.models import CaseInfo, CaseStatus

logger = logging.getLogger(__name__)

MessageBus = Callable[[str, Any], None]

_DEFAULT_ANALYSIS_START_TIME = 300.0


# ---------------------------------------------------------------------------
# Configuration dataclass (no UI types)
# ---------------------------------------------------------------------------

@dataclass
class PostProcConfig:
    """
    Settings that control which post-processing tasks are executed.

    All fields are plain Python types — no tk.BooleanVar etc.
    The ViewModel translates from presentation-layer before constructing this.
    """
    run_convert_csv: bool = True
    run_dalembert: bool = True
    run_plotting: bool = True
    run_frequency_analysis: bool = False
    frequency_analysis_column: str = "PtfmHeave"
    num_workers: int = 1


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PostProcessingService:
    """
    Runs the full post-processing pipeline for a batch of cases.

    Usage::

        cfg = PostProcConfig(run_plotting=True, num_workers=4)
        svc = PostProcessingService(message_bus=my_bus, config=cfg)
        svc.start(cases)
    """

    def __init__(
        self,
        message_bus: MessageBus,
        config: PostProcConfig,
    ) -> None:
        self._bus = message_bus
        self._cfg = config
        self._plotting_lock = threading.Lock()

        self._job_queue: queue.Queue = queue.Queue()
        self._progress_lock = threading.Lock()
        self._completed = 0
        self._total = 0

    def start(self, cases: Dict[str, CaseInfo]) -> None:
        """Submit all *cases* and begin processing in daemon threads."""
        if not cases:
            return

        self._completed = 0
        self._total = len(cases)

        while not self._job_queue.empty():
            try:
                self._job_queue.get_nowait()
            except queue.Empty:
                break

        for item_id, case_info in cases.items():
            self._job_queue.put((item_id, case_info))

        self._bus(
            "post_proc_log",
            f"Starting post-processing of {self._total} case(s) "
            f"with {self._cfg.num_workers} worker(s)…",
        )

        threads = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(max(1, self._cfg.num_workers))
        ]
        manager = threading.Thread(
            target=self._manager, args=(threads,), daemon=True
        )
        manager.start()

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    def _manager(self, threads: List[threading.Thread]) -> None:
        for t in threads:
            t.start()
        self._job_queue.join()
        self._bus("post_proc_log", "\n--- All post-processing tasks completed. ---")
        self._bus("enable_post_proc_button", None)

    def _worker(self) -> None:
        while True:
            try:
                item_id, case_info = self._job_queue.get_nowait()
            except queue.Empty:
                return

            self._bus(
                "post_proc_tree_update",
                (item_id, "Status", CaseStatus.PROCESSING.value),
            )
            self._bus("post_proc_log", f"--- Processing {case_info.case_name} ---")

            success = self._run_pipeline(case_info)
            status = CaseStatus.COMPLETED.value if success else CaseStatus.FAILED.value
            result = "Success" if success else "Task(s) failed"

            self._bus("post_proc_tree_update", (item_id, "Status", status))
            self._bus("post_proc_tree_update", (item_id, "Result", result))

            with self._progress_lock:
                self._completed += 1
                pct = (self._completed / self._total) * 100
                self._bus("post_proc_progress", pct)

            self._job_queue.task_done()
            self._bus("post_proc_log", f"[{case_info.case_name}] GC complete.")
            gc.collect()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, case_info: CaseInfo) -> bool:
        """Execute all enabled post-processing steps.  Returns True on success."""
        case_path = case_info.path
        case_name = case_info.case_name
        cfg = self._cfg

        # --- Find main .out file ---
        self._bus("post_proc_log", f"[{case_name}] Searching for .out file…")
        out_files = [
            f for f in case_path.glob("*.out")
            if "MD.out" not in f.name and "MoorDyn.out" not in f.name
        ]
        if not out_files:
            self._bus(
                "post_proc_log",
                f"[{case_name}] ERROR: No .out file found. Simulation may have failed.",
            )
            return False

        main_out = out_files[0]
        if len(out_files) > 1:
            self._bus(
                "post_proc_log",
                f"[{case_name}] WARNING: Multiple .out files; using '{main_out.name}'",
            )

        # --- Determine analysis start time ---
        analysis_start = _DEFAULT_ANALYSIS_START_TIME
        try:
            fst_content = (case_path / case_info.fst_file).read_text()
            m = re.search(
                r"^\s*([\d.eE+-]+)\s+TMax", fst_content, re.IGNORECASE | re.MULTILINE
            )
            if m:
                analysis_start = float(m.group(1)) / 3.0
        except Exception:
            pass
        self._bus(
            "post_proc_log",
            f"[{case_name}] Analysis start time: {analysis_start:.2f}s",
        )

        csv_name = f"{case_path.name}_{main_out.name}".replace(".out", ".csv")
        csv_path = case_path / csv_name
        overall_success = True

        # --- Step 1: Convert ---
        if cfg.run_convert_csv:
            overall_success &= self._step_convert(case_name, main_out, csv_path)
            if not overall_success:
                return False

        # --- Step 2: d'Alembert ---
        if cfg.run_dalembert:
            overall_success &= self._step_dalembert(
                case_name, case_path, case_info.fst_file, main_out, analysis_start
            )

        # --- Step 3: Plotting ---
        if cfg.run_plotting and csv_path.exists():
            overall_success &= self._step_plotting(
                case_name, case_path, csv_path, analysis_start
            )

        # --- Step 4: Frequency analysis ---
        if cfg.run_frequency_analysis and csv_path.exists():
            overall_success &= self._step_frequency(
                case_name, case_path, csv_path, analysis_start
            )

        return overall_success

    def _step_convert(self, case_name: str, out_file: Path, csv_path: Path) -> bool:
        try:
            from processing import ConverterRunner  # type: ignore
            converter = ConverterRunner(self._bus, case_name, "post_proc_log")
            if not converter.convert_openfast_to_csv_robust(str(out_file), str(csv_path)):
                self._bus("post_proc_log", f"[{case_name}] CSV conversion failed.")
                return False
            return True
        except Exception as exc:
            self._bus(
                "post_proc_log",
                f"[{case_name}] FATAL ERROR in CSV conversion: {exc}\n{traceback.format_exc()}",
            )
            return False

    def _step_dalembert(
        self,
        case_name: str,
        case_path: Path,
        fst_file: str,
        out_file: Path,
        analysis_start: float,
    ) -> bool:
        try:
            from processing import DalembertRunner  # type: ignore
            dal_dir = case_path / "dalembert_analysis"
            dal_dir.mkdir(exist_ok=True)
            DalembertRunner(self._bus, case_name, "post_proc_log").run(
                fst=str(case_path / fst_file),
                glue_out=str(out_file),
                outdir=str(dal_dir),
                analysis_start_time=analysis_start,
            )
            return True
        except Exception as exc:
            self._bus(
                "post_proc_log",
                f"[{case_name}] ERROR in d'Alembert: {exc}\n{traceback.format_exc()}",
            )
            return False

    def _step_plotting(
        self,
        case_name: str,
        case_path: Path,
        csv_path: Path,
        analysis_start: float,
    ) -> bool:
        with self._plotting_lock:
            try:
                from processing import PlottingRunner  # type: ignore
                plot_dir = case_path / "plots"
                plot_dir.mkdir(exist_ok=True)
                PlottingRunner(self._bus, case_name, "post_proc_log").run(
                    csv_file=str(csv_path),
                    output_dir=str(plot_dir),
                    case_name=case_name,
                    mean_start=analysis_start,
                    always_minmax=False,
                    minmax_range_frac=0.05,
                    minmax_abs=0.0,
                )
                return True
            except Exception as exc:
                self._bus(
                    "post_proc_log",
                    f"[{case_name}] ERROR in plotting: {exc}\n{traceback.format_exc()}",
                )
                return False

    def _step_frequency(
        self,
        case_name: str,
        case_path: Path,
        csv_path: Path,
        analysis_start: float,
    ) -> bool:
        with self._plotting_lock:
            try:
                from processing import FrequencyAnalysisRunner, SCIPY_AVAILABLE  # type: ignore
                if not SCIPY_AVAILABLE:
                    self._bus(
                        "post_proc_log",
                        f"[{case_name}] Frequency analysis skipped (scipy not available).",
                    )
                    return True
                freq_dir = case_path / "frequency_analysis"
                freq_dir.mkdir(exist_ok=True)
                FrequencyAnalysisRunner(self._bus, case_name, "post_proc_log").run(
                    csv_file=str(csv_path),
                    column_name=self._cfg.frequency_analysis_column,
                    output_dir=str(freq_dir),
                    start_time=analysis_start,
                )
                return True
            except Exception as exc:
                self._bus(
                    "post_proc_log",
                    f"[{case_name}] ERROR in frequency analysis: {exc}\n{traceback.format_exc()}",
                )
                return False
