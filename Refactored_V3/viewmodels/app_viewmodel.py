"""
viewmodels/app_viewmodel.py
============================
Central MVVM ViewModel for the OpenFAST Workflow Manager.

Responsibilities
----------------
* Owns all presentation state as ``ObservableProperty`` attributes.
* Exposes ``Command`` objects that the View layer can bind to buttons.
* Delegates every non-trivial action to the appropriate ``core/`` or
  ``services/`` module.
* Acts as the *message bus* — all background-thread messages flow into a
  ``queue.Queue`` that the View polls and dispatches to widget updates.

The ViewModel has **no** tkinter import.  It is fully testable headless.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.models import (
    AppConfig,
    CaseInfo,
    DistributionType,
    FileInfo,
    GeometryCase,
    ParameterInfo,
    ParameterVariation,
)
from core.parameter_engine import discover_all_files
from core.geometry_service import validate_geometry_csv_columns
from core.case_generator import generate_test_cases
from services.execution_service import SimulationRunService
from services.post_processing_service import PostProcessingService, PostProcConfig
from viewmodels.observable import Command, ObservableList, ObservableMixin, ObservableProperty

logger = logging.getLogger(__name__)


class AppViewModel(ObservableMixin):
    """
    Single ViewModel that drives the entire application.

    All public attributes that start with an upper-case letter are
    :class:`~viewmodels.observable.ObservableProperty` instances; the View
    subscribes to them to keep widgets in sync.
    """

    # ------------------------------------------------------------------
    # Observable properties (presentation state)
    # ------------------------------------------------------------------
    base_fst_path = ObservableProperty("")
    output_dir = ObservableProperty("")
    openfast_exe = ObservableProperty("")
    num_cases = ObservableProperty(10)
    num_threads = ObservableProperty(1)
    distribution = ObservableProperty(DistributionType.GRID_SEARCH.value)
    discovery_status = ObservableProperty("Select a .fst file and click 'Discover Parameters'")
    geometry_status = ObservableProperty("No geometry file loaded.")
    geometry_csv_path = ObservableProperty("")

    # Post-processing task flags
    run_convert_csv = ObservableProperty(True)
    run_dalembert = ObservableProperty(True)
    run_plotting = ObservableProperty(True)
    run_frequency_analysis = ObservableProperty(False)
    frequency_analysis_column = ObservableProperty("PtfmHeave")

    def __init__(self) -> None:
        super().__init__()

        # Default values
        self.output_dir = str(Path.cwd() / "test_cases")
        self.num_threads = max(1, (os.cpu_count() or 2) // 2)

        # Domain state (not directly observable, exposed via commands/methods)
        self.file_structure: Dict[str, FileInfo] = {}
        self.discovered_parameters: Dict[str, Dict[str, ParameterInfo]] = {}
        self.parameter_variations: ObservableList[ParameterVariation] = ObservableList()
        self.geometry_cases: List[GeometryCase] = []
        self.run_cases: Dict[str, CaseInfo] = {}
        self.post_proc_cases: Dict[str, CaseInfo] = {}

        # Thread-safe message bus (consumed by the View via poll loop)
        self.message_queue: queue.Queue = queue.Queue()

        # Commands (bound to buttons in the View)
        self.cmd_browse_fst = Command(self._browse_fst_requested)
        self.cmd_browse_output_dir = Command(self._browse_output_dir_requested)
        self.cmd_browse_openfast_exe = Command(self._browse_openfast_exe_requested)
        self.cmd_browse_geometry_csv = Command(self._browse_geometry_csv_requested)
        self.cmd_discover_parameters = Command(
            self._do_discover_parameters,
            lambda: bool(self.base_fst_path),
        )
        self.cmd_generate_cases = Command(
            self._do_generate_cases,
            lambda: bool(self.base_fst_path and self.file_structure),
        )
        self.cmd_load_run_cases = Command(self._do_load_run_cases)
        self.cmd_run_simulations = Command(
            self._do_run_simulations,
            lambda: bool(self.run_cases and self.openfast_exe),
        )
        self.cmd_load_post_proc_cases = Command(self._do_load_post_proc_cases)
        self.cmd_run_post_proc = Command(
            self._do_run_post_proc,
            lambda: bool(self.post_proc_cases),
        )
        self.cmd_save_config = Command(self._do_save_config)
        self.cmd_load_config = Command(self._do_load_config)
        self.cmd_clear_parameters = Command(self._do_clear_parameters)

    # ------------------------------------------------------------------
    # Message bus helpers (View calls these to post/consume messages)
    # ------------------------------------------------------------------

    def post_message(self, channel: str, payload: Any = None) -> None:
        """Used by background services to post a message."""
        self.message_queue.put((channel, payload))

    # ------------------------------------------------------------------
    # Browse commands (signal-only — actual dialog shown by View)
    # ------------------------------------------------------------------

    def _browse_fst_requested(self) -> None:
        """Signal the View to open an FST file dialog."""
        self.message_queue.put(("dialog_browse_fst", None))

    def _browse_output_dir_requested(self) -> None:
        self.message_queue.put(("dialog_browse_output_dir", None))

    def _browse_openfast_exe_requested(self) -> None:
        self.message_queue.put(("dialog_browse_openfast_exe", None))

    def _browse_geometry_csv_requested(self) -> None:
        self.message_queue.put(("dialog_browse_geometry_csv", None))

    # ------------------------------------------------------------------
    # Geometry CSV loading (called by View after user picks a file)
    # ------------------------------------------------------------------

    def load_geometry_csv(self, path: str) -> bool:
        """
        Parse and validate a geometry CSV file.

        Returns ``True`` on success, ``False`` on error.
        The View should call this after the user selects a file.
        """
        try:
            df = pd.read_csv(path)
            validate_geometry_csv_columns(set(df.columns))
            self.geometry_cases = [
                GeometryCase(id=row.get("ID", i), data=dict(row))
                for i, row in enumerate(df.to_dict("records"))
            ]
            self.geometry_csv_path = path
            self.geometry_status = f"Loaded {len(self.geometry_cases)} geometry case(s)."
            self._setup_log(f"Loaded {len(self.geometry_cases)} geometry cases from {Path(path).name}")
            return True
        except Exception as exc:
            self._setup_log(f"Error loading geometry CSV: {exc}")
            self.geometry_cases = []
            self.geometry_csv_path = ""
            self.geometry_status = f"Error: {exc}"
            return False

    # ------------------------------------------------------------------
    # Parameter discovery
    # ------------------------------------------------------------------

    def _do_discover_parameters(self) -> None:
        self._setup_log("Starting parameter discovery…")
        self.discovery_status = "Scanning…"
        try:
            file_info_map = discover_all_files(
                Path(self.base_fst_path),
                progress_callback=self._setup_log,
            )
            self.file_structure = file_info_map
            self.discovered_parameters = {
                key: info.parameters
                for key, info in file_info_map.items()
                if info.parameters
            }
            total = sum(len(p) for p in self.discovered_parameters.values())
            self.discovery_status = (
                f"Discovered {total} parameters across {len(self.file_structure)} files."
            )
            self._setup_log(f"Discovery complete: {len(self.file_structure)} files.")
            self.message_queue.put(("discovery_complete", self.discovered_parameters))
        except Exception as exc:
            self.discovery_status = f"Error: {exc}"
            self._setup_log(f"Error during discovery: {exc}\n{traceback.format_exc()}")
            self.message_queue.put(("error", f"Failed to discover parameters: {exc}"))

    # ------------------------------------------------------------------
    # Parameter variation management
    # ------------------------------------------------------------------

    def add_parameter_variation(self, variation: ParameterVariation) -> bool:
        """Add a parameter variation. Returns False if already present."""
        already = any(
            v.param_info.file_key == variation.param_info.file_key
            and v.param_info.name == variation.param_info.name
            for v in self.parameter_variations
        )
        if already:
            self._setup_log(
                f"Parameter {variation.param_info.file_key}/{variation.param_info.name} already added."
            )
            return False
        self.parameter_variations.append(variation)
        return True

    def remove_parameter_variation(self, variation: ParameterVariation) -> None:
        if variation in self.parameter_variations:
            self.parameter_variations.remove(variation)

    def _do_clear_parameters(self) -> None:
        self.parameter_variations.clear()

    # ------------------------------------------------------------------
    # Case generation
    # ------------------------------------------------------------------

    def _do_generate_cases(self, confirm_large: Optional[Callable[[int], bool]] = None) -> None:
        """
        Generate test cases.  *confirm_large* is a callable the View provides
        that shows a confirmation dialog for large batches.
        """
        if not self.file_structure:
            self.message_queue.put(("error", "Run 'Discover Parameters' first."))
            return
        if not self.parameter_variations and not self.geometry_cases:
            self.message_queue.put(
                ("error", "Add at least one parameter variation or import a geometry CSV.")
            )
            return

        self._setup_log("Generating test cases…")
        try:
            summary = generate_test_cases(
                base_fst_path=Path(self.base_fst_path),
                output_path=Path(self.output_dir),
                file_structure=self.file_structure,
                discovered_parameters=self.discovered_parameters,
                parameter_variations=list(self.parameter_variations),
                geometry_cases=self.geometry_cases,
                distribution=DistributionType(self.distribution),
                num_samples=self.num_cases,
                log=self._setup_log,
                confirm_large=confirm_large,
            )
            self.message_queue.put(("cases_generated", summary))
        except Exception as exc:
            self._setup_log(f"Error generating cases: {exc}\n{traceback.format_exc()}")
            self.message_queue.put(("error", f"Failed to generate cases: {exc}"))

    # ------------------------------------------------------------------
    # Load cases
    # ------------------------------------------------------------------

    def _load_cases_from_dir(self, test_dir: str) -> Dict[str, CaseInfo]:
        """Internal helper: read summary JSON and return {item_id: CaseInfo}."""
        summary_file = Path(test_dir) / "test_cases_summary.json"
        if not summary_file.exists():
            self.message_queue.put(
                ("error", f"'test_cases_summary.json' not found in {test_dir}")
            )
            return {}

        with open(summary_file, "r") as f:
            summary = json.load(f)

        cases: Dict[str, CaseInfo] = {}
        for i, case_data in enumerate(summary.get("test_cases", [])):
            item_id = f"item_{i}"
            ci = CaseInfo.from_dict(case_data, base_dir=Path(test_dir))
            cases[item_id] = ci
        return cases

    def _do_load_run_cases(self) -> None:
        cases = _load_cases_from_dir_or_dialog(self)
        if cases:
            self.run_cases = cases
            self.message_queue.put(("run_cases_loaded", cases))

    def _do_load_post_proc_cases(self) -> None:
        cases = _load_cases_from_dir_or_dialog(self)
        if cases:
            self.post_proc_cases = cases
            self.message_queue.put(("post_proc_cases_loaded", cases))

    # ------------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------------

    def _do_run_simulations(self, selected_ids: List[str]) -> None:
        selected = {k: v for k, v in self.run_cases.items() if k in selected_ids}
        if not selected:
            self.message_queue.put(("warn", "No cases selected."))
            return

        svc = SimulationRunService(
            message_bus=self.post_message,
            openfast_exe=self.openfast_exe,
            num_workers=self.num_threads,
        )
        svc.start(selected)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _do_run_post_proc(self, selected_ids: List[str]) -> None:
        selected = {k: v for k, v in self.post_proc_cases.items() if k in selected_ids}
        if not selected:
            self.message_queue.put(("warn", "No cases selected."))
            return

        cfg = PostProcConfig(
            run_convert_csv=self.run_convert_csv,
            run_dalembert=self.run_dalembert,
            run_plotting=self.run_plotting,
            run_frequency_analysis=self.run_frequency_analysis,
            frequency_analysis_column=self.frequency_analysis_column,
            num_workers=self.num_threads,
        )
        svc = PostProcessingService(message_bus=self.post_message, config=cfg)
        svc.start(selected)

    # ------------------------------------------------------------------
    # Config save / load
    # ------------------------------------------------------------------

    def build_config(self) -> AppConfig:
        """Snapshot the current state into an :class:`AppConfig`."""
        params_data: List[Dict[str, Any]] = []
        for v in self.parameter_variations:
            pd_entry: Dict[str, Any] = {
                "file_key": v.param_info.file_key,
                "param_name": v.param_info.name,
                "start": v.start,
                "end": v.end,
                "steps": v.steps,
                "int_mode": v.int_mode,
                "int_list": v.int_list,
                "bool_choice": v.bool_choice,
                "options_list": v.options_list,
                "csv_values": v.csv_values,
            }
            params_data.append(pd_entry)

        return AppConfig(
            base_fst_path=self.base_fst_path,
            output_dir=self.output_dir,
            num_cases=self.num_cases,
            distribution=self.distribution,
            geometry_csv_path=self.geometry_csv_path,
            parameters=params_data,
        )

    def _do_save_config(self, path: str) -> None:
        try:
            cfg = self.build_config()
            with open(path, "w") as f:
                json.dump(cfg.__dict__, f, indent=4)
            self._setup_log(f"Configuration saved to: {path}")
        except Exception as exc:
            self.message_queue.put(("error", f"Failed to save config: {exc}"))

    def _do_load_config(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            cfg = AppConfig(**{k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__})
            self.base_fst_path = cfg.base_fst_path
            self.output_dir = cfg.output_dir
            self.num_cases = cfg.num_cases
            self.distribution = cfg.distribution

            if cfg.geometry_csv_path and Path(cfg.geometry_csv_path).exists():
                self.load_geometry_csv(cfg.geometry_csv_path)
            elif cfg.geometry_csv_path:
                self._setup_log(f"Warning: Geometry CSV not found: {cfg.geometry_csv_path}")

            # Re-run discovery if needed
            if cfg.base_fst_path and not self.discovered_parameters:
                self._do_discover_parameters()

            # Re-create parameter variations
            self._do_clear_parameters()
            for pd_entry in cfg.parameters:
                fk = pd_entry.get("file_key", "")
                pn = pd_entry.get("param_name", "")
                if fk in self.discovered_parameters and pn in self.discovered_parameters[fk]:
                    pinfo = self.discovered_parameters[fk][pn]
                    v = ParameterVariation(
                        param_info=pinfo,
                        start=pd_entry.get("start", 0.0),
                        end=pd_entry.get("end", 1.0),
                        steps=pd_entry.get("steps", 5),
                        int_mode=pd_entry.get("int_mode", "Range"),
                        int_list=pd_entry.get("int_list", []),
                        bool_choice=pd_entry.get("bool_choice", "Vary (True & False)"),
                        options_list=pd_entry.get("options_list", []),
                        csv_values=pd_entry.get("csv_values", []),
                    )
                    self.parameter_variations.append(v)

            self._setup_log(f"Configuration loaded from: {path}")
            self.message_queue.put(("config_loaded", cfg))
        except Exception as exc:
            self.message_queue.put(("error", f"Failed to load config: {exc}"))
            self._setup_log(f"Error loading config: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_log(self, message: str) -> None:
        """Post a message to the setup-tab log channel."""
        self.message_queue.put(("setup_log", message))


def _load_cases_from_dir_or_dialog(vm: AppViewModel) -> Dict[str, CaseInfo]:
    """
    Load cases from *vm.output_dir*.  If the directory has no summary, post
    a dialog request so the View can ask the user to pick a directory.
    """
    test_dir = vm.output_dir
    summary_file = Path(test_dir) / "test_cases_summary.json" if test_dir else None

    if not summary_file or not summary_file.exists():
        vm.message_queue.put(("dialog_browse_test_dir", None))
        return {}

    return vm._load_cases_from_dir(test_dir)
