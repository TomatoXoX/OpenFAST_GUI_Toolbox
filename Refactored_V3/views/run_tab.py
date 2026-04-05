"""
views/run_tab.py
=================
"2. Run Simulations" tab view.
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from viewmodels.app_viewmodel import AppViewModel
from views.widgets.task_panel import TaskPanel


class RunTab(ttk.Frame):
    """Renders the Run Simulations tab and binds it to the ViewModel."""

    def __init__(self, parent: ttk.Widget, vm: AppViewModel) -> None:
        super().__init__(parent)
        self._vm = vm
        self._build()
        self._bind_vm()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # --- Configuration ---
        config_frame = ttk.LabelFrame(self, text="Run Configuration", padding="10")
        config_frame.pack(fill="x", pady=5, padx=10)

        ttk.Label(config_frame, text="OpenFAST Path:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self._exe_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self._exe_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Button(config_frame, text="Browse", command=self._browse_exe).grid(
            row=0, column=2, padx=5, pady=2
        )

        ttk.Label(config_frame, text="Parallel runs:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self._threads_var = tk.IntVar(value=1)
        ttk.Spinbox(
            config_frame,
            from_=1,
            to=os.cpu_count() or 8,
            textvariable=self._threads_var,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        config_frame.columnconfigure(1, weight=1)

        # --- Task panel ---
        self._panel = TaskPanel(
            parent=self,
            title="Test Cases to Run",
            columns=("Status", "Parameters", "Runtime", "Result"),
            col_widths={"Status": 180, "Parameters": 300, "Runtime": 100, "Result": 200},
            run_button_text="Run Selected Simulations",
            on_load=self._load_cases,
            on_run=self._run_selected,
            on_context_menu=self._context_menu,
        )
        self._panel.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # VM bindings
    # ------------------------------------------------------------------

    def _bind_vm(self) -> None:
        vm = self._vm
        vm.subscribe("openfast_exe", lambda o, n: self._exe_var.set(n))
        vm.subscribe("num_threads", lambda o, n: self._threads_var.set(n))
        self._exe_var.trace_add("write", lambda *_: setattr(vm, "openfast_exe", self._exe_var.get()))
        self._threads_var.trace_add("write", lambda *_: self._sync_threads())
        # Initial sync
        self._exe_var.set(vm.openfast_exe)
        self._threads_var.set(vm.num_threads)

    def _sync_threads(self) -> None:
        try:
            self._vm.num_threads = self._threads_var.get()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse_exe(self) -> None:
        path = filedialog.askopenfilename(
            title="Select OpenFAST executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if path:
            self._vm.openfast_exe = path

    def _load_cases(self) -> None:
        # Try current output_dir; fall back to dialog
        test_dir = self._vm.output_dir
        summary = Path(test_dir) / "test_cases_summary.json" if test_dir else None
        if not summary or not summary.exists():
            test_dir = filedialog.askdirectory(title="Select Test Case Directory")
            if not test_dir:
                return
            self._vm.output_dir = test_dir

        cases = self._vm._load_cases_from_dir(test_dir)
        if not cases:
            messagebox.showerror("Error", f"No test_cases_summary.json found in {test_dir}")
            return
        self._vm.run_cases = cases
        self._panel.populate(cases)
        self._panel.append_log(f"Loaded {len(cases)} cases from {test_dir}")

    def _run_selected(self, selected_ids: list) -> None:
        if not selected_ids:
            messagebox.showwarning("Warning", "No cases selected.")
            return
        if not self._vm.openfast_exe:
            messagebox.showerror("Error", "Please specify the OpenFAST executable path.")
            return
        if not messagebox.askyesno("Confirm", f"Run {len(selected_ids)} simulation(s)?"):
            return

        self._panel.set_run_button_state(False)
        self._panel.set_progress(0)
        self._vm._do_run_simulations(selected_ids)

    def _context_menu(self, event: tk.Event, tree: ttk.Treeview) -> None:
        item_id = tree.identify_row(event.y)
        if not item_id:
            return
        tree.selection_set(item_id)
        case_info = self._vm.run_cases.get(item_id)
        if not case_info:
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label=f"Open Folder for '{case_info.case_name}'",
            command=lambda: _open_folder(case_info.path),
        )
        menu.post(event.x_root, event.y_root)

    # ------------------------------------------------------------------
    # Message handler (called by MainWindow)
    # ------------------------------------------------------------------

    def handle_message(self, channel: str, payload: Any) -> None:
        if channel == "run_log":
            self._panel.append_log(str(payload))
        elif channel == "run_tree_update":
            item_id, col, val = payload
            self._panel.update_row(item_id, col, val)
        elif channel == "run_progress":
            self._panel.set_progress(float(payload))
        elif channel == "enable_run_button":
            self._panel.set_run_button_state(True)
        elif channel == "run_cases_loaded":
            self._panel.populate(payload)


def _open_folder(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore
        elif sys.platform == "darwin":
            import subprocess; subprocess.Popen(["open", str(path)])
        else:
            import subprocess; subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        messagebox.showerror("Error", f"Could not open folder: {exc}")
