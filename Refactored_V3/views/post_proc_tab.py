"""
views/post_proc_tab.py
=======================
"3. Post-Process Results" tab view.
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

try:
    from processing import SCIPY_AVAILABLE  # type: ignore
except ImportError:
    SCIPY_AVAILABLE = False


class PostProcTab(ttk.Frame):
    """Renders the Post-Process tab and binds it to the ViewModel."""

    def __init__(self, parent: ttk.Widget, vm: AppViewModel) -> None:
        super().__init__(parent)
        self._vm = vm
        self._build()
        self._bind_vm()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", pady=5, padx=10)

        # --- Directory config ---
        config_frame = ttk.LabelFrame(top_frame, text="Configuration", padding="10")
        config_frame.pack(fill="x", expand=True, side="left", padx=(0, 5))
        ttk.Label(config_frame, text="Results Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self._dir_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self._dir_var, width=50).grid(
            row=0, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Button(config_frame, text="Browse", command=self._browse_dir).grid(
            row=0, column=2, padx=5, pady=2
        )
        config_frame.columnconfigure(1, weight=1)

        # --- Task flags ---
        tasks_frame = ttk.LabelFrame(top_frame, text="Tasks to Run", padding="10")
        tasks_frame.pack(fill="x", side="left", padx=5)

        self._convert_var = tk.BooleanVar(value=True)
        self._dalembert_var = tk.BooleanVar(value=True)
        self._plotting_var = tk.BooleanVar(value=True)
        self._freq_var = tk.BooleanVar(value=False)
        self._freq_col_var = tk.StringVar(value="PtfmHeave")

        ttk.Checkbutton(tasks_frame, text="Convert .out to .csv", variable=self._convert_var).pack(anchor="w")
        ttk.Checkbutton(tasks_frame, text="Run d'Alembert Analysis", variable=self._dalembert_var).pack(anchor="w")
        ttk.Checkbutton(tasks_frame, text="Generate Plots", variable=self._plotting_var).pack(anchor="w")

        freq_frame = ttk.Frame(tasks_frame)
        freq_frame.pack(anchor="w", fill="x", pady=(5, 0))
        freq_check = ttk.Checkbutton(
            freq_frame, text="Run Frequency Analysis on column:", variable=self._freq_var
        )
        freq_check.pack(side="left")
        freq_entry = ttk.Entry(freq_frame, textvariable=self._freq_col_var, width=18)
        freq_entry.pack(side="left", padx=5)

        if not SCIPY_AVAILABLE:
            freq_check.config(state="disabled")
            freq_entry.config(state="disabled")
            ttk.Label(
                tasks_frame, text="(Frequency Analysis requires 'scipy')",
                foreground="gray", font=("TkDefaultFont", 8),
            ).pack(anchor="w")

        # --- Task panel ---
        self._panel = TaskPanel(
            parent=self,
            title="Cases to Process",
            columns=("Status", "Parameters", "Result"),
            col_widths={"Status": 120, "Parameters": 400, "Result": 200},
            run_button_text="Run Post-Processing",
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
        vm.subscribe("output_dir", lambda o, n: self._dir_var.set(n))
        self._dir_var.trace_add("write", lambda *_: setattr(vm, "output_dir", self._dir_var.get()))

        # Sync checkbox flags → VM
        self._convert_var.trace_add("write", lambda *_: setattr(vm, "run_convert_csv", self._convert_var.get()))
        self._dalembert_var.trace_add("write", lambda *_: setattr(vm, "run_dalembert", self._dalembert_var.get()))
        self._plotting_var.trace_add("write", lambda *_: setattr(vm, "run_plotting", self._plotting_var.get()))
        self._freq_var.trace_add("write", lambda *_: setattr(vm, "run_frequency_analysis", self._freq_var.get()))
        self._freq_col_var.trace_add(
            "write", lambda *_: setattr(vm, "frequency_analysis_column", self._freq_col_var.get())
        )

        # Initial sync
        self._dir_var.set(vm.output_dir)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(title="Select Results Directory", initialdir=self._vm.output_dir)
        if d:
            self._vm.output_dir = d

    def _load_cases(self) -> None:
        test_dir = self._vm.output_dir
        summary = Path(test_dir) / "test_cases_summary.json" if test_dir else None
        if not summary or not summary.exists():
            test_dir = filedialog.askdirectory(title="Select Results Directory")
            if not test_dir:
                return
            self._vm.output_dir = test_dir

        cases = self._vm._load_cases_from_dir(test_dir)
        if not cases:
            messagebox.showerror("Error", f"No test_cases_summary.json found in {test_dir}")
            return
        self._vm.post_proc_cases = cases
        self._panel.populate(cases)
        self._panel.append_log(f"Loaded {len(cases)} cases from {test_dir}")

    def _run_selected(self, selected_ids: list) -> None:
        if not selected_ids:
            messagebox.showwarning("Warning", "No cases selected.")
            return
        tasks_on = any([
            self._convert_var.get(),
            self._dalembert_var.get(),
            self._plotting_var.get(),
            self._freq_var.get(),
        ])
        if not tasks_on:
            messagebox.showwarning("Warning", "No post-processing tasks selected.")
            return
        if self._freq_var.get() and not self._freq_col_var.get().strip():
            messagebox.showerror("Input Error", "Specify a column name for Frequency Analysis.")
            return
        if not messagebox.askyesno("Confirm", f"Run post-processing on {len(selected_ids)} case(s)?"):
            return

        self._panel.set_run_button_state(False)
        self._panel.set_progress(0)
        self._vm._do_run_post_proc(selected_ids)

    def _context_menu(self, event: tk.Event, tree: ttk.Treeview) -> None:
        item_id = tree.identify_row(event.y)
        if not item_id:
            return
        tree.selection_set(item_id)
        case_info = self._vm.post_proc_cases.get(item_id)
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
        if channel == "post_proc_log":
            self._panel.append_log(str(payload))
        elif channel == "post_proc_tree_update":
            item_id, col, val = payload
            self._panel.update_row(item_id, col, val)
        elif channel == "post_proc_progress":
            self._panel.set_progress(float(payload))
        elif channel == "enable_post_proc_button":
            self._panel.set_run_button_state(True)
        elif channel == "post_proc_cases_loaded":
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
