"""
views/setup_tab.py
==================
"1. Setup Cases" tab view.

All user actions are delegated to the ViewModel via Commands or direct
method calls.  This file imports tkinter freely but never contains
business logic.
"""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

from core.models import ParameterVariation
from viewmodels.app_viewmodel import AppViewModel
from views.widgets.parameter_row import ParameterRow


class SetupTab(ttk.Frame):
    """
    Renders the Setup Cases tab and binds it to an :class:`AppViewModel`.
    """

    def __init__(self, parent: ttk.Widget, vm: AppViewModel) -> None:
        super().__init__(parent)
        self._vm = vm
        self._param_rows: List[ParameterRow] = []
        self._build()
        self._bind_vm()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build(self) -> None:
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)
        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        self._canvas_window = canvas.create_window(
            (0, 0), window=self._scroll_frame, anchor="nw"
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(self._canvas_window, width=e.width),
        )
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        sf = self._scroll_frame
        self._build_file_selection(sf)
        self._build_test_config(sf)
        self._build_geometry_section(sf)
        self._build_parameter_discovery(sf)
        self._build_parameter_section(sf)
        self._build_action_section(sf)

        # Log area at the bottom
        log_frame = ttk.LabelFrame(self, text="Output Log", padding="10")
        log_frame.pack(fill="x", side="bottom", pady=5, padx=5)
        self._log_widget = scrolledtext.ScrolledText(
            log_frame, height=6, wrap=tk.WORD, bg="#f0f0f0", relief="sunken", borderwidth=1
        )
        self._log_widget.pack(fill="both", expand=False)

    def _build_file_selection(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        frame.pack(fill="x", pady=5, padx=5)

        ttk.Label(frame, text="Base FST File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self._fst_entry = ttk.Entry(frame, width=60)
        self._fst_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse", command=self._browse_fst).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self._output_entry = ttk.Entry(frame, width=60)
        self._output_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse", command=self._browse_output).grid(row=1, column=2, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)

    def _build_test_config(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Test Configuration", padding="10")
        frame.pack(fill="x", pady=5, padx=5)

        ttk.Label(frame, text="Number of Test Cases:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self._num_cases_var = tk.IntVar(value=10)
        self._num_cases_spinbox = ttk.Spinbox(
            frame, from_=2, to=10000, textvariable=self._num_cases_var, width=10
        )
        self._num_cases_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(frame, text="Parameter Distribution:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self._dist_var = tk.StringVar(value="grid_search")
        dist_combo = ttk.Combobox(
            frame,
            textvariable=self._dist_var,
            values=["grid_search", "csv_columnwise", "latin_hypercube", "uniform", "normal"],
            width=20,
            state="readonly",
        )
        dist_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        dist_combo.bind("<<ComboboxSelected>>", self._on_distribution_change)

        self._dist_help = ttk.Label(
            frame, text="Controls how parameters are varied.", foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        self._dist_help.grid(row=1, column=2, sticky="w", padx=10)
        ttk.Button(frame, text="Refresh", command=self._on_distribution_change).grid(
            row=1, column=3, padx=5, pady=5
        )
        frame.columnconfigure(2, weight=1)

    def _build_geometry_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Geometry Import (Optional)", padding="10")
        frame.pack(fill="x", pady=5, padx=5)

        ttk.Label(frame, text="Geometry CSV File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self._geo_entry = ttk.Entry(frame, width=60, state="readonly")
        self._geo_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse & Import", command=self._browse_geometry_csv).grid(
            row=0, column=2, padx=5
        )
        self._geo_status_lbl = ttk.Label(frame, text="No geometry file loaded.", foreground="gray")
        self._geo_status_lbl.grid(row=1, column=1, sticky="w", padx=5, pady=(2, 0))
        frame.columnconfigure(1, weight=1)

    def _build_parameter_discovery(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Parameter Discovery", padding="10")
        frame.pack(fill="x", pady=5, padx=5)
        ttk.Button(
            frame, text="Discover Parameters",
            command=self._discover_parameters,
            style="Accent.TButton",
        ).pack(side="left", padx=5)
        self._discovery_status_lbl = ttk.Label(
            frame, text="Select a .fst file and click 'Discover Parameters'"
        )
        self._discovery_status_lbl.pack(side="left", padx=20)

    def _build_parameter_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Parameter Configuration", padding="10")
        frame.pack(fill="x", pady=5, padx=5)

        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=5)
        ttk.Button(ctrl, text="Add from Discovery", command=self._show_param_selector).pack(side="left", padx=5)
        ttk.Button(ctrl, text="Clear All", command=self._clear_parameters).pack(side="left", padx=5)

        scroll_container = ttk.Frame(frame, height=250)
        scroll_container.pack(fill="x", pady=5)
        scroll_container.pack_propagate(False)

        canvas = tk.Canvas(scroll_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        self._param_list_frame = ttk.Frame(canvas)
        self._param_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._param_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

    def _build_action_section(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, padding="5")
        frame.pack(fill="x", pady=10)
        ttk.Button(
            frame, text="Generate Test Cases",
            command=self._generate_cases,
            style="Accent.TButton",
        ).pack(side="left", padx=5)
        ttk.Button(frame, text="Load Configuration", command=self._load_config).pack(side="left", padx=5)
        ttk.Button(frame, text="Save Configuration", command=self._save_config).pack(side="left", padx=5)
        ttk.Button(frame, text="View File Structure", command=self._show_file_structure).pack(side="left", padx=5)

    # ------------------------------------------------------------------
    # VM bindings
    # ------------------------------------------------------------------

    def _bind_vm(self) -> None:
        vm = self._vm

        # Sync text entries from VM → widget on change
        vm.subscribe("base_fst_path", lambda o, n: self._sync_entry(self._fst_entry, n))
        vm.subscribe("output_dir", lambda o, n: self._sync_entry(self._output_entry, n))
        vm.subscribe("discovery_status", lambda o, n: self._discovery_status_lbl.config(text=n))
        vm.subscribe("geometry_status", lambda o, n: self._geo_status_lbl.config(text=n))
        vm.subscribe("geometry_csv_path", lambda o, n: self._sync_entry_readonly(self._geo_entry, n))
        vm.subscribe("num_cases", lambda o, n: self._num_cases_var.set(n))
        vm.subscribe("distribution", lambda o, n: self._dist_var.set(n))

        # Initial sync
        self._sync_entry(self._fst_entry, vm.base_fst_path)
        self._sync_entry(self._output_entry, vm.output_dir)

        # Sync text → VM when user types
        self._fst_entry.bind("<FocusOut>", lambda e: setattr(vm, "base_fst_path", self._fst_entry.get()))
        self._output_entry.bind("<FocusOut>", lambda e: setattr(vm, "output_dir", self._output_entry.get()))

        # Sync num_cases spinbox → VM
        self._num_cases_var.trace_add("write", lambda *_: self._sync_num_cases())

        # Param variations list
        vm.parameter_variations.on_change = self._refresh_param_rows

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse_fst(self) -> None:
        path = filedialog.askopenfilename(
            title="Select base FST file",
            filetypes=[("FST files", "*.fst"), ("All files", "*.*")],
        )
        if path:
            self._vm.base_fst_path = path
            if messagebox.askyesno("Discover Parameters", "Discover parameters for this file now?"):
                self._discover_parameters()

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select Output Directory", initialdir=self._vm.output_dir)
        if d:
            self._vm.output_dir = d

    def _browse_geometry_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Geometry CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            ok = self._vm.load_geometry_csv(path)
            if not ok:
                messagebox.showerror("Error", self._vm.geometry_status)

    def _discover_parameters(self) -> None:
        self._vm.cmd_discover_parameters()

    def _on_distribution_change(self, event=None) -> None:
        self._vm.distribution = self._dist_var.get()
        is_sampling = self._dist_var.get() in {"latin_hypercube", "uniform", "normal"}
        is_fixed = self._dist_var.get() in {"grid_search", "csv_columnwise"}
        self._num_cases_spinbox.config(state="disabled" if is_fixed else "normal")
        for row in self._param_rows:
            row.apply_distribution_mode(self._dist_var.get())
        self._update_total_cases()

    def _show_param_selector(self) -> None:
        if not self._vm.discovered_parameters:
            messagebox.showinfo("Info", "Run 'Discover Parameters' first.")
            return
        _ParameterSelectorDialog(self, self._vm, self._add_param_row)

    def _clear_parameters(self) -> None:
        for row in self._param_rows:
            row.destroy()
        self._param_rows.clear()
        self._vm.cmd_clear_parameters()

    def _add_param_row(self, file_key: str, param_name: str, param_info: Dict[str, Any]) -> None:
        # Check duplicate
        if any(r.file_key == file_key and r.param_name == param_name for r in self._param_rows):
            self._append_log(f"Parameter {file_key}/{param_name} is already added.")
            return

        row = ParameterRow(
            parent=self._param_list_frame,
            file_key=file_key,
            param_name=param_name,
            param_info=param_info,
            on_remove=self._remove_param_row,
            on_change=self._update_total_cases,
        )
        row.pack(fill="x", pady=4, padx=2)
        row.apply_distribution_mode(self._dist_var.get())
        self._param_rows.append(row)

        # Build ParameterVariation and push to VM
        kwargs = row.get_variation_kwargs()
        from core.models import ParameterVariation
        pinfo = self._vm.discovered_parameters[file_key][param_name]
        variation = ParameterVariation(param_info=pinfo, **kwargs)
        self._vm.add_parameter_variation(variation)
        self._update_total_cases()

    def _remove_param_row(self, row: ParameterRow) -> None:
        # Remove from ViewModel
        for v in list(self._vm.parameter_variations):
            if v.param_info.file_key == row.file_key and v.param_info.name == row.param_name:
                self._vm.remove_parameter_variation(v)
                break
        row.destroy()
        self._param_rows.remove(row)
        self._update_total_cases()

    def _refresh_param_rows(self) -> None:
        """Called when VM's parameter_variations list changes externally (e.g. load config)."""
        # Rebuild all rows from scratch
        for r in self._param_rows:
            r.destroy()
        self._param_rows.clear()
        for v in self._vm.parameter_variations:
            pinfo_dict = {
                "type": v.param_info.type.value,
                "original_value": v.param_info.original_value,
                "unit": v.param_info.unit,
                "description": v.param_info.description,
            }
            row = ParameterRow(
                parent=self._param_list_frame,
                file_key=v.param_info.file_key,
                param_name=v.param_info.name,
                param_info=pinfo_dict,
                on_remove=self._remove_param_row,
                on_change=self._update_total_cases,
            )
            row.pack(fill="x", pady=4, padx=2)
            row.apply_distribution_mode(self._dist_var.get())
            self._param_rows.append(row)
        self._update_total_cases()

    def _update_total_cases(self) -> None:
        dist_mode = self._dist_var.get()
        total = 0
        try:
            if dist_mode == "grid_search":
                total = 1 if self._param_rows else 0
                for row in self._param_rows:
                    total *= row.count_values(dist_mode)
            elif dist_mode == "csv_columnwise":
                if self._param_rows:
                    total = self._param_rows[0].count_values(dist_mode)
        except Exception:
            total = 0
        if dist_mode not in {"latin_hypercube", "uniform", "normal"}:
            self._num_cases_var.set(total)
            self._vm.num_cases = total

    def _generate_cases(self) -> None:
        # Sync all param row data to VM before generating
        self._sync_variations_to_vm()
        self._vm._do_generate_cases(
            confirm_large=lambda n: messagebox.askyesno("Large Job", f"Generate {n} cases?")
        )

    def _sync_variations_to_vm(self) -> None:
        """Push current row widget values back into VM's ParameterVariation objects."""
        for row, variation in zip(self._param_rows, self._vm.parameter_variations):
            kwargs = row.get_variation_kwargs()
            for k, v in kwargs.items():
                if hasattr(variation, k):
                    setattr(variation, k, v)

    def _save_config(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON config", "*.json")],
        )
        if path:
            self._vm._do_save_config(path)

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON config", "*.json")],
        )
        if path:
            self._vm._do_load_config(path)

    def _show_file_structure(self) -> None:
        if not self._vm.file_structure:
            messagebox.showinfo("Info", "Run 'Discover Parameters' first.")
            return
        _FileStructureDialog(self, self._vm)

    # ------------------------------------------------------------------
    # Message handler (called by MainWindow's poll loop)
    # ------------------------------------------------------------------

    def handle_message(self, channel: str, payload: Any) -> None:
        if channel == "setup_log":
            self._append_log(str(payload))
        elif channel == "discovery_complete":
            pass  # VM already updated discovery_status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_widget.insert(tk.END, f"[{ts}] {message}\n")
        self._log_widget.see(tk.END)

    def _sync_entry(self, widget: ttk.Entry, value: str) -> None:
        widget.delete(0, tk.END)
        widget.insert(0, value)

    def _sync_entry_readonly(self, widget: ttk.Entry, value: str) -> None:
        widget.config(state="normal")
        widget.delete(0, tk.END)
        widget.insert(0, value)
        widget.config(state="readonly")

    def _sync_num_cases(self) -> None:
        try:
            self._vm.num_cases = self._num_cases_var.get()
        except tk.TclError:
            pass


# ---------------------------------------------------------------------------
# Helper dialogs (nested to keep imports tight)
# ---------------------------------------------------------------------------

class _ParameterSelectorDialog(tk.Toplevel):
    def __init__(self, parent, vm: AppViewModel, on_add) -> None:
        super().__init__(parent)
        self.title("Select Parameters to Vary")
        self.geometry("900x700")
        self._vm = vm
        self._on_add = on_add

        search_frame = ttk.Frame(self)
        search_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side="left", padx=5)
        self._search_var = tk.StringVar()
        ttk.Entry(search_frame, textvariable=self._search_var, width=30).pack(side="left", padx=5)

        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self._tree = ttk.Treeview(
            tree_frame,
            columns=("Type", "Value", "Unit", "Description"),
            show="tree headings",
        )
        for col, w in [("#0", 200), ("Type", 80), ("Value", 100), ("Unit", 80), ("Description", 350)]:
            self._tree.heading(col, text=col if col != "#0" else "Parameter")
            self._tree.column(col, width=w)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self._all_items = []
        for fk, params in sorted(vm.discovered_parameters.items()):
            file_node = self._tree.insert("", "end", text=fk, open=False, tags=("file_node",))
            for pname, pinfo in sorted(params.items()):
                val = pinfo.original_value
                val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
                item = self._tree.insert(
                    file_node, "end", text=pname,
                    values=(pinfo.type.value, val_str, pinfo.unit, pinfo.description[:100]),
                )
                self._all_items.append((item, fk.lower(), pname.lower(), pinfo.description.lower()))
        self._tree.tag_configure("file_node", font=("TkDefaultFont", 10, "bold"))

        self._search_var.trace("w", self._search)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=10, padx=10)
        ttk.Button(btn_frame, text="Add Selected", command=self._add, style="Accent.TButton").pack(side="right")
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="right", padx=5)

    def _search(self, *_) -> None:
        term = self._search_var.get().lower()
        for child in self._tree.get_children():
            self._tree.item(child, open=False)
            self._tree.reattach(child, "", "end")
        if not term:
            return
        for child in self._tree.get_children():
            self._tree.detach(child)
        for item, fk, pn, desc in self._all_items:
            if term in pn or term in desc or term in fk:
                parent = self._tree.parent(item)
                self._tree.reattach(parent, "", "end")
                self._tree.item(parent, open=True)

    def _add(self) -> None:
        count = 0
        for item in self._tree.selection():
            parent = self._tree.parent(item)
            if parent:
                fk = self._tree.item(parent)["text"]
                pn = self._tree.item(item)["text"]
                pinfo_obj = self._vm.discovered_parameters[fk][pn]
                pinfo_dict = {
                    "type": pinfo_obj.type.value,
                    "original_value": pinfo_obj.original_value,
                    "unit": pinfo_obj.unit,
                    "description": pinfo_obj.description,
                }
                self._on_add(fk, pn, pinfo_dict)
                count += 1
        self.destroy()


class _FileStructureDialog(tk.Toplevel):
    def __init__(self, parent, vm: AppViewModel) -> None:
        super().__init__(parent)
        self.title("Discovered File Structure")
        self.geometry("800x600")
        from tkinter import scrolledtext
        text = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=("Consolas", 10))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("end", "OpenFAST File Structure:\n" + "=" * 60 + "\n\n")
        for fk, fi in sorted(vm.file_structure.items()):
            text.insert("end", f"{fk}:\n", "heading")
            text.insert("end", f"  Path: {fi.path}\n")
            text.insert("end", f"  Parameters Found: {len(vm.discovered_parameters.get(fk, {}))}\n\n")
        text.tag_config("heading", font=("Consolas", 11, "bold"), foreground="darkblue")
        text.config(state="disabled")
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=10)
