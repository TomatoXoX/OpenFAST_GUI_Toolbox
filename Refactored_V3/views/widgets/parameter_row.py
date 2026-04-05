"""
views/widgets/parameter_row.py
================================
A single-row Tkinter widget for configuring one parameter variation.

Responsibilities
----------------
* Display parameter metadata (name, type, unit, current value).
* Expose range / list / bool / option configuration fields depending on type.
* Show/hide the appropriate sub-widgets when the distribution mode changes.
* Report its current settings back to the ViewModel when queried.
* Call an ``on_remove`` callback when the Remove button is clicked.
* Call an ``on_change`` callback whenever any value changes (so the
  ViewModel can recompute the total case count).

This widget interacts with the rest of the app only through callbacks —
it never imports the ViewModel or any core module directly.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional


class ParameterRow(ttk.Frame):
    """
    A single row in the parameter-configuration list.

    Parameters
    ----------
    parent:
        Parent Tkinter widget (the scrollable canvas inner frame).
    file_key:
        The file-key string (e.g. ``"ElastoDyn.dat"``).
    param_name:
        The parameter name (e.g. ``"PtfmMass"``).
    param_info:
        Dict with keys ``type``, ``original_value``, ``unit``, ``description``.
    on_remove:
        Callable invoked when the user clicks Remove, receives *this* row.
    on_change:
        Callable invoked when any value widget changes (no arguments).
    """

    def __init__(
        self,
        parent: ttk.Widget,
        file_key: str,
        param_name: str,
        param_info: Dict[str, Any],
        on_remove: Callable[["ParameterRow"], None],
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self.file_key = file_key
        self.param_name = param_name
        self.param_info = param_info
        self._on_remove = on_remove
        self._on_change = on_change

        param_type: str = param_info.get("type", "float")
        current_val: Any = param_info.get("original_value", 0.0)

        # --- Label ---
        ttk.Label(
            self,
            text=f"{file_key} — {param_name}",
            width=35,
            anchor="w",
            wraplength=220,
        ).grid(row=0, column=0, rowspan=2, padx=5, sticky="w")

        # --- CSV values (used for csv_columnwise mode) ---
        self.csv_var = tk.StringVar(value=str(current_val))
        self.csv_var.trace_add("write", lambda *_: on_change())
        self._csv_lbl = ttk.Label(self, text="CSV Values:")
        self._csv_ent = ttk.Entry(self, textvariable=self.csv_var, width=40)

        # --- Type-specific widgets ---
        self._type_widgets: Dict[str, tk.Widget] = {}

        if param_type == "float":
            sv = abs(float(current_val)) if isinstance(current_val, (int, float)) else 1.0
            start_default = (current_val * 0.8) if sv > 1e-9 else -1.0
            end_default   = (current_val * 1.2) if sv > 1e-9 else  1.0
            self.start_var  = tk.DoubleVar(value=start_default)
            self.end_var    = tk.DoubleVar(value=end_default)
            self.steps_var  = tk.IntVar(value=5)
            for v in (self.start_var, self.end_var, self.steps_var):
                v.trace_add("write", lambda *_: on_change())
            self._type_widgets = {
                "range_lbl_s":  ttk.Label(self, text="Start:"),
                "range_ent_s":  ttk.Entry(self, textvariable=self.start_var, width=10),
                "range_lbl_e":  ttk.Label(self, text="End:"),
                "range_ent_e":  ttk.Entry(self, textvariable=self.end_var, width=10),
                "range_lbl_st": ttk.Label(self, text="Steps:"),
                "range_spn_st": ttk.Spinbox(self, from_=1, to=100, textvariable=self.steps_var, width=5),
            }

        elif param_type == "int":
            self.int_mode_var = tk.StringVar(value="Range")
            self.start_var    = tk.DoubleVar(value=float(current_val))
            self.end_var      = tk.DoubleVar(value=float(current_val) + 4)
            self.steps_var    = tk.IntVar(value=5)
            self.list_var     = tk.StringVar(value=str(current_val))
            for v in (self.int_mode_var, self.start_var, self.end_var, self.steps_var, self.list_var):
                v.trace_add("write", lambda *_: on_change())

            def _update_int_widgets() -> None:
                is_range = self.int_mode_var.get() == "Range"
                for n, w in self._type_widgets.items():
                    if n.startswith("range_"):
                        w.grid() if is_range else w.grid_remove()
                    if n.startswith("list_"):
                        w.grid() if not is_range else w.grid_remove()
                on_change()

            self._update_int_widgets = _update_int_widgets
            self._type_widgets = {
                "rad_range":    ttk.Radiobutton(self, text="Range", variable=self.int_mode_var, value="Range", command=_update_int_widgets),
                "rad_list":     ttk.Radiobutton(self, text="List",  variable=self.int_mode_var, value="List",  command=_update_int_widgets),
                "range_lbl_s":  ttk.Label(self, text="Start:"),
                "range_ent_s":  ttk.Entry(self, textvariable=self.start_var, width=8),
                "range_lbl_e":  ttk.Label(self, text="End:"),
                "range_ent_e":  ttk.Entry(self, textvariable=self.end_var, width=8),
                "range_lbl_st": ttk.Label(self, text="Steps:"),
                "range_spn_st": ttk.Spinbox(self, from_=1, to=100, textvariable=self.steps_var, width=5),
                "list_lbl":     ttk.Label(self, text="List (CSV):"),
                "list_ent":     ttk.Entry(self, textvariable=self.list_var, width=25),
            }

        elif param_type == "bool":
            self.bool_var = tk.StringVar(value="Vary (True & False)")
            self.bool_var.trace_add("write", lambda *_: on_change())
            self._type_widgets = {
                "bool_lbl":   ttk.Label(self, text="Value:"),
                "bool_combo": ttk.Combobox(
                    self,
                    textvariable=self.bool_var,
                    values=["Vary (True & False)", "True", "False"],
                    width=20,
                    state="readonly",
                ),
            }

        elif param_type == "option":
            self.options_var = tk.StringVar(value=f'"{current_val}"')
            self.options_var.trace_add("write", lambda *_: on_change())
            self._type_widgets = {
                "opt_lbl": ttk.Label(self, text="Options (CSV):"),
                "opt_ent": ttk.Entry(self, textvariable=self.options_var, width=30),
            }

        # --- Info + Remove ---
        unit = param_info.get("unit", "")
        self._info_lbl = ttk.Label(
            self,
            text=f"[{unit}] (Type: {param_type}, Current: {current_val})",
            foreground="gray",
        )
        self._remove_btn = ttk.Button(
            self, text="Remove",
            command=lambda: self._on_remove(self),
        )

        # Layout will be applied by apply_distribution_mode
        self.columnconfigure(8, weight=1)

    # ------------------------------------------------------------------
    # Public: called by parent when distribution mode changes
    # ------------------------------------------------------------------

    def apply_distribution_mode(self, dist_mode: str) -> None:
        """
        Show/hide the appropriate sub-widgets for the given distribution mode.
        """
        param_type: str = self.param_info.get("type", "float")
        is_csv      = dist_mode == "csv_columnwise"
        is_sampling = dist_mode in {"latin_hypercube", "uniform", "normal"}

        # Hide everything first
        self._csv_lbl.grid_remove()
        self._csv_ent.grid_remove()
        for w in self._type_widgets.values():
            w.grid_remove()

        if is_csv:
            self._csv_lbl.grid(row=0, column=1, padx=(10, 2))
            self._csv_ent.grid(row=0, column=2, columnspan=5, sticky="ew")
        elif param_type == "float":
            self._type_widgets["range_lbl_s"].grid(row=0, column=1, padx=(10, 2))
            self._type_widgets["range_ent_s"].grid(row=0, column=2)
            self._type_widgets["range_lbl_e"].grid(row=0, column=3, padx=5)
            self._type_widgets["range_ent_e"].grid(row=0, column=4)
            self._type_widgets["range_lbl_st"].grid(row=0, column=5, padx=5)
            self._type_widgets["range_spn_st"].grid(row=0, column=6)
            if is_sampling:
                for name in ("range_lbl_st", "range_spn_st", "range_ent_e", "range_lbl_e"):
                    pass  # start / end still editable; steps disabled
                self._type_widgets["range_spn_st"].config(state="disabled")
                self._type_widgets["range_lbl_st"].config(state="disabled")
        elif param_type == "int":
            self._type_widgets["rad_range"].grid(row=0, column=1, sticky="w", padx=5)
            self._type_widgets["rad_list"].grid(row=1, column=1, sticky="w", padx=5)
            if hasattr(self, "_update_int_widgets"):
                self._update_int_widgets()
        elif param_type == "bool":
            self._type_widgets["bool_lbl"].grid(row=0, column=1, padx=(10, 2))
            self._type_widgets["bool_combo"].grid(row=0, column=2, columnspan=3)
        elif param_type == "option":
            self._type_widgets["opt_lbl"].grid(row=0, column=1, padx=(10, 2))
            self._type_widgets["opt_ent"].grid(row=0, column=2, columnspan=5, sticky="ew")

        self._info_lbl.grid(row=0, column=8, padx=5, sticky="w")
        self._remove_btn.grid(row=0, column=9, rowspan=2, padx=10)

    # ------------------------------------------------------------------
    # Public: extract current ParameterVariation data
    # ------------------------------------------------------------------

    def get_variation_kwargs(self) -> Dict[str, Any]:
        """
        Return a dict suitable for constructing a :class:`ParameterVariation`.
        """
        param_type = self.param_info.get("type", "float")
        kwargs: Dict[str, Any] = {
            "csv_values": _parse_csv_values(self.csv_var.get(), param_type),
        }
        if param_type == "float":
            kwargs.update(
                start=self.start_var.get(),
                end=self.end_var.get(),
                steps=self.steps_var.get(),
            )
        elif param_type == "int":
            kwargs.update(
                int_mode=self.int_mode_var.get(),
                start=self.start_var.get(),
                end=self.end_var.get(),
                steps=self.steps_var.get(),
                int_list=[
                    int(float(x.strip()))
                    for x in self.list_var.get().split(",")
                    if x.strip()
                ],
            )
        elif param_type == "bool":
            kwargs["bool_choice"] = self.bool_var.get()
        elif param_type == "option":
            kwargs["options_list"] = [
                o.strip().strip("'\"")
                for o in self.options_var.get().split(",")
                if o.strip()
            ]
        return kwargs

    def count_values(self, dist_mode: str) -> int:
        """Return the number of distinct values this row contributes."""
        param_type = self.param_info.get("type", "float")
        try:
            if dist_mode == "csv_columnwise":
                return len([x for x in self.csv_var.get().split(",") if x.strip()])
            if dist_mode == "grid_search":
                if param_type == "float":
                    return self.steps_var.get()
                if param_type == "int":
                    if self.int_mode_var.get() == "Range":
                        return self.steps_var.get()
                    return len([x for x in self.list_var.get().split(",") if x.strip()])
                if param_type == "bool":
                    return 2 if "Vary" in self.bool_var.get() else 1
                if param_type == "option":
                    return len([o for o in self.options_var.get().split(",") if o.strip()])
        except (tk.TclError, ValueError, AttributeError):
            pass
        return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv_values(raw: str, param_type: str) -> list:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    try:
        if param_type == "float":
            return [float(t) for t in tokens]
        if param_type == "int":
            return [int(float(t)) for t in tokens]
        if param_type == "bool":
            return [t.lower() in {"true", "1"} for t in tokens]
    except ValueError:
        pass
    return [t.strip("'\"") for t in tokens]
