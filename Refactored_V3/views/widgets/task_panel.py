"""
views/widgets/task_panel.py
============================
Reusable composite widget: Treeview + action buttons + progress bar + log.

Used by both the Run and Post-Process tabs.  It knows nothing about
business logic; it exposes callbacks that the parent tab binds to
ViewModel commands.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any, Callable, Dict, Optional, Tuple


class TaskPanel(ttk.Frame):
    """
    A self-contained panel containing:
    - A :class:`ttk.Treeview` to display cases and their statuses.
    - Load / Select All / Deselect All / Run action buttons.
    - A :class:`ttk.Progressbar`.
    - A scrolled log widget.

    Parameters
    ----------
    parent:
        Parent Tkinter widget.
    title:
        LabelFrame caption.
    columns:
        Sequence of column names for the treeview (excluding the hidden #0 column).
    col_widths:
        ``{column_name: pixel_width}`` mapping.
    run_button_text:
        Label for the run/start button.
    on_load:
        Callback invoked when "Load Cases" is pressed.
    on_run:
        Callback invoked when the run button is pressed.  Receives the list
        of selected *item_ids* as its single argument.
    """

    LOG_MAX_LINES = 6_000
    LOG_PURGE_CHUNK = 800

    def __init__(
        self,
        parent: ttk.Widget,
        title: str,
        columns: Tuple[str, ...],
        col_widths: Dict[str, int],
        run_button_text: str,
        on_load: Callable[[], None],
        on_run: Callable[[list], None],
        on_context_menu: Optional[Callable[[tk.Event, ttk.Treeview], None]] = None,
    ) -> None:
        super().__init__(parent)
        self._on_run = on_run

        case_frame = ttk.LabelFrame(self, text=title, padding="10")
        case_frame.pack(fill="both", expand=True, pady=5, padx=10)

        # --- Button row ---
        btn_frame = ttk.Frame(case_frame)
        btn_frame.pack(fill="x", pady=5)
        ttk.Button(btn_frame, text="Load Cases", command=on_load).pack(side="left", padx=5)
        self.tree = ttk.Treeview(
            ttk.Frame(case_frame),  # placeholder — placed below
            columns=columns,
            show="headings",
            selectmode="extended",
        )
        ttk.Button(
            btn_frame, text="Select All",
            command=lambda: self.tree.selection_set(self.tree.get_children()),
        ).pack(side="left", padx=5)
        ttk.Button(
            btn_frame, text="Deselect All",
            command=lambda: self.tree.selection_set([]),
        ).pack(side="left", padx=5)
        self.run_button = ttk.Button(
            btn_frame,
            text=run_button_text,
            command=self._on_run_clicked,
            style="Accent.TButton",
        )
        self.run_button.pack(side="left", padx=20)

        # --- Treeview ---
        list_frame = ttk.Frame(case_frame)
        list_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("#0", text="Test Case")
        self.tree.column("#0", width=200, anchor="w")
        for col, width in col_widths.items():
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="center" if col == "Runtime" else "w")

        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        if on_context_menu:
            self.tree.bind("<Button-3>", lambda e: on_context_menu(e, self.tree))

        # --- Progress bar ---
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(case_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5, side="bottom")

        # --- Log widget ---
        log_frame = ttk.LabelFrame(case_frame, text="Execution Log", padding="5")
        log_frame.pack(fill="both", expand=True, pady=5)
        self.log_widget = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            wrap=tk.WORD,
            bg="#f0f0f0",
            relief="sunken",
            borderwidth=1,
        )
        self.log_widget.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Public API used by parent tabs
    # ------------------------------------------------------------------

    def get_selected_ids(self) -> list:
        return list(self.tree.selection())

    def populate(self, cases: Dict[str, Any]) -> None:
        """
        Clear the tree and repopulate from *cases*.

        Each value in *cases* must be a :class:`~core.models.CaseInfo`.
        """
        self.tree.delete(*self.tree.get_children())
        for item_id, case_info in cases.items():
            params = case_info.parameters or {}
            param_items = []
            if case_info.geometry_id is not None:
                param_items.append(f"geom={case_info.geometry_id}")
            param_items.extend(
                f"{k.split('/')[-1]}={v:.3g}" if isinstance(v, (int, float)) else f"{k.split('/')[-1]}={v}"
                for k, v in params.items()
            )
            params_str = ", ".join(param_items)
            self.tree.insert(
                "",
                "end",
                iid=item_id,
                text=case_info.case_name,
                values=("Ready", params_str, "-", "-"),
            )
        self.tree.selection_set(self.tree.get_children())

    def update_row(self, item_id: str, column: str, value: str) -> None:
        if self.tree.exists(item_id):
            self.tree.set(item_id, column, value)

    def set_progress(self, pct: float) -> None:
        self.progress_var.set(pct)

    def append_log(self, message: str) -> None:
        normalized = message.rstrip("\n") + "\n"
        self.log_widget.insert(tk.END, normalized)
        self.log_widget.see(tk.END)
        self._trim_log()

    def set_run_button_state(self, enabled: bool) -> None:
        self.run_button.config(state="normal" if enabled else "disabled")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_run_clicked(self) -> None:
        selected = self.get_selected_ids()
        self._on_run(selected)

    def _trim_log(self) -> None:
        try:
            count = int(self.log_widget.index("end-1c").split(".")[0])
        except tk.TclError:
            return
        if count > self.LOG_MAX_LINES:
            delete_to = max(1, count - self.LOG_MAX_LINES + self.LOG_PURGE_CHUNK)
            self.log_widget.delete("1.0", f"{delete_to}.0")
