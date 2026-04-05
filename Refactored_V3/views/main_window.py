"""
views/main_window.py
====================
Root Tkinter window and the central message-dispatch loop.

Responsibilities
----------------
* Create the root ``tk.Tk`` window with correct DPI settings.
* Build the ``ttk.Notebook`` and instantiate one tab class per tab.
* Run ``root.after(100, process_queue)`` to poll the ViewModel's message
  queue on the main thread and route messages to the correct tab.

This is the only place in the View layer that knows about all tabs.
"""
from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict

from viewmodels.app_viewmodel import AppViewModel
from views.setup_tab import SetupTab
from views.run_tab import RunTab
from views.post_proc_tab import PostProcTab
from views.tutorial_tab import TutorialTab


class MainWindow:
    """
    Creates the application window, builds all tabs, and owns the message loop.

    Parameters
    ----------
    root:
        A ``tk.Tk`` instance (caller is responsible for ``mainloop()``).
    vm:
        The :class:`AppViewModel` instance shared by all tabs.
    """

    _POLL_INTERVAL_MS = 100
    LOG_MAX_LINES = 6_000
    LOG_PURGE_CHUNK = 800

    def __init__(self, root: tk.Tk, vm: AppViewModel) -> None:
        self._root = root
        self._vm = vm

        self._configure_window()
        self._configure_style()
        self._build_notebook()
        self._start_message_loop()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _configure_window(self) -> None:
        self._root.title("OpenFAST Test Case Workflow Manager")
        self._root.geometry("1200x850")
        self._set_icon()

    def _set_icon(self) -> None:
        try:
            icon = Path(__file__).parent.parent / "logo.ico"
            if icon.exists():
                self._root.iconbitmap(str(icon))
        except Exception:
            pass

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Accent.TButton", foreground="white", background="#0078D7")
        style.map("Accent.TButton", background=[("active", "#005A9E")])

    # ------------------------------------------------------------------
    # Notebook / tabs
    # ------------------------------------------------------------------

    def _build_notebook(self) -> None:
        nb = ttk.Notebook(self._root)
        nb.pack(fill="both", expand=True, padx=5, pady=5)

        self._tutorial_tab = TutorialTab(nb)
        self._setup_tab   = SetupTab(nb, self._vm)
        self._run_tab     = RunTab(nb, self._vm)
        self._post_tab    = PostProcTab(nb, self._vm)

        nb.add(self._tutorial_tab, text="Tutorial")
        nb.add(self._setup_tab,    text="1. Setup Cases")
        nb.add(self._run_tab,      text="2. Run Simulations")
        nb.add(self._post_tab,     text="3. Post-Process Results")

        self._notebook = nb

        # Map notebook frame → tab object for fast routing
        self._tab_map = {
            str(self._setup_tab): self._setup_tab,
            str(self._run_tab):   self._run_tab,
            str(self._post_tab):  self._post_tab,
        }

    # ------------------------------------------------------------------
    # Message-queue polling loop
    # ------------------------------------------------------------------

    def _start_message_loop(self) -> None:
        self._root.after(self._POLL_INTERVAL_MS, self._process_queue)

    def _process_queue(self) -> None:
        try:
            while True:
                channel, payload = self._vm.message_queue.get_nowait()
                self._dispatch(channel, payload)
        except queue.Empty:
            pass
        finally:
            self._root.after(self._POLL_INTERVAL_MS, self._process_queue)

    def _dispatch(self, channel: str, payload: Any) -> None:
        """Route a channel+payload to the correct tab or handle it here."""

        # --- Setup tab channels ---
        if channel in ("setup_log", "discovery_complete", "config_loaded"):
            self._setup_tab.handle_message(channel, payload)

        # --- Run tab channels ---
        elif channel.startswith("run_") or channel == "enable_run_button":
            self._run_tab.handle_message(channel, payload)

        # --- Post-proc tab channels ---
        elif channel.startswith("post_proc_") or channel == "enable_post_proc_button":
            self._post_tab.handle_message(channel, payload)

        # --- Cross-tab notifications ---
        elif channel == "cases_generated":
            cases: list = payload
            self._setup_tab.handle_message(channel, payload)
            if cases and messagebox.askyesno(
                "Success",
                f"Generated {len(cases)} test cases.\nSwitch to 'Run Simulations' tab?",
            ):
                self._notebook.select(self._run_tab)
                self._run_tab._load_cases()

        # --- Dialog requests (emitted by ViewModel when a dialog is needed) ---
        elif channel == "dialog_browse_test_dir":
            import tkinter.filedialog as fd
            d = fd.askdirectory(title="Select Test Case Directory")
            if d:
                self._vm.output_dir = d

        # --- Errors / warnings ---
        elif channel == "error":
            messagebox.showerror("Error", str(payload))
        elif channel == "warn":
            messagebox.showwarning("Warning", str(payload))
