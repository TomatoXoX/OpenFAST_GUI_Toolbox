"""
main.py — Entry point for the OpenFAST Workflow Manager (Refactored V3).

Architecture: MVVM
  core/          — pure business logic, no UI
  services/      — thread orchestration, no UI
  viewmodels/    — Observable state + Commands, no tkinter
  views/         — Tkinter presentation, no business logic
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so that `processing/` and `geometry.py`
# (which live at the same level as this file) are importable.
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    # --- DPI awareness (Windows only) ---
    try:
        from ctypes import windll  # type: ignore
        windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError, OSError):
        pass

    import tkinter as tk
    from viewmodels.app_viewmodel import AppViewModel
    from views.main_window import MainWindow

    root = tk.Tk()
    vm = AppViewModel()
    MainWindow(root, vm)

    # Greet user via the setup log channel
    vm.post_message("setup_log", "Welcome to the OpenFAST Workflow Manager!")

    root.mainloop()


if __name__ == "__main__":
    main()
