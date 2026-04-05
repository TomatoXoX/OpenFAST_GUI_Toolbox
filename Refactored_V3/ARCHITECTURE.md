# OpenFAST Workflow Manager — Architecture (Refactored V3)

## Overview

Refactored V3 decomposes the original 2 094-line monolithic `openfast_gui.py`
into a layered **MVVM** (Model–View–ViewModel) architecture.  The decomposition
enforces a strict dependency rule:

```
Views  →  ViewModels  →  Services  →  Core
  ↑             ↑             ↑           ↑
(tkinter)   (Observable)  (threads)  (pure logic)
```

Each layer may only import from the layer to its right.  No layer may import
from a layer to its left.  This means the entire `core/`, `services/`, and
`viewmodels/` stack is importable and testable **without a display server**.

---

## Directory Structure

```
Refactored_V3/
│
├── main.py                         ← Entry point (DPI + tk.Tk + mainloop)
│
├── core/                           ← Pure business logic — ZERO UI imports
│   ├── __init__.py
│   ├── models.py                   ← Data-transfer objects & domain enums
│   ├── parameter_engine.py         ← FST file scanning & parameter extraction
│   ├── case_generator.py           ← Test-case directory creation & combinations
│   ├── file_modifier.py            ← In-place parameter & geometry file edits
│   └── geometry_service.py         ← Adapter to geometry.py engine
│
├── services/                       ← Threading & pipeline orchestration
│   ├── __init__.py
│   ├── execution_service.py        ← Parallel OpenFAST simulation runner
│   └── post_processing_service.py  ← CSV → d'Alembert → Plots → Freq pipeline
│
├── viewmodels/                     ← MVVM ViewModel layer — no tkinter
│   ├── __init__.py
│   ├── observable.py               ← ObservableProperty, ObservableList, Command
│   └── app_viewmodel.py            ← Central ViewModel; owns all state & commands
│
├── views/                          ← Tkinter presentation layer only
│   ├── __init__.py
│   ├── main_window.py              ← Root window + message-dispatch loop
│   ├── tutorial_tab.py
│   ├── setup_tab.py
│   ├── run_tab.py
│   ├── post_proc_tab.py
│   └── widgets/
│       ├── __init__.py
│       ├── task_panel.py           ← Reusable Treeview + log + progress bar
│       └── parameter_row.py        ← Reusable single-parameter configuration row
│
├── processing/                     ← Unchanged from Refactored_V2
│   ├── __init__.py
│   ├── converters.py
│   ├── dalembert.py
│   ├── frequency.py
│   └── plotting.py
│
├── geometry.py                     ← Unchanged from Refactored_V2
├── utils/
│   ├── __init__.py
│   └── file_utils.py
└── logo.ico
```

---

## Architectural Decisions

### 1. MVVM Pattern

| Role        | Module(s)           | Responsibility                                          |
|-------------|---------------------|---------------------------------------------------------|
| **Model**   | `core/models.py`    | Data-transfer objects — pure dataclasses, no behaviour  |
| **ViewModel** | `viewmodels/`     | State + Commands; mediates between View and Core/Services |
| **View**    | `views/`            | Tkinter widgets; reads ViewModel state, calls Commands  |

This separation means:
* The entire business stack is testable headless (no `tk.Tk()` required).
* A future migration to PyQt6, CustomTkinter, or a web backend only requires
  rewriting `views/` and the polling bridge in `main_window.py`.
* No business logic needs to change if the UI framework is swapped.

### 2. Observable Property Binding (Push, not Poll)

`ObservableProperty` (a Python descriptor) replaces `tk.StringVar` /
`tk.BooleanVar` throughout the ViewModel.  The View subscribes to property
changes via `vm.subscribe("base_fst_path", callback)`.  When the ViewModel
updates a property, all registered callbacks fire immediately — no polling,
no coupled write-backs.

`ObservableList` extends `list` with an `on_change` callback so the View is
notified when parameter-variation rows are added or removed.

### 3. Thread-Safe Message Bus

Background worker threads (simulations, post-processing) **must not** touch
Tkinter widgets directly (not thread-safe).  The solution:

* All threads call `vm.post_message(channel, payload)` which enqueues to a
  `queue.Queue`.
* `MainWindow._process_queue()` runs on the main thread every 100 ms via
  `root.after()` and routes messages to the correct tab's `handle_message()`.
* This is the same pattern as Refactored_V2 (`process_queue`), but now
  centralised in `MainWindow` rather than scattered across the GUI class.

### 4. Command Pattern

`Command` objects encapsulate a callable and an optional `can_execute`
predicate.  Buttons bind to commands rather than calling methods directly:

```python
ttk.Button(..., command=vm.cmd_discover_parameters).pack(...)
```

`cmd.can_execute()` enables/disables the button.  The View calls
`cmd.subscribe_can_execute_changed(callback)` to react when executability
changes (e.g. after a file is selected).

### 5. Services Layer

`SimulationRunService` and `PostProcessingService` own all threading logic.
They accept a `message_bus: Callable[[str, Any], None]` — in production this
is `vm.post_message`; in tests it can be any mock callable.  This makes
multi-threaded behaviour fully testable without a running GUI.

### 6. Core Layer Purity

Every function in `core/` follows these rules:
* **No** `import tkinter`.
* Progress communicated via an **optional** `log: Callable[[str], None]`
  argument (default `None`).
* All state passed in as arguments; no global mutable state.
* Returns plain Python objects; raises standard exceptions on failure.

### 7. Single-Responsibility Modules

| Module | Single responsibility |
|--------|-----------------------|
| `core/parameter_engine.py` | Scan & parse OpenFAST input files |
| `core/case_generator.py`   | Build all (geometry × parameter) combinations |
| `core/file_modifier.py`    | Rewrite values inside input files |
| `core/geometry_service.py` | Bridge to `geometry.py` engine |
| `services/execution_service.py` | Run OpenFAST subprocesses |
| `services/post_processing_service.py` | Run the analysis pipeline |
| `views/widgets/task_panel.py` | Reusable treeview+log+progress |
| `views/widgets/parameter_row.py` | Single parameter variation row |

---

## How to swap the UI framework

1. Delete `views/` entirely.
2. Write a new `views/` package that imports your new framework.
3. Each tab class must:
   - Subscribe to `vm.*` properties for live updates.
   - Call `vm.cmd_*()` Commands on user actions.
   - Call `vm.post_message(channel, payload)` via a service or consume
     `vm.message_queue` on its own event loop.
4. `main.py` needs only minor changes: replace `tk.Tk()` with the new
   framework's application setup.
5. **Zero changes** needed in `core/`, `services/`, or `viewmodels/`.

---

## Running the Application

```bash
cd Refactored_V3
python main.py
```

Dependencies: `numpy`, `pandas`, `matplotlib` (optional), `scipy` (optional).
