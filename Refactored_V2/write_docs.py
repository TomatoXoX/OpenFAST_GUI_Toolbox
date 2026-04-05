cat << 'EOF' > Refactored_V2/write_docs.py
import os

guide_content = """# OpenFAST GUI Toolbox - Frontend Integration Guide

**Version:** 2.0 (Refactored)  
**Date:** 2026-04-06  
**Author:** Technical Documentation for Frontend Developers

---

## Executive Summary

This comprehensive technical analysis documents the `Refactored_V2` codebase architecture, data models, API specifications, and integration patterns for frontend developers implementing a modern UI layer.

**Codebase Statistics:**
- **Total Lines:** ~4,300 Python LOC
- **Core Modules:** 10 files (6 main + 4 processing)
- **Public API Methods:** 15+ callable functions
- **Data Models:** 12+ TypeScript interface mappings

**Key Features:**
1. **Parametric Test Case Generation** - Grid search, Latin hypercube sampling, CSV import
2. **Platform Geometry Calculator** - Mass, CG, inertia, mooring points
3. **Parallel Simulation Execution** - Multi-threaded OpenFAST runs with real-time logging
4. **Automated Post-Processing** - CSV conversion, d'Alembert staticization, plotting, frequency analysis

---

## 1. Architecture Overview

### System Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  Frontend Layer (To Be Built)                    │
│           React/Vue/Angular + TypeScript/WebSocket               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API / IPC
┌──────────────────────────▼──────────────────────────────────────┐
│               OpenFAST GUI Core (Python Backend)                 │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────┐  ┌───────────────────┐                  │
│ │  Geometry Module    │  │ Test Case Manager │                  │
│ │  (geometry.py)      │  │ - Discovery       │                  │
│ │  - Mass/CG/Inertia  │  │ - Generation      │                  │
│ │  - Mooring points   │  │                   │                  │
│ └─────────────────────┘  └───────────────────┘                  │
│                                                                   │
│ ┌─────────────────────┐  ┌───────────────────┐                  │
│ │  Simulation Runner  │  │ Post-Processing   │                  │
│ │  - Multi-threading  │  │ - CSV conversion  │                  │
│ │  - Log streaming    │  │ - d'Alembert      │                  │
│ │  - Progress track   │  │ - Plotting        │                  │
│ └─────────────────────┘  │ - Frequency       │                  │
│                           └───────────────────┘                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ File I/O
┌──────────────────────────▼──────────────────────────────────────┐
│           OpenFAST Input/Output Files (.fst, .dat, .out)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
Refactored_V2/
├── main.py (4 LOC)                    # Entry point
├── openfast_gui.py (2094 LOC)         # Main GUI controller
│   └── Class: OpenFASTTestCaseGUI
│       ├── discover_parameters()      # Scan FST files
│       ├── generate_test_cases()      # Create case directories
│       ├── run_selected_cases()       # Execute simulations
│       └── run_selected_post_proc()   # Post-process results
│
├── geometry.py (427 LOC)              # Platform geometry calculator
│   ├── Class: SemiSubmersiblePlatform
│   └── calculate_semisub_properties() # PUBLIC API
│
├── processing/
│   ├── __init__.py                    # Module exports
│   ├── converters.py (150 LOC)        # .out → .csv converter
│   │   └── Class: ConverterRunner
│   ├── dalembert.py (909 LOC)         # Load staticization
│   │   └── Class: DalembertRunner
│   ├── frequency.py (252 LOC)         # Modal frequency analysis
│   │   └── Class: FrequencyAnalysisRunner
│   └── plotting.py (469 LOC)          # Automated plot generation
│       └── Class: PlottingRunner
│
├── utils/
│   └── file_utils.py (45 LOC)         # File parsing helpers
│
└── tests/                              # pytest test suite
    ├── test_openfast_gui.py
    └── testgeo.py
```

---

## 3. TypeScript Interface Mappings

### 3.1 Platform Geometry

```typescript
// INPUT: CSV row or form data
interface PlatformGeometryInput {
  ID: string | number;
  
  // Main Column (center, vertical)
  MC_radius: number;                  // m, valid range: 0.1-50
  MC_height_above_SWL: number;        // m, above still water level
  MC_height_below_SWL: number;        // m, draft
  MC_thickness: number;               // m, wall thickness
  
  // Upper Columns (3x, outer, radial layout @ 0°, 120°, 240°)
  distance: number;                   // m, radial distance from center
  UC_radius: number;                  // m
  UC_height_above_SWL: number;        // m
  UC_height_below_SWL: number;        // m, total draft
  UC_thickness: number;               // m
  
  // Base Columns (3x, underwater pontoons)
  BC_radius: number;                  // m
  BC_height: number;                  // m
  BC_thickness: number;               // m
}

// OUTPUT: Calculated properties
interface PlatformGeometryOutput {
  total_properties_no_ballast: {
    weight: number;                   // kg
    cg: [number, number, number];     // [x, y, z] in meters
  };
  
  mooring_points: Array<{             // 3 fairlead coordinates
    x: number;  y: number;  z: number;  // m, global frame
  }>;
  
  total_inertia_about_cm: {
    roll: number;                     // kg·m² (Ixx)
    pitch: number;                    // kg·m² (Iyy)
    yaw: number;                      // kg·m² (Izz)
  };
  
  column_properties: {
    main: { radius: number; thickness: number; };
    upper: { radius: number; thickness: number; };
    base: { radius: number; thickness: number; };
  };
}
```

### 3.2 OpenFAST Parameters

```typescript
interface OpenFASTParameter {
  file_type: string;                  // e.g., "ElastoDyn.dat"
  param_name: string;                 // e.g., "PtfmMass"
  param_info: {
    line_number: number;              // Location in source file
    original_value: number | string | boolean;
    type: "float" | "int" | "bool" | "option";
    description: string;              // Inline comment
    unit: string;                     // e.g., "kg", "m/s"
  };
}

interface ParameterDiscoveryResult {
  discovered_parameters: {
    [fileKey: string]: {              // e.g., "ElastoDyn.dat"
      [paramName: string]: {          // e.g., "PtfmMass"
        line_number: number;
        original_value: any;
        type: string;
        description: string;
        unit: string;
      };
    };
  };
  file_structure: {
    [fileKey: string]: {
      path: string;
    };
  };
}
```

### 3.3 Test Case Configuration

```typescript
interface TestCaseConfig {
  base_fst_path: string;              // Path to template .fst
  output_dir: string;                 // Root output directory
  num_cases: number;                  // For sampling methods
  distribution: 
    | "grid_search"                   // Cartesian product
    | "csv_columnwise"                // Zip columns
    | "latin_hypercube"               // LHS sampling
    | "uniform"                       // Uniform random
    | "normal";                       // Normal distribution
  
  geometry_csv_path?: string;         // Optional geometry variations
  
  parameters: Array<{
    file_type: string;
    param_name: string;
    variation_config: {
      start?: number;                 // Min value
      end?: number;                   // Max value
      steps?: number;                 // Number of points
      csv_values?: string;            // "1.0, 2.0, 3.0"
      int_list?: string;              // "1,2,5,10"
      bool_choice?: string;           // "Vary (True & False)"
      options_list?: string;          // "opt1, opt2"
    };
  }>;
}

interface TestCaseMetadata {
  case_name: string;                  // e.g., "case_0042_geom_G3"
  fst_file: string;                   // Main FST filename
  geometry_id: string | number;       // Geometry ID or "base"
  parameters: {
    [key: string]: any;               // Changed parameter values
  };
  status: "pending" | "running" | "completed" | "failed";
}

interface TestCasesSummary {
  base_fst: string;
  total_cases: number;
  generation_method: string;
  timestamp: string;
  cases: TestCaseMetadata[];
}
```

### 3.4 Simulation Status

```typescript
interface SimulationStatus {
  case_name: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;                   // 0-100%
  log_tail: string[];                 // Last N lines of console output
  error_message?: string;
  start_time?: string;
  end_time?: string;
}
```

### 3.5 Post-Processing Configuration

```typescript
interface PostProcessConfig {
  cases: string[];                    // List of case directories
  operations: {
    convert_csv: boolean;
    dalembert: boolean;
    frequency: boolean;
    plot: boolean;
  };
  dalembert_options?: {
    method: "ConNFx" | "SecondaryFile" | "Geometric";
    mooring_file?: string;
  };
  plot_options?: {
    channels: string[];               // e.g., ["PtfmPitch", "TwrBsMyt"]
    format: "png" | "pdf" | "svg";
  };
}

interface DalembertLoad {
  time: number;
  Fx: number; Fy: number; Fz: number;
  Mx: number; My: number; Mz: number;
}

interface DalembertExtrema {
  max_Fx: DalembertLoad;
  min_Fx: DalembertLoad;
  max_My: DalembertLoad;
  min_My: DalembertLoad;
  // ... other channels
}

interface FrequencyAnalysisResult {
  channels: string[];
  peak_frequencies: { [channel: string]: number[] }; // Hz
  psd_data: { [channel: string]: { freq: number[], psd: number[] } };
}
```

---

## 4. Public API Methods

The Python backend exposes several core methods that the frontend will need to invoke.

### 4.1 Parameter Discovery
```python
def discover_parameters(fst_path: str) -> ParameterDiscoveryResult:
    \"\"\"
    Scans a base .fst file and all linked files to find modifiable parameters.
    \"\"\"
```

### 4.2 Geometry Calculation
```python
def calculate_semisub_properties(geom_input: dict) -> dict:
    \"\"\"
    Calculates mass, CG, inertia, and mooring points for a semi-submersible.
    \"\"\"
```

### 4.3 Test Case Generation
```python
def generate_test_cases(config: dict) -> TestCasesSummary:
    \"\"\"
    Generates OpenFAST test cases based on parameter variations and geometry.
    \"\"\"
```

### 4.4 Simulation Execution
```python
def run_selected_cases(cases: list[str], max_workers: int = 4) -> None:
    \"\"\"
    Executes OpenFAST simulations in parallel using a thread pool.
    \"\"\"
```

### 4.5 Post-Processing
```python
def run_selected_post_proc(config: dict) -> None:
    \"\"\"
    Runs selected post-processing operations on completed test cases.
    \"\"\"
```

---

## 5. Data Flow & State Management

### 5.1 Message Queue Pattern
The current architecture uses Python's `queue.Queue` for thread-safe communication between worker threads and the GUI. This is the primary integration point for any new UI layer.

```python
# Backend Queue Setup
self.log_queue = queue.Queue()
self.progress_queue = queue.Queue()

# Worker Thread
self.log_queue.put({"case": case_name, "msg": log_line})
self.progress_queue.put({"case": case_name, "progress": percent})

# Frontend Polling (or WebSocket push)
while not self.log_queue.empty():
    msg = self.log_queue.get()
    # Update UI state
```

### 5.2 State Management Schema
The frontend should maintain the following global state:
1.  **Project State:** Base FST path, output directory, discovered parameters.
2.  **Test Matrix State:** Selected parameters, variation ranges, geometry CSV.
3.  **Execution State:** List of generated cases, individual case status (pending/running/completed/failed), overall progress.
4.  **Post-Processing State:** Selected operations, selected channels, generated artifacts (plots, CSVs).

---

## 6. Integration Patterns

For a modern frontend (React/Vue/Angular), two primary integration patterns are recommended:

### 6.1 FastAPI + WebSocket (Recommended for Web/Local Server)
Wrap the Python core in a FastAPI server.
-   **REST Endpoints:** For synchronous operations (discovery, generation, geometry).
-   **WebSocket:** For real-time streaming of simulation logs and progress updates.

### 6.2 Electron + IPC (Recommended for Desktop App)
Package the Python backend as a child process of an Electron app.
-   **IPC Messages:** Send commands from the renderer to the main process, which forwards them to the Python backend via `stdin`/`stdout` or a local socket.

---

## 7. Error Handling

The frontend must handle the following common error scenarios gracefully:

| Error Scenario | Backend Behavior | Frontend Action |
| :--- | :--- | :--- |
| Invalid Base FST | `discover_parameters` raises FileNotFoundError or parsing error. | Show error dialog, prompt for valid file. |
| Invalid Geometry | `calculate_semisub_properties` returns NaN or raises ValueError. | Highlight invalid input fields, show validation message. |
| OpenFAST Crash | Worker thread catches `subprocess.CalledProcessError`. | Mark case as "failed", display `error_message` from queue. |
| Missing Executable | `run_selected_cases` fails to find `openfast` binary. | Prompt user to configure OpenFAST executable path. |
| Post-Proc Missing Data | `run_selected_post_proc` cannot find `.out` files. | Disable post-processing options for incomplete cases. |

---

## 8. Example Workflows

### Scenario 1: Basic Parameter Sweep
1.  User selects `Test18.fst`.
2.  Frontend calls `discover_parameters`.
3.  User selects `PtfmMass` and sets range 1.0e6 to 2.0e6, 5 steps.
4.  Frontend calls `generate_test_cases` with `grid_search`.
5.  Backend creates 5 directories.
6.  User clicks "Run". Frontend calls `run_selected_cases`.
7.  Frontend listens to WebSocket/Queue for progress.

### Scenario 2: Geometry Optimization
1.  User uploads `geometry_variations.csv`.
2.  Frontend parses CSV and calls `calculate_semisub_properties` for each row to preview properties.
3.  User configures test matrix using the geometry CSV.
4.  Backend generates cases, merging new geometry properties into the respective `.dat` files.

---

## 9. Performance Considerations

-   **File I/O:** Generating hundreds of test cases involves significant file I/O. The frontend should show a loading spinner during `generate_test_cases`.
-   **Memory:** Post-processing (especially d'Alembert and Frequency analysis) loads large `.out` files into memory (pandas DataFrames). The backend processes cases sequentially to limit memory usage, but the frontend should be aware of potential delays.
-   **Log Streaming:** OpenFAST generates verbose output. The frontend should throttle log updates (e.g., update UI every 100ms) to prevent UI freezing.

---

## 10. Testing & Validation

The backend includes a pytest suite (`tests/`). Frontend developers should:
1.  Run the backend tests to ensure the core logic is functioning correctly in their environment.
2.  Create mock API responses based on the TypeScript interfaces above to develop the UI independently of the backend.

---

## 11. Migration Path from Tkinter

1.  **Decouple:** The current `openfast_gui.py` tightly couples Tkinter UI code with business logic. The first step is to extract the core logic (discovery, generation, execution) into a separate, pure Python class (e.g., `OpenFASTController`).
2.  **API Layer:** Build the API layer (FastAPI or IPC) around the `OpenFASTController`.
3.  **Frontend Build:** Develop the React/Vue UI consuming the API.
4.  **Deprecate:** Remove the Tkinter code.

---

## 12. Additional Resources

-   **OpenFAST Documentation:** https://openfast.readthedocs.io/
-   **NREL GitHub:** https://github.com/OpenFAST/openfast
"""

with open("Refactored_V2/FRONTEND_INTEGRATION_GUIDE.md", "w", encoding="utf-8") as f:
    f.write(guide_content)

architecture_content = """# OpenFAST GUI Toolbox - Architecture Overview

## System Architecture

The Refactored_V2 OpenFAST GUI Toolbox is designed as a modular desktop application for managing, executing, and analyzing OpenFAST offshore wind turbine simulations. 

### Core Layers

1.  **Presentation Layer (GUI)**
    *   Currently implemented in `openfast_gui.py` using Tkinter.
    *   Responsible for user input, configuration, and displaying real-time simulation progress.
    *   *Target for future replacement with a modern web-based frontend (React/Vue).*

2.  **Application Logic Layer**
    *   Manages the lifecycle of test cases.
    *   Handles parameter discovery from OpenFAST input files (`.fst`, `.dat`).
    *   Generates test matrices using various sampling methods (Grid Search, LHS, etc.).
    *   Orchestrates parallel execution of OpenFAST binaries using Python's `concurrent.futures.ThreadPoolExecutor`.

3.  **Processing Modules Layer**
    *   `geometry.py`: Calculates physical properties (mass, CG, inertia) for semi-submersible platforms based on parametric inputs.
    *   `processing/converters.py`: Converts OpenFAST binary/text output (`.out`, `.outb`) to standard CSV format.
    *   `processing/dalembert.py`: Performs staticization of loads using d'Alembert's principle.
    *   `processing/frequency.py`: Conducts modal frequency analysis (FFT/PSD) on simulation output.
    *   `processing/plotting.py`: Automates generation of time-series and frequency plots.

4.  **Data Persistence Layer**
    *   Relies on the file system.
    *   Reads template OpenFAST input files.
    *   Writes generated test case directories, modified input files, and execution summaries (`case_info.json`, `test_cases_summary.json`).

### Data Flow

1.  **Initialization:** User selects a base `.fst` file. The system parses this file and all linked module files to build a tree of modifiable parameters.
2.  **Configuration:** User defines parameter ranges, sampling methods, and optional geometry variations.
3.  **Generation:** The system generates a matrix of test cases, creating a new directory for each case with modified input files.
4.  **Execution:** The ThreadPoolExecutor runs multiple OpenFAST instances concurrently. Standard output/error streams are captured and placed in thread-safe queues (`queue.Queue`) for the GUI to consume and display progress.
5.  **Post-Processing:** Upon completion, selected processing modules analyze the output files, generating CSVs, plots, and analysis reports.
"""

with open("Refactored_V2/ARCHITECTURE.md", "w", encoding="utf-8") as f:
    f.write(architecture_content)

print("Successfully wrote FRONTEND_INTEGRATION_GUIDE.md and ARCHITECTURE.md")
EOF
python Refactored_V2/write_docs.py
rm Refactored_V2/write_docs.py

/usr/bin/bash: line 227: warning: here-document at line 1 delimited by end-of-file (wanted `EOF')
