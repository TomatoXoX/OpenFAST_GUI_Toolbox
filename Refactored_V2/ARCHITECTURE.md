# OpenFAST GUI Toolbox - Technical Architecture Documentation

**Version:** 2.0 (Refactored)  
**Date:** 2026-04-06  
**Purpose:** Frontend Integration Guide

---

## Directory Structure

```
Refactored_V2/
├── main.py (4 lines)               # Entry point
├── openfast_gui.py (2094 lines)    # Main GUI controller
├── geometry.py (427 lines)         # Platform geometry calculator
├── processing/                     # Post-processing modules
│   ├── converters.py (150 lines)   # .out to .csv conversion
│   ├── dalembert.py (909 lines)    # Load staticization
│   ├── frequency.py (252 lines)    # Modal frequency analysis
│   └── plotting.py (469 lines)     # Automated plotting
├── utils/
│   └── file_utils.py (45 lines)    # File parsing utilities
└── tests/                          # pytest test suite
```

Total: ~4,300 lines of Python code

---

## Core Modules

### 1. Geometry Calculator (`geometry.py`)
**Purpose:** Calculate platform mass properties and mooring points

**Public API:**
```python
def calculate_semisub_properties(
    MC_radius, MC_height_above_SWL, MC_height_below_SWL, MC_thickness,
    distance, UC_radius, UC_height_above_SWL, UC_height_below_SWL, UC_thickness,
    BC_radius, BC_height, BC_thickness,
    braces_params=None, ballast_mass=None, print_results=True
) -> dict
```

**Returns:**
- `total_properties_no_ballast`: Mass (kg) & CG coordinates (m)
- `mooring_points`: 3 fairlead coordinates [x, y, z]
- `total_inertia_about_cm`: Roll, Pitch, Yaw inertia (kg·m²)
- `column_properties`: Radius & thickness for main/upper/base columns

---

### 2. Test Case Manager (`openfast_gui.py`)

**Key Methods:**

#### `discover_parameters()`
Recursively scans FST and all referenced files to find modifiable parameters.

**Output format:**
```python
{
  "ElastoDyn.dat": {
    "PtfmMass": {
      "line_number": 42,
      "original_value": 13917000.0,
      "type": "float",
      "description": "Platform mass (kg)",
      "unit": "kg"
    },
    # ... more parameters
  },
  # ... more files
}
```

#### `generate_test_cases()`
Creates test case directories with modified input files.

**Algorithm:**
1. Generate parameter combinations (grid/sampling/CSV)
2. Load geometry cases from CSV (optional)
3. For each (geometry, params) combination:
   - Create case directory: `case_{idx:04d}_geom_{geom_id}/`
   - Copy all base files
   - Modify ElastoDyn (mass, CG, inertia)
   - Modify MoorDyn (fairlead coordinates)
   - Modify HydroDyn (column properties)
   - Apply parameter variations
4. Write `test_cases_summary.json`

---

### 3. Simulation Runner (`openfast_gui.py`)

#### `run_selected_cases()`
Executes OpenFAST in parallel with real-time logging.

**Features:**
- Multi-threaded execution (configurable, default = CPU cores / 2)
- Real-time stdout/stderr streaming
- Progress tracking
- Error detection via keyword scanning

**Message Queue Pattern:**
```python
# Workers send updates
message_queue.put(("run_log", "Starting case_0042..."))
message_queue.put(("run_tree_update", (item_id, "Status", "Running")))
message_queue.put(("run_progress", 65.5))

# Main thread polls every 100ms
def process_queue():
    msg_type, msg_data = message_queue.get_nowait()
    # Update UI based on message type
```

---

### 4. Post-Processing Modules

#### CSV Converter (`processing/converters.py`)
- **Method:** `convert_openfast_to_csv_robust(input_file, output_file)`
- **Features:** Memory-efficient streaming, duplicate column handling, Fortran notation conversion
- **Outputs:** CSV data file + metadata file with column descriptions

#### d'Alembert Staticization (`processing/dalembert.py`)
- **Method:** `run(fst, glue_out, outdir, analysis_start_time, **kwargs)`
- **Process:** Extract quasi-static loads from dynamic simulation
- **Outputs:**
  - `loads_timeseries_staticized.csv` - All load time-series
  - `loads_extrema_after{t}s.csv` - Max force & moment cases
  - `staticized_report.txt` - Summary statistics

#### Plotting (`processing/plotting.py`)
- **Method:** `run(csv_file, output_dir, case_name, mean_start, ...)`
- **Generated Plots:**
  - Individual channels (e.g., PtfmHeave.png)
  - Group plots (Roll/Pitch/Yaw, Surge/Sway/Heave, Fairlead Tensions)
  - Vector magnitudes (TwrBsF, HydroF)

#### Frequency Analysis (`processing/frequency.py`)
- **Method:** `run(csv_file, column_name, output_dir, start_time)`
- **Requires:** SciPy
- **Algorithm:** Peak detection → logarithmic decrement → damping ratio → natural frequency
- **Outputs:** JSON results + decay plot with fitted envelope

---

## TypeScript Interface Mappings

### Platform Geometry Input
```typescript
interface PlatformGeometryInput {
  ID: string | number;
  MC_radius: number;  MC_height_above_SWL: number;  MC_height_below_SWL: number;  MC_thickness: number;
  distance: number;
  UC_radius: number;  UC_height_above_SWL: number;  UC_height_below_SWL: number;  UC_thickness: number;
  BC_radius: number;  BC_height: number;  BC_thickness: number;
}
```

### Platform Geometry Output
```typescript
interface PlatformGeometryOutput {
  total_properties_no_ballast: { weight: number; cg: [number, number, number]; };
  mooring_points: Array<{ x: number; y: number; z: number; }>;
  total_inertia_about_cm: { roll: number; pitch: number; yaw: number; };
  column_properties: {
    main: { radius: number; thickness: number; };
    upper: { radius: number; thickness: number; };
    base: { radius: number; thickness: number; };
  };
}
```

### Test Case Configuration
```typescript
interface TestCaseConfig {
  base_fst_path: string;
  output_dir: string;
  num_cases: number;
  distribution: "grid_search" | "csv_columnwise" | "latin_hypercube" | "uniform" | "normal";
  geometry_csv_path?: string;
  parameters: Array<{
    file_type: string;
    param_name: string;
    variation_config: {
      start?: number;  end?: number;  steps?: number;
      csv_values?: string;  int_list?: string;  bool_choice?: string;  options_list?: string;
    };
  }>;
}
```

### Simulation Status
```typescript
interface SimulationStatus {
  case_id: string;
  case_name: string;
  status: "Ready" | "Running" | "Completed" | "Failed";
  runtime_seconds?: number;
  result?: "Success" | string;
}
```

### D'Alembert Load Output
```typescript
interface DalembertLoad {
  Time: number;
  LoadName: string;
  Px: number;  Py: number;  Pz: number;  // Load point (m)
  Fx: number;  Fy: number;  Fz: number;  // Force (N)
  Mx: number;  My: number;  Mz: number;  // Moment (N·m)
  F_norm: number;  M_norm: number;
}
```

### Frequency Analysis Output
```typescript
interface FrequencyAnalysisResult {
  damped_period_s: number;
  damped_frequency_hz: number;
  damped_frequency_rad_s: number;
  logarithmic_decrement: number;
  damping_ratio_zeta: number;
  natural_period_s: number;
  natural_frequency_rad_s: number;
  peak_indices: number[];
  peak_times: number[];
  peak_values: number[];
}
```

---

## Data Flow

### Test Case Generation Flow
```
1. User selects base FST file
2. discover_parameters() scans all files
3. User configures parameter variations
4. (Optional) User uploads geometry CSV
5. generate_test_cases() creates directories:
   - Copies base files
   - Calculates geometry properties
   - Modifies ElastoDyn/MoorDyn/HydroDyn
   - Applies parameter variations
   - Writes case_info.json
6. Writes test_cases_summary.json
```

### Simulation Execution Flow
```
1. User selects cases from tree
2. run_selected_cases() spawns N worker threads
3. Each worker:
   - Pops case from job queue
   - Executes: subprocess.Popen([openfast_exe, fst_file])
   - Streams stdout to log via message queue
   - Updates status: "Running" → "Completed"/"Failed"
   - Records runtime
4. Progress bar updates in real-time
```

### Post-Processing Flow
```
For each case:
1. (Optional) Convert .out to .csv
2. (Optional) Run d'Alembert staticization
3. (Optional) Generate plots
4. (Optional) Run frequency analysis
All tasks log to message queue for UI updates
```

---

## Integration Recommendations

### Architecture Option A: FastAPI + WebSocket
```
Frontend (React/Vue) ←─HTTP/WS─→ FastAPI ←─→ OpenFAST Core (Pyth
