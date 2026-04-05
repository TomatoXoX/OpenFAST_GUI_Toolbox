# OpenFAST GUI Toolbox - Frontend Integration Guide

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
│ │  (geometry.py)      │  │ (openfast_gui.py) │                  │
│ │  - Mass/CG/Inertia  │  │ - Discovery       │                  │
│ │  - Mooring points   │  │ - Generation      │                  │
│ └─────────────────────┘  └───────────────────┘                  │
│                                                                   │
│ ┌─────────────────────┐  ┌───────────────────┐                  │
│ │  Simulation Runner  │  │ Post-Processing   │                  │
│ │  (openfast_gui.py)  │  │ (processing/*.py) │                  │
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
    [key: 
