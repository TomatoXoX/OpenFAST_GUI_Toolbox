import gc
import itertools
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import geometry as calc_geo
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from processing import (
    ConverterRunner,
    DalembertRunner,
    FrequencyAnalysisRunner,
    PlottingRunner,
    SCIPY_AVAILABLE,
)
from utils.file_utils import _strip_quotes


# ### NEW ###
def calculate_platform_properties(
    MC_radius: float,
    MC_height_above_SWL: float,
    MC_height_below_SWL: float,
    MC_thickness: float,
    distance: float,
    UC_radius: float,
    UC_height_above_SWL: float,
    UC_height_below_SWL: float,
    UC_thickness: float,
    BC_radius: float,
    BC_height: float,
    BC_thickness: float,
    **kwargs, # To absorb any other columns from the CSV
) -> dict:
    """
    Calculates and returns all platform properties, including mass, CG, and moments of inertia.
    This is a placeholder function. It returns dummy data in the correct format.
    """
    print(f"--- Calculating geometry for MC_radius={MC_radius}, UC_radius={UC_radius}, BC_radius={BC_radius} ---")
    platform_geometry = {
    "MC_radius": MC_radius,
    "MC_height_above_SWL": MC_height_above_SWL,
    "MC_height_below_SWL": MC_height_below_SWL,
    "MC_thickness": MC_thickness,
    "distance": distance,
    "UC_radius":UC_radius,
    "UC_height_above_SWL": UC_height_above_SWL,
    "UC_height_below_SWL": UC_height_below_SWL,
    "UC_thickness": UC_thickness,
    "BC_radius": BC_radius,
    "BC_height": BC_height,
    "BC_thickness": BC_thickness,
}
    platform_results = calc_geo.calculate_semisub_properties(**platform_geometry,
                                                              print_results=False)
    # Return dictionary in the format specified by the prompt
    structural_props_dict = platform_results['structural_properties']
    total_mass = structural_props_dict['weight']
    mooring_points_list = platform_results['mooring_points']
    structural_cg_tuple = structural_props_dict['cg']
    structural_cg_z = structural_cg_tuple[2]
    total_inertia = platform_results['total_inertia_about_cm']
    fairlead_1_x = mooring_points_list[0]['x']
    fairlead_1_y = mooring_points_list[0]['y']
    fairlead_1_z = mooring_points_list[0]['z']
    fairlead_2_x = mooring_points_list[1]['x']
    fairlead_2_y = mooring_points_list[1]['y']
    fairlead_2_z = mooring_points_list[1]['z']
    fairlead_3_x = mooring_points_list[2]['x']
    fairlead_3_y = mooring_points_list[2]['y']
    fairlead_3_z = mooring_points_list[2]['z']
    I_xx = total_inertia['roll']
    I_yy = total_inertia['pitch']
    I_zz = total_inertia['yaw']
    return {
        'total_properties_no_ballast': {
            'weight': total_mass,
            'cg': (0.0, 0.0, structural_cg_z),
        },
        'mooring_points': [
            {'x': fairlead_1_x, 'y': fairlead_1_y, 'z': fairlead_1_z},
            {'x': fairlead_2_x, 'y': fairlead_2_y, 'z': fairlead_2_z},
            {'x': fairlead_3_x, 'y': fairlead_3_y, 'z': fairlead_3_z},
        ],
        'total_inertia_about_cm': {
            'roll': I_xx,
            'pitch': I_yy,
            'yaw': I_zz,
        },
        # ### NEW ### Add column properties for HydroDyn
        'column_properties': {
            'main': {'radius': MC_radius, 'thickness': MC_thickness},
            'upper': {'radius': UC_radius, 'thickness': UC_thickness},
            'base': {'radius': BC_radius, 'thickness': BC_thickness},
        }
    }


class OpenFASTTestCaseGUI:
    """
    GUI for managing OpenFAST test case generation, execution, and post-processing.
    """

    TUTORIAL_TAB_NAME = "Tutorial"
    SETUP_TAB_NAME = "1. Setup Cases"
    RUN_TAB_NAME = "2. Run Simulations"
    POST_PROC_TAB_NAME = "3. Post-Process Results"
    DEFAULT_ANALYSIS_START_TIME = 300.0
    SAMPLING_DISTRIBUTIONS = {"latin_hypercube", "uniform", "normal"}

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OpenFAST Test Case Workflow Manager")
        self.root.geometry("1200x850")
        self._set_app_icon()

        self._setup_style()
        self._init_vars()
        self._create_notebook_and_tabs()

        self.process_queue()
        self.log("Welcome to the OpenFAST Workflow Manager!")

    def _set_app_icon(self) -> None:
        try:
            icon_path = Path(__file__).parent / "logo.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception as exc:  # pragma: no cover - GUI integration
            print(f"Could not load custom icon: {exc}")

    def _setup_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Accent.TButton", foreground="white", background="#0078D7")
        style.map("Accent.TButton", background=[("active", "#005A9E")])

    def _init_vars(self) -> None:
        self.base_fst_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "test_cases"))
        self.openfast_exe = tk.StringVar()

        self.discovered_parameters: Dict[str, Dict[str, Any]] = {}
        self.file_structure: Dict[str, Dict[str, Any]] = {}
        self.parameter_entries: List[Dict[str, Any]] = []
        self.num_cases = tk.IntVar(value=10)

        # ### NEW ### Variables for geometry import
        self.geometry_csv_path = tk.StringVar()
        self.geometry_cases: List[Dict] = []

        self.message_queue: "queue.Queue[Any]" = queue.Queue()
        self.num_threads = tk.IntVar(value=max(1, (os.cpu_count() or 2) // 2))

        self.task_data = {
            "run": {
                "cases": {},
                "job_queue": queue.Queue(),
                "progress_lock": threading.Lock(),
                "completed": 0,
                "total": 0,
            },
            "post_proc": {
                "cases": {},
                "job_queue": queue.Queue(),
                "progress_lock": threading.Lock(),
                "completed": 0,
                "total": 0,
            },
        }
        self.plotting_lock = threading.Lock()

        self.run_convert_csv = tk.BooleanVar(value=True)
        self.run_dalembert = tk.BooleanVar(value=True)
        self.run_plotting = tk.BooleanVar(value=True)
        self.run_frequency_analysis = tk.BooleanVar(value=False)
        self.frequency_analysis_column = tk.StringVar(value="PtfmHeave")

    def _create_notebook_and_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        tab_creators = {
            self.TUTORIAL_TAB_NAME: self.create_tutorial_tab,
            self.SETUP_TAB_NAME: self.create_setup_tab,
            self.RUN_TAB_NAME: self.create_run_tab,
            self.POST_PROC_TAB_NAME: self.create_post_proc_tab,
        }

        self.tabs: Dict[str, ttk.Frame] = {}
        for name, creator_func in tab_creators.items():
            frame = ttk.Frame(self.notebook)
            self.tabs[name] = frame
            self.notebook.add(frame, text=name)
            creator_func(frame)

    # -------------------------------------------------------------------------
    # TAB BUILDERS
    # -------------------------------------------------------------------------
    def create_setup_tab(self, parent_frame: ttk.Frame) -> None:
        main_frame = ttk.Frame(parent_frame)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind(
            "<Configure>", lambda event: canvas.itemconfig(canvas_window, width=event.width)
        )

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.create_file_selection_section(scrollable_frame)
        self.create_test_config_section(scrollable_frame)
        # ### NEW ### Add geometry import section
        self.create_geometry_section(scrollable_frame)
        self.create_parameter_discovery_section(scrollable_frame)
        self.create_parameter_section(scrollable_frame)
        self.create_action_section(scrollable_frame)

        log_frame = ttk.LabelFrame(parent_frame, text="Output Log", padding="10")
        log_frame.pack(fill="x", side="bottom", pady=5, padx=5)
        self.setup_log = scrolledtext.ScrolledText(
            log_frame,
            height=6,
            wrap=tk.WORD,
            bg="#f0f0f0",
            relief="sunken",
            borderwidth=1,
        )
        self.setup_log.pack(fill="both", expand=False)

    def create_run_tab(self, parent_frame: ttk.Frame) -> None:
        config_frame = ttk.LabelFrame(parent_frame, text="Run Configuration", padding="10")
        config_frame.pack(fill="x", pady=5, padx=10)

        ttk.Label(config_frame, text="OpenFAST Path:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(config_frame, textvariable=self.openfast_exe, width=50).grid(
            row=0, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Button(config_frame, text="Browse", command=self.browse_openfast_exe).grid(
            row=0, column=2, padx=5, pady=2
        )

        ttk.Label(config_frame, text="Parallel runs:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(
            config_frame,
            from_=1,
            to=os.cpu_count() or 8,
            textvariable=self.num_threads,
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        config_frame.columnconfigure(1, weight=1)

        widgets = self._create_task_tab_layout(
            parent=parent_frame,
            task_key="run",
            title="Test Cases to Run",
            columns=("Status", "Parameters", "Runtime", "Result"),
            col_widths={
                "Status": 180,
                "Parameters": 300,
                "Runtime": 100,
                "Result": 200,
            },
            load_cmd=self.load_run_cases,
            run_cmd=self.run_selected_cases,
            run_button_text="Run Selected Simulations",
        )
        self.run_widgets = widgets

    def create_post_proc_tab(self, parent_frame: ttk.Frame) -> None:
        top_frame = ttk.Frame(parent_frame)
        top_frame.pack(fill="x", pady=5, padx=10)

        config_frame = ttk.LabelFrame(top_frame, text="Configuration", padding="10")
        config_frame.pack(fill="x", expand=True, side="left", padx=(0, 5))
        ttk.Label(config_frame, text="Results Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(config_frame, textvariable=self.output_dir, width=50).grid(
            row=0, column=1, sticky="ew", padx=5, pady=2
        )
        ttk.Button(config_frame, text="Browse", command=self.browse_output_dir).grid(
            row=0, column=2, padx=5, pady=2
        )
        config_frame.columnconfigure(1, weight=1)

        tasks_frame = ttk.LabelFrame(top_frame, text="Tasks to Run", padding="10")
        tasks_frame.pack(fill="x", side="left", padx=5)
        ttk.Checkbutton(tasks_frame, text="Convert .out to .csv", variable=self.run_convert_csv).pack(anchor="w")
        ttk.Checkbutton(tasks_frame, text="Run d'Alembert Analysis", variable=self.run_dalembert).pack(anchor="w")
        ttk.Checkbutton(tasks_frame, text="Generate Plots", variable=self.run_plotting).pack(anchor="w")

        freq_frame = ttk.Frame(tasks_frame)
        freq_frame.pack(anchor="w", fill="x", pady=(5, 0))
        freq_check = ttk.Checkbutton(
            freq_frame,
            text="Run Frequency Analysis on column:",
            variable=self.run_frequency_analysis,
        )
        freq_check.pack(side="left")
        freq_entry = ttk.Entry(freq_frame, textvariable=self.frequency_analysis_column, width=18)
        freq_entry.pack(side="left", padx=5)
        if not SCIPY_AVAILABLE:
            freq_check.config(state="disabled")
            freq_entry.config(state="disabled")
            ttk.Label(
                tasks_frame,
                text="(Frequency Analysis requires 'scipy')",
                foreground="gray",
                font=("TkDefaultFont", 8),
            ).pack(anchor="w")

        widgets = self._create_task_tab_layout(
            parent=parent_frame,
            task_key="post_proc",
            title="Cases to Process",
            columns=("Status", "Parameters", "Result"),
            col_widths={"Status": 120, "Parameters": 400, "Result": 200},
            load_cmd=self.load_post_proc_cases,
            run_cmd=self.run_selected_post_proc,
            run_button_text="Run Post-Processing",
        )
        self.post_proc_widgets = widgets

    # -------------------------------------------------------------------------
    # COMMON TAB HELPER
    # -------------------------------------------------------------------------
    def _create_task_tab_layout(
        self,
        parent: ttk.Frame,
        task_key: str,
        title: str,
        columns: tuple,
        col_widths: Dict[str, int],
        load_cmd,
        run_cmd,
        run_button_text: str,
    ) -> Dict[str, Any]:
        case_frame = ttk.LabelFrame(parent, text=title, padding="10")
        case_frame.pack(fill="both", expand=True, pady=5, padx=10)

        btn_frame = ttk.Frame(case_frame)
        btn_frame.pack(fill="x", pady=5)

        list_frame = ttk.Frame(case_frame)
        list_frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="extended")

        ttk.Button(btn_frame, text="Load Cases", command=load_cmd).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Select All", command=lambda: tree.selection_set(tree.get_children())).pack(
            side="left", padx=5
        )
        ttk.Button(btn_frame, text="Deselect All", command=lambda: tree.selection_set([])).pack(side="left", padx=5)
        run_button = ttk.Button(btn_frame, text=run_button_text, command=run_cmd, style="Accent.TButton")
        run_button.pack(side="left", padx=20)

        tree.heading("#0", text="Test Case")
        tree.column("#0", width=200, anchor="w") # ### MODIFIED ### Increased width for new name
        for col, width in col_widths.items():
            tree.heading(col, text=col)
            tree.column(col, width=width, anchor="center" if col == "Runtime" else "w")

        tree_scroll_y = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree_scroll_x = ttk.Scrollbar(list_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        tree.bind(
            "<Button-3>",
            lambda event: self.show_case_context_menu(event, tree, self.task_data[task_key]["cases"]),
        )

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(case_frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill="x", pady=5, side="bottom")

        log_widget = self.create_log_section(case_frame, f"{task_key}_log", "Execution Log")

        return {
            "tree": tree,
            "run_button": run_button,
            "progress_bar": progress_bar,
            "progress_var": progress_var,
            "log": log_widget,
        }

    # -------------------------------------------------------------------------
    # SETUP TAB SECTIONS
    # -------------------------------------------------------------------------
    def create_file_selection_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        frame.pack(fill="x", pady=5, padx=5)

        ttk.Label(frame, text="Base FST File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame, textvariable=self.base_fst_path, width=60).grid(
            row=0,
            column=1,
            padx=5,
            sticky=tk.EW,
        )
        ttk.Button(frame, text="Browse", command=self.browse_fst_file).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.output_dir, width=60).grid(
            row=1,
            column=1,
            padx=5,
            pady=5,
            sticky=tk.EW,
        )
        ttk.Button(frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)

        frame.columnconfigure(1, weight=1)

    def create_test_config_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Test Configuration", padding="10")
        frame.pack(fill="x", pady=5, padx=5)

        ttk.Label(frame, text="Number of Test Cases:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_cases_spinbox = ttk.Spinbox(
            frame,
            from_=2,
            to=10000,
            textvariable=self.num_cases,
            width=10,
        )
        self.num_cases_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(frame, text="Parameter Distribution:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.distribution_var = tk.StringVar(value="grid_search")
        dist_combo = ttk.Combobox(
            frame,
            textvariable=self.distribution_var,
            values=["grid_search", "csv_columnwise", "latin_hypercube", "uniform", "normal"],
            width=20,
            state="readonly",
        )
        dist_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        dist_combo.bind("<<ComboboxSelected>>", self.on_distribution_change)

        self.dist_help_label = ttk.Label(
            frame,
            text="Controls how standard parameters are varied.",
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        self.dist_help_label.grid(row=1, column=2, sticky="w", padx=10)

        ttk.Button(frame, text="Refresh", command=self.refresh_distribution_settings).grid(
            row=1, column=3, padx=5, pady=5
        )

        frame.columnconfigure(2, weight=1)

    # ### NEW ### Section for importing geometry CSV
    def create_geometry_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Geometry Import (Optional)", padding="10")
        frame.pack(fill="x", pady=5, padx=5)

        ttk.Label(frame, text="Geometry CSV File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame, textvariable=self.geometry_csv_path, width=60, state="readonly").grid(
            row=0, column=1, padx=5, sticky=tk.EW
        )
        ttk.Button(frame, text="Browse & Import", command=self.browse_and_load_geometry_csv).grid(
            row=0, column=2, padx=5
        )

        self.geometry_status_label = ttk.Label(frame, text="No geometry file loaded.", foreground="gray")
        self.geometry_status_label.grid(row=1, column=1, sticky="w", padx=5, pady=(2, 0))

        frame.columnconfigure(1, weight=1)

    def create_parameter_discovery_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Parameter Discovery", padding="10")
        frame.pack(fill="x", pady=5, padx=5)
        ttk.Button(
            frame,
            text="Discover Parameters",
            command=self.discover_parameters,
            style="Accent.TButton",
        ).pack(side="left", padx=5)
        self.discovery_status = ttk.Label(
            frame,
            text="Select a .fst file and click 'Discover Parameters'",
        )
        self.discovery_status.pack(side="left", padx=20)

    def create_parameter_section(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Parameter Configuration", padding="10")
        frame.pack(fill="x", pady=5, padx=5)

        control_frame = ttk.Frame(frame)
        control_frame.pack(fill="x", pady=5)
        ttk.Button(control_frame, text="Add from Discovery", command=self.show_parameter_selector).pack(
            side="left", padx=5
        )
        ttk.Button(control_frame, text="Clear All", command=self.clear_parameters).pack(side="left", padx=5)

        scroll_container = ttk.Frame(frame, height=250)
        scroll_container.pack(fill="x", pady=5)
        scroll_container.pack_propagate(False)

        canvas = tk.Canvas(scroll_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        self.param_list_frame = ttk.Frame(canvas)

        self.param_list_frame.bind(
            "<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.param_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.bind_all(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"),
        )

    def create_action_section(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, padding="5")
        frame.pack(fill="x", pady=10)
        ttk.Button(
            frame,
            text="Generate Test Cases",
            command=self.generate_test_cases,
            style="Accent.TButton",
        ).pack(side="left", padx=5)
        ttk.Button(frame, text="Load Configuration", command=self.load_config).pack(side="left", padx=5)
        ttk.Button(frame, text="Save Configuration", command=self.save_config).pack(side="left", padx=5)
        ttk.Button(frame, text="View File Structure", command=self.show_file_structure).pack(side="left", padx=5)

    def create_log_section(self, parent: ttk.Frame, log_attr_name: str, title: str = "Output Log"):
        frame = ttk.LabelFrame(parent, text=title, padding="10")
        frame.pack(fill="both", expand=True, pady=5)
        log_widget = scrolledtext.ScrolledText(
            frame,
            height=8,
            wrap=tk.WORD,
            bg="#f0f0f0",
            relief="sunken",
            borderwidth=1,
        )
        log_widget.pack(fill="both", expand=True)
        setattr(self, log_attr_name, log_widget)
        return log_widget

    # -------------------------------------------------------------------------
    # ### NEW ### GEOMETRY IMPORT AND FILE MODIFICATION
    # -------------------------------------------------------------------------
    def browse_and_load_geometry_csv(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Geometry CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filename:
            return

        try:
            self.geometry_csv_path.set(filename)
            df = pd.read_csv(filename)

            # Validate required columns based on geometry.py docstring
            required_cols = {
                "ID", "MC_radius", "MC_height_above_SWL", "MC_height_below_SWL",
                "MC_thickness", "distance", "UC_radius", "UC_height_above_SWL",
                "UC_height_below_SWL", "UC_thickness", "BC_radius", "BC_height", "BC_thickness"
            }
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                messagebox.showerror(
                    "CSV Error",
                    f"The selected CSV is missing required columns: {', '.join(missing_cols)}"
                )
                self.geometry_csv_path.set("")
                return

            self.geometry_cases = df.to_dict('records')
            num_geoms = len(self.geometry_cases)
            self.geometry_status_label.config(text=f"Loaded {num_geoms} geometry case(s).")
            self.log(f"Successfully loaded {num_geoms} geometry cases from {Path(filename).name}")

        except Exception as exc:
            self.log(f"Error loading geometry CSV: {exc}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to load or parse geometry CSV: {exc}")
            self.geometry_csv_path.set("")
            self.geometry_cases = []
            self.geometry_status_label.config(text="Failed to load geometry file.")

    def _apply_geometry_modifications(self, case_dir: Path, geo_case_data: Dict[str, Any]) -> bool:
        """Calls the geometry script and applies modifications to OpenFAST files."""
        try:
            self.log(f"  Calculating platform properties for geometry ID: {geo_case_data.get('ID')}")
            # This is where the external script is called.
            platform_results = calculate_platform_properties(**geo_case_data)

            # 1. Modify ElastoDyn.dat
            self._modify_elastodyn_dat(case_dir, platform_results)

            # 2. Modify MoorDyn.dat
            self._modify_moordyn_dat(case_dir, platform_results)

            # 3. Modify HydroDyn.dat
            self._modify_hydrodyn_dat(case_dir, platform_results)

            return True

        except Exception as exc:
            self.log(f"  ERROR applying geometry modifications for geom ID {geo_case_data.get('ID')}: {exc}\n{traceback.format_exc()}")
            return False

    def _modify_elastodyn_dat(self, case_dir: Path, platform_results: Dict) -> None:
        """Modifies the ElastoDyn.dat file with new platform properties."""
        self.log("    Modifying ElastoDyn.dat...")
        total_props = platform_results['total_properties_no_ballast']
        total_inertia = platform_results['total_inertia_about_cm']

        params_to_set = {
            "PtfmMass": total_props['weight'],
            "PtfmCMzt": total_props['cg'][2],
            "PtfmRIner": total_inertia['roll'],
            "PtfmPIner": total_inertia['pitch'],
            "PtfmYIner": total_inertia['yaw'],
        }
        
        for param_name, value in params_to_set.items():
            # We need to find the file key for ElastoDyn.dat
            elastodyn_key = None
            for key, info in self.file_structure.items():
                if "elastodyn" in info["path"].name.lower():
                    elastodyn_key = key
                    break
            
            if elastodyn_key and param_name in self.discovered_parameters.get(elastodyn_key, {}):
                param_info = self.discovered_parameters[elastodyn_key][param_name]
                self.modify_parameter_in_file(case_dir, elastodyn_key, param_name, value, param_info)
                self.log(f" replaced {param_name}' in discovered ElastoDyn parameters to {value}.")
            else:
                self.log(f"    WARNING: Could not find '{param_name}' in discovered ElastoDyn parameters. Modification skipped.")

    def _modify_moordyn_dat(self, case_dir: Path, platform_results: Dict) -> None:
        """Modifies the POINTS table in the MoorDyn.dat file."""
        self.log("    Modifying MoorDyn.dat...")
        moordyn_path = None
        for key, info in self.file_structure.items():
            if "moordyn" in info["path"].name.lower():
                moordyn_path = case_dir / info["path"].name
                break
        
        if not moordyn_path or not moordyn_path.exists():
            self.log("    WARNING: MoorDyn.dat file not found. Skipping modification.")
            return

        fairleads = platform_results['mooring_points']
        if len(fairleads) != 3:
            self.log(f"    WARNING: Expected 3 mooring points, but geometry script returned {len(fairleads)}. Skipping MoorDyn modification.")
            return

        lines = moordyn_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        new_lines = []
        in_points_section = False
        header_passed = False
        
        for line in lines:
            if "POINTS" in line.upper():
                in_points_section = True
                new_lines.append(line)
                continue
            
            if not in_points_section:
                new_lines.append(line)
                continue

            # Detect end of the table
            if line.strip().startswith(("---", "LINES")):
                in_points_section = False
                new_lines.append(line)
                continue

            # Skip header lines within the section
            if not header_passed and ("ID" in line or "(-)" in line):
                 new_lines.append(line)
                 continue
            header_passed = True

            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                new_lines.append(line)
                continue

            point_id = int(parts[0])
            if 4 <= point_id <= 6:
                fairlead_data = fairleads[point_id - 4]
                # ID  Attachment    X        Y        Z        M     V     CdA   CA
                # (-)   (-)        (m)      (m)      (m)      (kg) (m^3) (m^2)  (-)
                new_line = (
                    f"{point_id:<5d} {'Vessel':<10s} {fairlead_data['x']:<12.5f} {fairlead_data['y']:<12.5f} "
                    f"{fairlead_data['z']:<12.5f} {parts[5]:<8s} {parts[6]:<8s} {parts[7]:<8s} {parts[8]:<5s}"
                )
                new_lines.append(new_line)
                self.log(f"      Updated MoorDyn POINT ID {point_id} to ({fairlead_data['x']:.2f}, {fairlead_data['y']:.2f}, {fairlead_data['z']:.2f})")
            else:
                new_lines.append(line)
        
        moordyn_path.write_text("\n".join(new_lines), encoding='utf-8')

    def _modify_hydrodyn_dat(self, case_dir: Path, platform_results: Dict) -> None:
        """Modifies the CYLINDRICAL MEMBER table in HydroDyn.dat."""
        self.log("    Modifying HydroDyn.dat...")
        hydrodyn_path = None
        for key, info in self.file_structure.items():
            if "hydrodyn" in info["path"].name.lower():
                hydrodyn_path = case_dir / info["path"].name
                break

        if not hydrodyn_path or not hydrodyn_path.exists():
            self.log("    WARNING: HydroDyn.dat file not found. Skipping modification.")
            return

        col_props = platform_results.get('column_properties')
        if not col_props:
            self.log("    WARNING: 'column_properties' not found in geometry results. Skipping HydroDyn modification.")
            return

        lines = hydrodyn_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        new_lines = []
        in_cyl_section = False
        header_passed = False

        for line in lines:
            if "CYLINDRICAL MEMBER CROSS-SECTION PROPERTIES" in line.upper():
                in_cyl_section = True
                new_lines.append(line)
                continue

            if not in_cyl_section:
                new_lines.append(line)
                continue

            if line.strip().startswith(("---", "RECTANGULAR")):
                in_cyl_section = False
                new_lines.append(line)
                continue
            
            if not header_passed and ("PropSetID" in line or "(-)" in line):
                 new_lines.append(line)
                 continue
            header_passed = True

            parts = line.strip().split()
            comment = " ".join(re.findall(r"!\s*(.*)", line)).lower()

            prop_to_update = None
            if "main column" in comment:
                prop_to_update = col_props.get('main')
            elif "upper column" in comment:
                prop_to_update = col_props.get('upper')
            elif "base column" in comment:
                prop_to_update = col_props.get('base')

            if prop_to_update and len(parts) >= 3:
                # PropSetID   PropD     PropThck
                #   (-)        (m)        (m)
                new_d = prop_to_update['radius'] * 2.0  # PropD is diameter
                new_thck = prop_to_update['thickness']
                original_line_start = line.split('!')[0]
                comment_part = f"! {comment.title()}" if comment else ""
                
                new_line = f"{parts[0]:<12s} {new_d:<10.5f} {new_thck:<12.5f} {comment_part}"
                new_lines.append(new_line)
                self.log(f"      Updated HydroDyn member '{comment.title()}' to D={new_d:.3f}, Thck={new_thck:.4f}")
            else:
                new_lines.append(line)

        hydrodyn_path.write_text("\n".join(new_lines), encoding='utf-8')

    # -------------------------------------------------------------------------
    # PARAMETER DISCOVERY & CASE GENERATION
    # -------------------------------------------------------------------------
    def discover_parameters(self) -> None:
        if not self.base_fst_path.get():
            messagebox.showerror("Error", "Please select a base FST file first")
            return

        self.log("Starting deep parameter discovery...")
        self.discovery_status.config(text="Scanning all referenced files...")
        self.root.update()

        file_info_by_path: Dict[Path, Dict[str, Any]] = {}
        processed_paths: set = set()

        try:
            self._discover_and_parse_files_recursively(
                Path(self.base_fst_path.get()),
                file_info_by_path,
                processed_paths,
            )

            self.file_structure = {}
            self.discovered_parameters = {}
            final_keys: set = set()

            for path, info in file_info_by_path.items():
                key = info["key"]
                if key in final_keys:
                    suffix = sum(1 for k in final_keys if k.startswith(path.stem)) + 1
                    key = f"{path.stem}_{suffix}{path.suffix}"
                final_keys.add(key)

                self.file_structure[key] = {"path": path, "original_strings": info["original_strings"]}
                if info["params"]:
                    self.discovered_parameters[key] = info["params"]

            total_params = sum(len(p) for p in self.discovered_parameters.values())
            self.discovery_status.config(
                text=f"Discovered {total_params} parameters across {len(self.file_structure)} files."
            )
            self.log(f"Discovery complete: Found {len(self.file_structure)} total files.")

        except Exception as exc:
            self.log(f"Error during parameter discovery: {exc}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to discover parameters: {exc}")

    def _discover_and_parse_files_recursively(
        self,
        file_path: Path,
        file_info_by_path: Dict[Path, Dict[str, Any]],
        processed_paths: set,
    ) -> None:
        if not file_path or not file_path.exists() or file_path in processed_paths:
            return

        self.log(f"  Scanning: {file_path.name}")
        processed_paths.add(file_path)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            if file_path not in file_info_by_path:
                file_info_by_path[file_path] = {"key": file_path.name, "original_strings": set(), "params": {}}

            params = self.extract_parameters_from_file(content.splitlines())
            if params:
                file_info_by_path[file_path]["params"] = params

            pattern = re.compile(r'(["\'])((?:[a-zA-Z]:)?[a-zA-Z0-9_.\-\s\\/]+)\1')

            for match in pattern.finditer(content):
                path_inside = match.group(2)
                if not path_inside or path_inside.lower() in {"default", "unused", "none"}:
                    continue

                resolved_path = (file_path.parent / path_inside).resolve()

                if resolved_path.is_file():
                    if resolved_path not in processed_paths:
                        self._discover_and_parse_files_recursively(
                            resolved_path,
                            file_info_by_path,
                            processed_paths,
                        )
                else:
                    parent_dir = resolved_path.parent
                    root_name = resolved_path.name
                    if parent_dir.is_dir():
                        for item in parent_dir.glob(f"{root_name}.*"):
                            if item.is_file() and item not in processed_paths:
                                self.log(f"  [Discovery] Found root name family member: {item.name}")
                                self._discover_and_parse_files_recursively(
                                    item,
                                    file_info_by_path,
                                    processed_paths,
                                )

        except Exception as exc:
            self.log(f"Could not process file {file_path.name}: {exc}")

    def extract_parameters_from_file(self, lines: List[str]) -> Dict[str, Dict[str, Any]]:
        parameters: Dict[str, Dict[str, Any]] = {}
        param_pattern = re.compile(r"^\s*([^\s!#]+)\s+([a-zA-Z_][a-zA-Z0-9_()]*)", re.IGNORECASE)

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith(("!", "#"))
                or all(char in "-=_ " for char in stripped)
            ):
                continue

            match = param_pattern.match(stripped)
            if not match:
                continue

            value_str, param_name = match.groups()
            if param_name.lower() in {"true", "false", "default", "unused", "none", "end"}:
                continue
            if any(ext in value_str.lower() for ext in [".dat", ".txt", ".csv", ".twr", ".bld", ".fst"]):
                continue

            try:
                param_info = self.parse_parameter_value(value_str, line)
                if param_info:
                    comment_match = re.search(r"[-!]\s*(.+)$", line)
                    description = comment_match.group(1).strip() if comment_match else ""
                    parameters[param_name] = {
                        "line_number": idx,
                        "original_value": param_info["value"],
                        "type": param_info["type"],
                        "description": description,
                        "unit": self.extract_unit(line),
                    }
            except Exception:
                continue
        return parameters

    @staticmethod
    def extract_unit(line: str) -> str:
        matches = re.findall(r"\(([^)]+)\)", line)
        for match in matches:
            if len(match) < 10 and not any(word in match.lower() for word in ["flag", "switch", "see"]):
                return match
        return ""

    @staticmethod
    def parse_parameter_value(value_str: str, description: str) -> Optional[Dict[str, Any]]:
        value_str = value_str.strip().strip('"\'')
        if value_str.upper() == "DEFAULT":
            return None
        try:
            value = float(value_str)
            keywords = ["switch", "flag", "mode", "method", "order", "num", "index"]
            if any(keyword in description.lower() for keyword in keywords):
                if value == int(value) and "." not in value_str and "e" not in value_str.lower():
                    return {"value": int(value), "type": "int"}
            return {"value": value, "type": "float"}
        except ValueError:
            if value_str.lower() in {"true", "false"}:
                return {"value": value_str.lower() == "true", "type": "bool"}
            if any(keyword in description.lower() for keyword in ["option", "name", "file", "type"]):
                return {"value": value_str, "type": "option"}
        return None

    # ### MODIFIED ### Major rewrite to handle geometry cases
    def generate_test_cases(self) -> None:
        if not self.base_fst_path.get() or not self.file_structure:
            messagebox.showerror("Error", "Please select a base FST file and run 'Discover Parameters' first.")
            return
        if not self.parameter_entries and not self.geometry_cases:
            messagebox.showerror("Error", "Please add at least one parameter to vary or import a geometry CSV.")
            return

        self.setup_log.delete(1.0, tk.END)
        self.log("Starting test case generation...")

        try:
            output_path = Path(self.output_dir.get())
            if output_path.exists() and any(output_path.iterdir()):
                if not messagebox.askyesno("Warning", f"Output directory '{output_path}' is not empty. Overwrite?"):
                    return
            shutil.rmtree(output_path, ignore_errors=True)
            output_path.mkdir(parents=True, exist_ok=True)

            # --- 1. Generate Standard Parameter Combinations ---
            standard_param_combinations = []
            dist_type = self.distribution_var.get()

            if self.parameter_entries:
                if dist_type == "grid_search":
                    # (logic is the same as original)
                    param_values_list = []
                    for entry in self.parameter_entries:
                        param_type = entry["param_info"]["type"]
                        values: List[Any] = []
                        if param_type == "float":
                            start, end, steps = entry["start_var"].get(), entry["end_var"].get(), entry["steps_var"].get()
                            values = np.linspace(start, end, steps).tolist() if steps > 1 else [start]
                        elif param_type == "int":
                            if entry["int_mode_var"].get() == "Range":
                                start, end, steps = entry["start_var"].get(), entry["end_var"].get(), entry["steps_var"].get()
                                values = np.round(np.linspace(start, end, steps)).astype(int).tolist() if steps > 1 else [int(round(start))]
                            else:
                                values = [int(i.strip()) for i in entry["list_var"].get().split(",") if i.strip()]
                        elif param_type == "bool":
                            values = [True, False] if "Vary" in entry["bool_var"].get() else [entry["bool_var"].get() == "True"]
                        elif param_type == "option":
                            values = [opt.strip().strip('"\'') for opt in entry["options_var"].get().split(",") if opt.strip()]
                        param_values_list.append(values if values else [entry["param_info"]["original_value"]])
                    if param_values_list:
                        standard_param_combinations = list(itertools.product(*param_values_list))

                elif dist_type == "csv_columnwise":
                    # (logic is the same as original)
                    all_lists = []
                    for entry in self.parameter_entries:
                        str_values = [item.strip() for item in entry["csv_var"].get().split(",") if item.strip()]
                        try:
                            if entry["param_info"]["type"] == "float": typed_values = [float(v) for v in str_values]
                            elif entry["param_info"]["type"] == "int": typed_values = [int(float(v)) for v in str_values]
                            elif entry["param_info"]["type"] == "bool": typed_values = [v.lower() in {"true", "1"} for v in str_values]
                            else: typed_values = [v.strip('"\'') for v in str_values]
                            all_lists.append(typed_values)
                        except ValueError as exc:
                            messagebox.showerror("Input Error", f"Invalid CSV value for '{entry['param_name']}': {exc}"); return
                    if all_lists and all_lists[0] and all(len(lst) == len(all_lists[0]) for lst in all_lists):
                        standard_param_combinations = list(zip(*all_lists))
                    elif all_lists:
                        messagebox.showerror("Input Error", "All CSV inputs must have the same number of values."); return
                
                else: # Sampling distributions
                    # (logic is the same as original)
                    num_samples = self.num_cases.get()
                    numeric_params = [p for p in self.parameter_entries if p["param_info"]["type"] in {"float", "int"}]
                    if not numeric_params: messagebox.showerror("Error", "Sampling distributions require numeric parameters."); return
                    try:
                        from scipy.stats import qmc
                        sample = qmc.LatinHypercube(d=len(numeric_params)).sample(n=num_samples)
                    except ImportError:
                        self.log("Warning: 'scipy' not found. Falling back to uniform random."); sample = np.random.rand(num_samples, len(numeric_params))
                    param_values = [entry["start_var"].get() + (entry["end_var"].get() - entry["start_var"].get()) * sample[:, idx] for idx, entry in enumerate(numeric_params)]
                    standard_param_combinations = list(zip(*param_values))

            # If no parameter variations are defined, create a single dummy run
            if not standard_param_combinations:
                standard_param_combinations = [()] # A single empty tuple for the loop

            # --- 2. Get Geometry Cases to Run ---
            # If no geometry CSV is loaded, use a single dummy geometry case
            geometry_cases_to_run = self.geometry_cases or [{'ID': 'base', 'is_dummy': True}]

            # --- 3. Generate All Case Combinations ---
            num_param_combos = len(standard_param_combinations)
            num_geom_cases = len(geometry_cases_to_run)
            total_cases_to_generate = num_param_combos * num_geom_cases
            self.log(f"Total combinations to generate: {total_cases_to_generate} ({num_geom_cases} geometries x {num_param_combos} parameter sets)")
            if total_cases_to_generate > 10000 and not messagebox.askyesno("Large Job", f"Generate {total_cases_to_generate} cases?"):
                return

            test_summary = []
            overall_case_idx = 0
            
            for geo_case in geometry_cases_to_run:
                for param_combo in standard_param_combinations:
                    overall_case_idx += 1
                    geom_id = geo_case.get('ID', 'N/A')
                    case_name = f"case_{overall_case_idx:04d}_geom_{geom_id}"
                    case_dir = output_path / case_name
                    self.log(f"Creating test case {overall_case_idx}/{total_cases_to_generate}: {case_name}")
                    case_dir.mkdir()

                    # Copy all base files first
                    for file_key, file_info in self.file_structure.items():
                        self._copy_and_rewrite_paths(file_info["path"], case_dir / file_info["path"].name)

                    # Apply geometry-specific modifications if not a dummy case
                    if not geo_case.get('is_dummy', False):
                        if not self._apply_geometry_modifications(case_dir, geo_case):
                            self.log(f"  Skipping case {case_name} due to geometry modification error.")
                            shutil.rmtree(case_dir) # Clean up failed case
                            continue

                    # Apply standard parameter variations
                    case_params: Dict[str, Any] = {}
                    for param_idx, value in enumerate(param_combo):
                        entry = self.parameter_entries[param_idx]
                        file_key = entry["file_type"]
                        param_name = entry["param_name"]
                        param_info = self.discovered_parameters[file_key][param_name]

                        if isinstance(value, np.integer): value = int(value)
                        elif isinstance(value, np.floating): value = float(value)

                        case_params[f"{file_key}/{param_name}"] = value
                        self.modify_parameter_in_file(case_dir, file_key, param_name, value, param_info)

                    # Save case metadata
                    case_info_data = {
                        "case_name": case_name,
                        "fst_file": Path(self.base_fst_path.get()).name,
                        "geometry_id": geom_id,
                        "parameters": case_params,
                    }
                    test_summary.append(case_info_data)
                    (case_dir / "case_info.json").write_text(json.dumps(case_info_data, indent=2))

            # --- 4. Finalize and Save Summary ---
            final_num_cases = len(test_summary)
            summary_file = output_path / "test_cases_summary.json"
            summary_data = {
                "generation_date": datetime.now().isoformat(),
                "base_fst_file": self.base_fst_path.get(),
                "num_cases": final_num_cases,
                "test_cases": test_summary,
            }
            summary_file.write_text(json.dumps(summary_data, indent=4))

            self.log(f"Successfully generated {final_num_cases} test cases in '{output_path}'")
            if messagebox.askyesno("Success", f"Generated {final_num_cases} test cases.\nSwitch to 'Run Simulations' tab?"):
                self.notebook.select(self.tabs[self.RUN_TAB_NAME])
                self.load_run_cases()

        except Exception as exc:
            self.log(f"Error: {exc}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to generate test cases: {exc}")

    def _copy_and_rewrite_paths(self, source_path: Path, dest_path: Path) -> None:
        if source_path.suffix.lower() not in {".fst", ".dat", ".twr", ".bld", ".ipt", ".txt", ".in"}:
            shutil.copy2(source_path, dest_path)
            return

        try:
            content = source_path.read_text(encoding="utf-8", errors="ignore")
            pattern = re.compile(r'(["\'])((?:\.\.[\\/])*[a-zA-Z0-9_.\-\s\\/]+)\1')

            def replacer(match: re.Match) -> str:
                quote = match.group(1)
                path_str = match.group(2)
                if path_str.lower() in {"default", "unused", "none"}:
                    return match.group(0)
                new_basename = Path(path_str).name
                return f"{quote}{new_basename}{quote}"

            new_content = pattern.sub(replacer, content)
            if new_content != content:
                self.log(f"    Rewrote internal paths in {dest_path.name}")
            dest_path.write_text(new_content, encoding="utf-8")
        except Exception as exc:
            self.log(f"    Warning: Error rewriting {source_path.name}: {exc}. Copying as-is.")
            shutil.copy2(source_path, dest_path)

    def modify_parameter_in_file(
        self,
        case_dir: Path,
        file_key: str,
        param_name: str,
        value: Any,
        param_info: Dict[str, Any],
    ) -> None:
        file_path = case_dir / self.file_structure[file_key]["path"].name
        if not file_path.exists():
            self.log(f"Warning: File {file_path} not found for param {param_name}")
            return

        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
        line_num = param_info.get("line_number", -1)

        if 0 <= line_num < len(lines) and param_name in lines[line_num]:
            lines[line_num] = self.format_parameter_line(lines[line_num], value, param_info)
            file_path.write_text("".join(lines), encoding="utf-8")
        else:
            self.log(f"Warning: Parameter '{param_name}' not found at expected line in {file_path.name}. Searching file...")
            for idx, line in enumerate(lines):
                if re.search(rf"\b{re.escape(param_name)}\b", line) and not line.strip().startswith(("!", "#")):
                    lines[idx] = self.format_parameter_line(line, value, param_info)
                    file_path.write_text("".join(lines), encoding="utf-8")
                    return
            self.log(f"Error: Could not find parameter '{param_name}' to modify in {file_path.name}")

    @staticmethod
    def format_parameter_line(line: str, new_value: Any, param_info: Dict[str, Any]) -> str:
        param_type = param_info.get("type")
        if param_type == "float":
            value_str = f"{float(new_value):.7G}"
        elif param_type == "bool":
            value_str = str(bool(new_value)).upper()
        elif param_type == "option":
            value_str = f'"{new_value}"' if " " in str(new_value) else str(new_value)
        else:
            value_str = str(new_value)

        parts = line.split()
        if not parts:
            return line
        return re.sub(r"^\s*[^\s]+", f"{value_str: >{len(parts[0])}}", line, count=1)

    # -------------------------------------------------------------------------
    # CASE TREE LOADING & EXECUTION
    # -------------------------------------------------------------------------
    def _load_cases_to_tree(self, tree, case_dict, log_widget) -> bool:
        test_dir = self.output_dir.get() or filedialog.askdirectory(title="Select Test Case Directory")
        if not test_dir:
            return False
        self.output_dir.set(test_dir)

        tree.delete(*tree.get_children())
        case_dict.clear()

        summary_file = Path(test_dir) / "test_cases_summary.json"
        if not summary_file.exists():
            messagebox.showerror("Error", f"Could not find 'test_cases_summary.json' in {test_dir}")
            return False

        with open(summary_file, "r") as f_in:
            summary = json.load(f_in)

        for case_info in summary.get("test_cases", []):
            param_items = []
            # ### MODIFIED ### Add geometry ID to the parameter string
            if 'geometry_id' in case_info:
                param_items.append(f"geom={case_info['geometry_id']}")
            
            param_items.extend(
                [
                    f"{k.split('/')[-1]}={v:.3g}" if isinstance(v, (int, float)) else f"{k.split('/')[-1]}={v}"
                    for k, v in case_info.get("parameters", {}).items()
                ]
            )
            params_str = ", ".join(param_items)

            item_id = tree.insert(
                "",
                "end",
                text=case_info["case_name"],
                values=("Ready", params_str, "-", "-"),
            )
            case_dict[item_id] = {
                "path": Path(test_dir) / case_info["case_name"],
                "fst_file": case_info["fst_file"],
                "name": case_info["case_name"],
            }

        log_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(case_dict)} cases from {test_dir}\n")
        tree.selection_set(tree.get_children())
        return True

    def load_run_cases(self) -> None:
        self._load_cases_to_tree(self.run_widgets["tree"], self.task_data["run"]["cases"], self.run_widgets["log"])

    def load_post_proc_cases(self) -> None:
        self._load_cases_to_tree(
            self.post_proc_widgets["tree"],
            self.task_data["post_proc"]["cases"],
            self.post_proc_widgets["log"],
        )

    def run_selected_cases(self) -> None:
        self._start_task("run", "OpenFAST simulations")

    def run_selected_post_proc(self) -> None:
        tasks_selected = (
            self.run_convert_csv.get()
            or self.run_dalembert.get()
            or self.run_plotting.get()
            or self.run_frequency_analysis.get()
        )
        if not tasks_selected:
            messagebox.showwarning("Warning", "No post-processing tasks selected.")
            return
        if self.run_frequency_analysis.get() and not self.frequency_analysis_column.get().strip():
            messagebox.showerror("Input Error", "Please specify a column name for Frequency Analysis.")
            return
        self._start_task("post_proc", "post-processing tasks")

    def _start_task(self, task_key: str, task_name: str) -> None:
        widgets = self.run_widgets if task_key == "run" else self.post_proc_widgets
        task_info = self.task_data[task_key]

        selected_items = widgets["tree"].selection()
        if not selected_items:
            messagebox.showwarning("Warning", f"No cases selected for {task_name}.")
            return
        if not messagebox.askyesno("Confirm", f"This will run {len(selected_items)} {task_name}. Continue?"):
            return

        widgets["progress_var"].set(0)
        task_info["completed"] = 0
        task_info["total"] = len(selected_items)

        while not task_info["job_queue"].empty():
            task_info["job_queue"].get()
        for item_id in selected_items:
            task_info["job_queue"].put(item_id)

        widgets["run_button"].config(state="disabled")
        manager_thread = threading.Thread(
            target=self._task_manager_thread,
            args=(task_key,),
            daemon=True,
        )
        manager_thread.start()

    def _task_manager_thread(self, task_key: str) -> None:
        task_info = self.task_data[task_key]
        worker_func = self.run_worker if task_key == "run" else self.post_proc_worker
        num_workers = self.num_threads.get()

        self.message_queue.put((f"{task_key}_log", f"Starting {task_info['total']} tasks with {num_workers} parallel workers..."))
        threads = [threading.Thread(target=worker_func, daemon=True) for _ in range(num_workers)]
        for thread in threads:
            thread.start()

        task_info["job_queue"].join()

        self.message_queue.put((f"{task_key}_log", "\n--- All tasks completed. ---"))
        self.message_queue.put((f"enable_{task_key}_button", None))

    def run_worker(self) -> None:
        while True:
            try:
                item_id = self.task_data["run"]["job_queue"].get_nowait()
            except queue.Empty:
                return

            case_data = self.task_data["run"]["cases"][item_id]
            case_path = case_data["path"]
            case_name = case_data["name"]

            self.message_queue.put(("run_tree_update", (item_id, "Status", "Running")))
            self.message_queue.put(("run_log", f"--- Running {case_name} ---"))
            start_time = datetime.now()

            try:
                cmd = [self.openfast_exe.get(), case_data["fst_file"]]
                process = subprocess.Popen(
                    cmd,
                    cwd=str(case_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )

                has_error = False
                error_keywords = ["error:", "error ", "aborting", "failed", "fortran runtime error"]

                for line in iter(process.stdout.readline, ""):
                    log_line = f"[{case_name}] {line.strip()}"
                    self.message_queue.put(("run_log", log_line))
                    if any(keyword in line.lower() for keyword in error_keywords):
                        has_error = True

                process.wait()
                runtime = (datetime.now() - start_time).total_seconds()

                if process.returncode != 0 or has_error:
                    result = f"Error (code {process.returncode})" if not has_error else "Error (in output)"
                    status = "Failed"
                else:
                    result = "Success"
                    status = "Completed"

            except Exception as exc:
                runtime = (datetime.now() - start_time).total_seconds()
                result = f"Exception: {exc}"
                status = "Failed"
                self.message_queue.put(
                    ("run_log", f"FATAL ERROR launching {case_name}: {exc}\n{traceback.format_exc()}")
                )

            self.message_queue.put(("run_tree_update", (item_id, "Status", status)))
            self.message_queue.put(("run_tree_update", (item_id, "Result", result)))
            self.message_queue.put(("run_tree_update", (item_id, "Runtime", f"{runtime:.1f}s")))

            with self.task_data["run"]["progress_lock"]:
                self.task_data["run"]["completed"] += 1
                progress = (self.task_data["run"]["completed"] / self.task_data["run"]["total"]) * 100
                self.message_queue.put(("run_progress", progress))

            self.task_data["run"]["job_queue"].task_done()
            gc.collect()

    def post_proc_worker(self) -> None:
        while True:
            try:
                item_id = self.task_data["post_proc"]["job_queue"].get_nowait()
            except queue.Empty:
                return

            case_data = self.task_data["post_proc"]["cases"][item_id]
            case_path = case_data["path"]
            case_name = case_data["name"]

            self.message_queue.put(("post_proc_tree_update", (item_id, "Status", "Processing")))
            self.message_queue.put(("post_proc_log", f"--- Processing {case_name} ---"))

            success = self.run_post_processing_steps(case_data)
            status = "Completed" if success else "Failed"
            result = "Success" if success else "Task(s) failed"

            self.message_queue.put(("post_proc_tree_update", (item_id, "Status", status)))
            self.message_queue.put(("post_proc_tree_update", (item_id, "Result", result)))

            with self.task_data["post_proc"]["progress_lock"]:
                self.task_data["post_proc"]["completed"] += 1
                progress = (
                    self.task_data["post_proc"]["completed"] / self.task_data["post_proc"]["total"]
                ) * 100
                self.message_queue.put(("post_proc_progress", progress))

            self.task_data["post_proc"]["job_queue"].task_done()
            self.message_queue.put(("post_proc_log", f"[{case_name}] Requesting garbage collection to free memory."))
            gc.collect()

    def run_post_processing_steps(self, case_data: Dict[str, Any]) -> bool:
        case_path = case_data["path"]
        case_name = case_data["name"]
        self.message_queue.put(("post_proc_log", f"[{case_name}] Searching for main .out file..."))

        out_files = [
            f
            for f in case_path.glob("*.out")
            if "MD.out" not in f.name and "MoorDyn.out" not in f.name
        ]
        if not out_files:
            self.message_queue.put(
                ("post_proc_log", f"[{case_name}] ERROR: No suitable .out file found. Simulation may have failed.")
            )
            return False
        main_out_file = out_files[0]
        if len(out_files) > 1:
            self.message_queue.put(
                ("post_proc_log", f"[{case_name}] WARNING: Multiple .out files found, using '{main_out_file.name}'")
            )

        csv_path = main_out_file.with_suffix(".csv")
        overall_success = True

        analysis_start_time = self.DEFAULT_ANALYSIS_START_TIME
        try:
            fst_content = (case_path / case_data["fst_file"]).read_text()
            tmax_match = re.search(r"^\s*([\d.eE+-]+)\s+TMax", fst_content, re.IGNORECASE | re.MULTILINE)
            if tmax_match:
                analysis_start_time = float(tmax_match.group(1)) / 3.0
        except Exception:
            pass
        self.message_queue.put(("post_proc_log", f"[{case_name}] Using analysis start time: {analysis_start_time:.2f}s"))

        if self.run_convert_csv.get():
            try:
                converter = ConverterRunner(self.message_queue, case_name, "post_proc_log")
                if not converter.convert_openfast_to_csv_robust(str(main_out_file), str(csv_path)):
                    self.message_queue.put(
                        ("post_proc_log", f"[{case_name}] CSV conversion failed. Halting subsequent tasks.")
                    )
                    return False
            except Exception as exc:
                self.message_queue.put(
                    (
                        "post_proc_log",
                        f"[{case_name}] FATAL ERROR during CSV conversion: {exc}\n{traceback.format_exc()}",
                    )
                )
                return False

        if self.run_dalembert.get():
            try:
                dalembert_dir = case_path / "dalembert_analysis"
                dalembert_dir.mkdir(exist_ok=True)
                DalembertRunner(self.message_queue, case_name, "post_proc_log").run(
                    fst=str(case_path / case_data["fst_file"]),
                    glue_out=str(main_out_file),
                    outdir=str(dalembert_dir),
                    analysis_start_time=analysis_start_time,
                )
            except Exception as exc:
                self.message_queue.put(
                    ("post_proc_log", f"[{case_name}] ERROR in d'Alembert analysis: {exc}\n{traceback.format_exc()}")
                )
                overall_success = False

        if self.run_plotting.get() and csv_path.exists():
            with self.plotting_lock:
                try:
                    plot_dir = case_path / "plots"
                    plot_dir.mkdir(exist_ok=True)
                    PlottingRunner(self.message_queue, case_name, "post_proc_log").run(
                        csv_file=str(csv_path),
                        output_dir=str(plot_dir),
                        case_name=case_name,
                        mean_start=analysis_start_time,
                        always_minmax=False,
                        minmax_range_frac=0.05,
                        minmax_abs=0.0,
                    )
                except Exception as exc:
                    self.message_queue.put(
                        ("post_proc_log", f"[{case_name}] ERROR in plotting: {exc}\n{traceback.format_exc()}")
                    )
                    overall_success = False

        if self.run_frequency_analysis.get() and SCIPY_AVAILABLE and csv_path.exists():
            with self.plotting_lock:
                try:
                    freq_dir = case_path / "frequency_analysis"
                    freq_dir.mkdir(exist_ok=True)
                    FrequencyAnalysisRunner(self.message_queue, case_name, "post_proc_log").run(
                        csv_file=str(csv_path),
                        column_name=self.frequency_analysis_column.get(),
                        output_dir=str(freq_dir),
                        start_time=analysis_start_time,
                    )
                except Exception as exc:
                    self.message_queue.put(
                        ("post_proc_log", f"[{case_name}] ERROR in Frequency Analysis: {exc}\n{traceback.format_exc()}")
                    )
                    overall_success = False

        return overall_success

    # -------------------------------------------------------------------------
    # GUI HELPERS
    # -------------------------------------------------------------------------
    def show_case_context_menu(self, event, tree, case_dict) -> None:
        item_id = tree.identify_row(event.y)
        if not item_id:
            return
        tree.selection_set(item_id)
        case_data = case_dict.get(item_id)
        if not case_data:
            return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label=f"Open Folder for '{case_data['name']}'",
            command=lambda path=case_data["path"]: self.open_folder(path),
        )
        menu.post(event.x_root, event.y_root)

    @staticmethod
    def open_folder(path: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open folder: {exc}")

    def show_file_structure(self) -> None:
        if not self.file_structure:
            messagebox.showinfo("Info", "Run 'Discover Parameters' first.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Discovered File Structure")
        dialog.geometry("800x600")

        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=("Consolas", 10))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("end", "OpenFAST File Structure:\n" + "=" * 60 + "\n\n")
        for file_key, file_info in sorted(self.file_structure.items()):
            text.insert("end", f"{file_key}:\n", "heading")
            text.insert("end", f"  Path: {file_info.get('path')}\n")
            text.insert("end", f"  Parameters Found: {len(self.discovered_parameters.get(file_key, {}))}\n\n")
        text.tag_config("heading", font=("Consolas", 11, "bold"), foreground="darkblue")
        text.config(state="disabled")
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def save_config(self) -> None:
        if not self.parameter_entries and not self.geometry_csv_path.get(): # ### MODIFIED ###
            messagebox.showinfo("Info", "No parameters or geometry to save.")
            return

        config = {
            "base_fst_path": self.base_fst_path.get(),
            "output_dir": self.output_dir.get(),
            "num_cases": self.num_cases.get(),
            "distribution": self.distribution_var.get(),
            "geometry_csv_path": self.geometry_csv_path.get(), # ### NEW ###
            "parameters": [],
        }

        for entry in self.parameter_entries:
            param_data = { "file_type": entry["file_type"], "param_name": entry["param_name"], "csv_list": entry["csv_var"].get() }
            param_type = entry["param_info"]["type"]
            if param_type == "float": param_data.update({"start": entry["start_var"].get(), "end": entry["end_var"].get(), "steps": entry["steps_var"].get()})
            elif param_type == "int": param_data.update({"int_mode": entry["int_mode_var"].get(), "start": entry["start_var"].get(), "end": entry["end_var"].get(), "steps": entry["steps_var"].get(), "int_list": entry["list_var"].get()})
            elif param_type == "bool": param_data.update({"bool_choice": entry["bool_var"].get()})
            elif param_type == "option": param_data.update({"options_list": entry["options_var"].get()})
            config["parameters"].append(param_data)

        filename = filedialog.asksaveasfilename(title="Save Configuration", defaultextension=".json", filetypes=[("JSON config", "*.json")])
        if filename:
            with open(filename, "w") as f_out: json.dump(config, f_out, indent=4)
            self.log(f"Configuration saved to: {filename}")

    def load_config(self) -> None:
        filename = filedialog.askopenfilename(title="Load Configuration", filetypes=[("JSON config", "*.json")])
        if not filename: return
        try:
            with open(filename, "r") as f_in: config = json.load(f_in)

            self.base_fst_path.set(config.get("base_fst_path", ""))
            self.output_dir.set(config.get("output_dir", "test_cases"))
            self.num_cases.set(config.get("num_cases", 10))
            self.distribution_var.set(config.get("distribution", "grid_search"))

            # ### NEW ### Load geometry file if specified in config
            self.geometry_cases = []
            self.geometry_csv_path.set("")
            self.geometry_status_label.config(text="No geometry file loaded.")
            geom_path = config.get("geometry_csv_path")
            if geom_path and Path(geom_path).exists():
                self.geometry_csv_path.set(geom_path) # Temporarily set to load
                self.browse_and_load_geometry_csv() # Reuse the loading and validation logic
                # This is a bit of a hack. We set the path, then call the function which opens a dialog.
                # A better way would be to refactor browse_and_load into load_geometry(path).
                # For this implementation, we'll just re-set the path after the dialog is cancelled.
                self.geometry_csv_path.set(geom_path)
            elif geom_path:
                self.log(f"Warning: Geometry CSV from config not found: {geom_path}")


            self.clear_parameters()

            if self.base_fst_path.get() and not self.discovered_parameters:
                self.log("Base FST found, running discovery...")
                self.discover_parameters()
            if not self.discovered_parameters:
                messagebox.showwarning("Warning", "Run parameter discovery before loading parameters.")
                return

            for param_config in config.get("parameters", []):
                file_type, param_name = param_config.get("file_type"), param_config.get("param_name")
                if file_type and param_name and file_type in self.discovered_parameters and param_name in self.discovered_parameters[file_type]:
                    param_info = self.discovered_parameters[file_type][param_name]
                    self.add_parameter_with_info(file_type, param_name, param_info)
                    entry = self.parameter_entries[-1]
                    if "csv_list" in param_config: entry["csv_var"].set(param_config.get("csv_list", ""))
                    if entry["param_info"]["type"] == "float": entry["start_var"].set(param_config.get("start", 0)); entry["end_var"].set(param_config.get("end", 1)); entry["steps_var"].set(param_config.get("steps", 5))
                    elif entry["param_info"]["type"] == "int": entry["int_mode_var"].set(param_config.get("int_mode", "Range")); entry["start_var"].set(param_config.get("start", 0)); entry["end_var"].set(param_config.get("end", 1)); entry["steps_var"].set(param_config.get("steps", 5)); entry["list_var"].set(param_config.get("int_list", "1,2,3"))
                    elif entry["param_info"]["type"] == "bool": entry["bool_var"].set(param_config.get("bool_choice", "Vary (True & False)"))
                    elif entry["param_info"]["type"] == "option": entry["options_var"].set(param_config.get("options_list", ""))
                else:
                    self.log(f"Warning: Could not find '{param_name}' in '{file_type}' from config.")
            self.log(f"Configuration loaded from: {filename}")
            self.on_distribution_change()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load configuration: {exc}")
            self.log(f"Error loading config: {exc}")

    def clear_parameters(self) -> None:
        for entry in self.parameter_entries:
            entry["frame"].destroy()
        self.parameter_entries.clear()
        self.update_total_cases()

    def log(self, message: str) -> None:
        self.setup_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.setup_log.see(tk.END)
        self.root.update_idletasks()

    def process_queue(self) -> None:
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                if msg_type.endswith("_log"):
                    log_widget = getattr(self, msg_type)
                    log_widget.insert(tk.END, msg_data + "\n")
                    log_widget.see(tk.END)
                elif msg_type.endswith("_tree_update"):
                    tree = self.run_widgets["tree"] if "run" in msg_type else self.post_proc_widgets["tree"]
                    item_id, column, value = msg_data
                    tree.set(item_id, column, value)
                elif msg_type.endswith("_progress"):
                    widgets = self.run_widgets if "run" in msg_type else self.post_proc_widgets
                    widgets["progress_bar"]["value"] = msg_data
                elif msg_type.startswith("enable_"):
                    key = msg_type.replace("enable_", "")
                    widgets = self.run_widgets if "run" in key else self.post_proc_widgets
                    widgets["run_button"].config(state="normal")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    # -------------------------------------------------------------------------
    # PARAMETER MANAGEMENT UI
    # -------------------------------------------------------------------------
    def create_tutorial_tab(self, parent_frame: ttk.Frame) -> None:
        text_widget = scrolledtext.ScrolledText(
            parent_frame,
            wrap=tk.WORD,
            relief="flat",
            padx=10,
            pady=10,
        )
        text_widget.pack(fill="both", expand=True)

        tutorial_text = [
                ("Welcome to the OpenFAST Workflow Manager!\n", 'h1'),
                ("This tool is designed to streamline the process of running large batches of OpenFAST simulations and analyzing their results. The workflow is organized into three main tabs.\n\n", ''),

                ("Tab 1: Setup Cases\n", 'h2'),
                ("The goal of this tab is to create a set of test case directories, each containing a modified version of a base OpenFAST model.\n\n", ''),
                ("1. File Selection:", 'bold'),
                (" First, select your main OpenFAST input file (", ''),
                (".fst", 'code'),
                (") and specify a root ", ''),
                ("Output Directory", 'code'),
                (" where all test cases will be generated.\n", ''),
                ("2. Geometry Import (New!):", 'bold'),
                (" Use the ", ''),
                ("Browse & Import", 'code'),
                (" button in the 'Geometry Import' section to load a CSV file containing different platform geometries. Each row in the CSV is a separate geometry case.\n", ''),
                ("3. Parameter Discovery:", 'bold'),
                (" Click ", ''),
                ("Discover Parameters", 'code'),
                (". The application will scan your ", ''),
                (".fst", 'code'),
                (" file and all referenced input files (ElastoDyn, AeroDyn, etc.) to find numerical parameters that can be varied.\n", ''),
                ("4. Parameter Configuration:", 'bold'),
                (" Click ", ''),
                ("Add from Discovery", 'code'),
                (" to add standard OpenFAST parameters you wish to vary (e.g., wave height, wind speed). These variations will be applied to *each* geometry case.\n", ''),
                ("5. Generate Cases:", 'bold'),
                (" Click ", ''),
                ("Generate Test Cases", 'code'),
                (". This will create a folder for each combination of geometry and parameter variation (e.g., 4 geometries x 10 parameter sets = 40 total cases).\n\n", ''),
                ("IMPORTANT NOTES: 5MW BASELINE FOLDER MUST BE COPY IN THE TEST CASE GENERATION IF USING EXAMPLE TEST CASE", 'h2'),

                ("\nTab 2: Run Simulations\n", 'h2'),
                ("The goal of this tab is to execute the OpenFAST simulations for the generated cases.\n\n", ''),
                ("1. Configuration:", 'bold'),
                (" Browse for your ", ''),
                ("OpenFAST executable", 'code'),
                (" and set the desired number of ", ''),
                ("parallel runs", 'code'),
                (" (a good starting point is half your CPU cores).\n", ''),
                ("2. Load Cases:", 'bold'),
                (" Click ", ''),
                ("Load Test Cases", 'code'),
                (". The application will automatically use the directory from the Setup tab. It reads the ", ''),
                ("test_cases_summary.json", 'code'),
                (" file to populate the list.\n", ''),
                ("3. Run Simulations:", 'bold'),
                (" Select the cases you want to run (or use 'Select All') and click ", ''),
                ("Run Selected Simulations", 'code'),
                (".\n", ''),
                ("4. Monitor Progress:", 'bold'),
                (" The status of each case will update in the table. The log at the bottom shows the real-time output from the OpenFAST simulations.\n\n", ''),

                ("Tab 3: Post-Process Results\n", 'h2'),
                ("The goal of this tab is to automatically analyze the output data from successfully completed simulations.\n\n", ''),
                ("1. Configuration:", 'bold'),
                (" Ensure the ", ''),
                ("Results Directory", 'code'),
                (" is correct. Select the analysis tasks you want to perform.\n", ''),
                ("2. Load Results:", 'bold'),
                (" Click ", ''),
                ("Load Results", 'code'),
                (" to populate the list with all available cases from the directory.\n", ''),
                ("3. Run Post-Processing:", 'bold'),
                (" Select the desired cases and click ", ''),
                ("Run Post-Processing", 'code'),
                (".\n", ''),
                ("4. Review Artifacts:", 'bold'),
                (" Once processing is complete, you can easily access the results. ", ''),
                ("Right-click on any case", 'bold'),
                (" in the list and select ", ''),
                ("Open Folder", 'code'),
                (" to view the generated CSV files, reports, and plots.\n", ''),

                ("Final Notes\n", 'h2'),
                ("Thank you for using the OpenFAST Workflow Manager! We hope this tool enhances your simulation workflow and analysis efficiency.\n", ''),
                ("For further assistance or to report issues, please visit our GitHub repository or contact the development team. \nAuthor: Trang Vinh Nghi\nDevelopment Supported By the Department of Aerospace Engineering - Ho Chi Minh City University of Technology - Viet Nam National University \nEmail: trangvinhnghi2212@gmail.com\nGitHub Repo Link: https://github.com/TomatoXoX/OpenFAST_GUI_Toolbox", '')
            ]

        text_widget.tag_configure("h1", font=("TkDefaultFont", 16, "bold"), spacing3=10)
        text_widget.tag_configure("h2", font=("TkDefaultFont", 12, "bold"), spacing1=15, spacing3=5)
        text_widget.tag_configure("bold", font=("TkDefaultFont", 9, "bold"))
        text_widget.tag_configure("code", font=("Consolas", 9), background="#f0f0f0")
        text_widget.tag_configure("list_item", lmargin1=20, lmargin2=20)
        for text, tag in tutorial_text:
            text_widget.insert(tk.END, text, tag)
        text_widget.config(state="disabled")

    def browse_fst_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select base FST file",
            filetypes=[("FST files", "*.fst"), ("All files", "*.*")],
        )
        if filename:
            self.base_fst_path.set(filename)
            self.log("Selected FST file: " + filename)
            if messagebox.askyesno("Discover Parameters", "Discover parameters for this file now?"):
                self.discover_parameters()

    def browse_output_dir(self) -> None:
        dirname = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir.get(),
        )
        if dirname:
            self.output_dir.set(dirname)
            self.log("Selected output directory: " + dirname)

    def browse_openfast_exe(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select OpenFAST executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if filename:
            self.openfast_exe.set(filename)
            self.message_queue.put(("run_log", f"Selected OpenFAST executable: {filename}"))

    def show_parameter_selector(self) -> None:
        if not self.discovered_parameters:
            messagebox.showinfo("Info", "Run 'Discover Parameters' first.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Parameters to Vary")
        dialog.geometry("900x700")

        search_frame = ttk.Frame(dialog)
        search_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side="left", padx=5)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var, width=30)
        search_entry.pack(side="left", padx=5)

        tree_frame = ttk.Frame(dialog)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        tree = ttk.Treeview(tree_frame, columns=("Type", "Value", "Unit", "Description"), show="tree headings")
        tree.heading("#0", text="Parameter")
        tree.heading("Type", text="Type")
        tree.heading("Value", text="Current Value")
        tree.heading("Unit", text="Unit")
        tree.heading("Description", text="Description")
        tree.column("#0", width=200)
        tree.column("Type", width=80)
        tree.column("Value", width=100, anchor="e")
        tree.column("Unit", width=80)
        tree.column("Description", width=350)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        all_items = []
        for file_type, params in sorted(self.discovered_parameters.items()):
            file_node = tree.insert("", "end", text=file_type, open=False, tags=("file_node",))
            for param_name, param_info in sorted(params.items()):
                val_str = (f"{param_info['original_value']:.4g}" if isinstance(param_info["original_value"], float) else str(param_info["original_value"]))
                item = tree.insert(file_node, "end", text=param_name, values=(param_info["type"], val_str, param_info.get("unit", ""), param_info["description"][:100]))
                all_items.append((item, file_type.lower(), param_name.lower(), param_info["description"].lower()))
        tree.tag_configure("file_node", font=("TkDefaultFont", 10, "bold"))

        def search_params(*_) -> None:
            term = search_var.get().lower()
            for child in tree.get_children(): tree.item(child, open=False); tree.reattach(child, "", "end")
            if not term: return
            for child in tree.get_children(): tree.detach(child)
            for item, file_type, param_name, desc in all_items:
                if term in param_name or term in desc or term in file_type:
                    parent = tree.parent(item)
                    tree.reattach(parent, "", "end")
                    tree.item(parent, open=True)

        search_var.trace("w", search_params)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", pady=10, padx=10)

        def add_selected() -> None:
            added_count = 0
            for item in tree.selection():
                parent = tree.parent(item)
                if parent:
                    file_type = tree.item(parent)["text"]
                    param_name = tree.item(item)["text"]
                    self.add_parameter_with_info(file_type, param_name, self.discovered_parameters[file_type][param_name])
                    added_count += 1
            dialog.destroy()
            if added_count > 0: self.log(f"Added {added_count} parameters for variation.")

        ttk.Button(btn_frame, text="Add Selected", command=add_selected, style="Accent.TButton").pack(side="right")
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=5)

    def _get_widget_state(self, widget) -> Optional[str]:
        try: return widget.cget("state")
        except (tk.TclError, AttributeError): return None

    def _apply_widget_state(self, widget, state: Optional[str]) -> None:
        target_state = "normal" if not state else state
        try: widget.configure(state=target_state)
        except (tk.TclError, AttributeError): pass

    def _reset_entry_widget_states(self, entry: Dict[str, Any]) -> None:
        default_states = entry.get("default_states")
        if default_states is None:
            default_states = {}
            for name, widget in entry["widgets"].items(): default_states[name] = self._get_widget_state(widget)
            entry["default_states"] = default_states
        for name, widget in entry["widgets"].items(): self._apply_widget_state(widget, default_states.get(name))

    def _apply_sampling_state(self, entry: Dict[str, Any], param_type: str) -> None:
        for name, widget in entry["widgets"].items():
            if name in {"info_lbl", "remove_btn"}: continue
            self._apply_widget_state(widget, "disabled")
        if param_type in {"float", "int"}:
            for key in ("range_ent_s", "range_ent_e", "range_spn_st"):
                if key in entry["widgets"]: self._apply_widget_state(entry["widgets"][key], "normal")
            if param_type == "int":
                entry["int_mode_var"].set("Range")
                if "update_func" in entry: entry["update_func"]()

    def add_parameter_with_info(self, file_type, param_name, param_info) -> None:
        if any(e["file_type"] == file_type and e["param_name"] == param_name for e in self.parameter_entries):
            self.log(f"Parameter {file_type} - {param_name} is already added."); return

        row_frame = ttk.Frame(self.param_list_frame)
        row_frame.pack(fill="x", pady=4, padx=2)
        ttk.Label(row_frame, text=f"{file_type} - {param_name}", width=35, anchor="w", wraplength=220).grid(row=0, column=0, rowspan=2, padx=5, sticky="w")

        param_type, current_val = param_info["type"], param_info["original_value"]
        entry_data: Dict[str, Any] = {"frame": row_frame, "file_type": file_type, "param_name": param_name, "param_info": param_info, "widgets": {}}
        csv_var = tk.StringVar(value=str(current_val))
        entry_data.update({"csv_var": csv_var, "widgets": {"csv_lbl": ttk.Label(row_frame, text="CSV Values:"), "csv_ent": ttk.Entry(row_frame, textvariable=csv_var, width=40)}})
        csv_var.trace_add("write", self.update_total_cases)

        if param_type == "float":
            start_default, end_default = (current_val * 0.8, current_val * 1.2) if isinstance(current_val, (int, float)) and abs(current_val) > 1e-9 else (-1.0, 1.0)
            start_var, end_var, steps_var = tk.DoubleVar(value=start_default), tk.DoubleVar(value=end_default), tk.IntVar(value=5)
            entry_data.update({"start_var": start_var, "end_var": end_var, "steps_var": steps_var})
            entry_data["widgets"].update({"range_lbl_s": ttk.Label(row_frame, text="Start:"), "range_ent_s": ttk.Entry(row_frame, textvariable=start_var, width=10), "range_lbl_e": ttk.Label(row_frame, text="End:"), "range_ent_e": ttk.Entry(row_frame, textvariable=end_var, width=10), "range_lbl_st": ttk.Label(row_frame, text="Steps:"), "range_spn_st": ttk.Spinbox(row_frame, from_=1, to=100, textvariable=steps_var, width=5)})
            steps_var.trace_add("write", self.update_total_cases)
        elif param_type == "int":
            mode_var, start_var, end_var, steps_var, list_var = tk.StringVar(value="Range"), tk.DoubleVar(value=current_val), tk.DoubleVar(value=current_val + 4), tk.IntVar(value=5), tk.StringVar(value=str(current_val))
            def update_int_widgets() -> None:
                is_range = mode_var.get() == "Range"
                for name, widget in entry_data["widgets"].items():
                    if name.startswith("range_"): widget.grid() if is_range else widget.grid_remove()
                    if name.startswith("list_"): widget.grid() if not is_range else widget.grid_remove()
                self.update_total_cases()
            entry_data.update({"int_mode_var": mode_var, "start_var": start_var, "end_var": end_var, "steps_var": steps_var, "list_var": list_var, "update_func": update_int_widgets})
            entry_data["widgets"].update({"rad_range": ttk.Radiobutton(row_frame, text="Range", variable=mode_var, value="Range", command=update_int_widgets), "rad_list": ttk.Radiobutton(row_frame, text="List", variable=mode_var, value="List", command=update_int_widgets), "range_lbl_s": ttk.Label(row_frame, text="Start:"), "range_ent_s": ttk.Entry(row_frame, textvariable=start_var, width=8), "range_lbl_e": ttk.Label(row_frame, text="End:"), "range_ent_e": ttk.Entry(row_frame, textvariable=end_var, width=8), "range_lbl_st": ttk.Label(row_frame, text="Steps:"), "range_spn_st": ttk.Spinbox(row_frame, from_=1, to=100, textvariable=steps_var, width=5), "list_lbl": ttk.Label(row_frame, text="List (CSV):"), "list_ent": ttk.Entry(row_frame, textvariable=list_var, width=25)})
            steps_var.trace_add("write", self.update_total_cases); list_var.trace_add("write", self.update_total_cases)
        elif param_type == "bool":
            bool_var = tk.StringVar(value="Vary (True & False)")
            entry_data.update({"bool_var": bool_var})
            entry_data["widgets"].update({"bool_lbl": ttk.Label(row_frame, text="Value:"), "bool_combo": ttk.Combobox(row_frame, textvariable=bool_var, values=["Vary (True & False)", "True", "False"], width=20, state="readonly")})
            bool_var.trace_add("write", self.update_total_cases)
        elif param_type == "option":
            options_var = tk.StringVar(value=f'"{current_val}"')
            entry_data.update({"options_var": options_var})
            entry_data["widgets"].update({"opt_lbl": ttk.Label(row_frame, text="Options (CSV):"), "opt_ent": ttk.Entry(row_frame, textvariable=options_var, width=30)})
            options_var.trace_add("write", self.update_total_cases)

        entry_data["widgets"]["info_lbl"] = ttk.Label(row_frame, text=f"[{param_info.get('unit', '')}] (Type: {param_type}, Current: {current_val})", foreground="gray")
        entry_data["widgets"]["remove_btn"] = ttk.Button(row_frame, text="Remove", command=lambda entry=entry_data: self.remove_parameter(entry))
        row_frame.columnconfigure(8, weight=1)
        entry_data["default_states"] = {name: self._get_widget_state(widget) for name, widget in entry_data["widgets"].items()}
        self.parameter_entries.append(entry_data)
        self.on_distribution_change()

    def remove_parameter(self, entry_to_remove: Dict[str, Any]) -> None:
        entry_to_remove["frame"].destroy()
        self.parameter_entries.remove(entry_to_remove)
        self.update_total_cases()

    def refresh_distribution_settings(self) -> None:
        self.on_distribution_change()
        self.log("Distribution settings refreshed.")

    def on_distribution_change(self, event=None) -> None:
        dist_mode = self.distribution_var.get()
        is_grid, is_csv, is_sampling = dist_mode == "grid_search", dist_mode == "csv_columnwise", dist_mode in self.SAMPLING_DISTRIBUTIONS
        self.num_cases_spinbox.config(state="disabled" if is_grid or is_csv else "normal")

        for entry in self.parameter_entries:
            self._reset_entry_widget_states(entry)
            for widget in entry["widgets"].values():
                if hasattr(widget, "grid_remove"): widget.grid_remove()
            param_type = entry["param_info"]["type"]

            if is_csv:
                entry["widgets"]["csv_lbl"].grid(row=0, column=1, padx=(10, 2)); entry["widgets"]["csv_ent"].grid(row=0, column=2, columnspan=5, sticky="ew")
            else:
                if param_type == "float": entry["widgets"]["range_lbl_s"].grid(row=0, column=1, padx=(10, 2)); entry["widgets"]["range_ent_s"].grid(row=0, column=2); entry["widgets"]["range_lbl_e"].grid(row=0, column=3, padx=5); entry["widgets"]["range_ent_e"].grid(row=0, column=4); entry["widgets"]["range_lbl_st"].grid(row=0, column=5, padx=5); entry["widgets"]["range_spn_st"].grid(row=0, column=6)
                elif param_type == "int": entry["widgets"]["rad_range"].grid(row=0, column=1, sticky="w", padx=5); entry["widgets"]["rad_list"].grid(row=1, column=1, sticky="w", padx=5); entry["update_func"]()
                elif param_type == "bool": entry["widgets"]["bool_lbl"].grid(row=0, column=1, padx=(10, 2)); entry["widgets"]["bool_combo"].grid(row=0, column=2, columnspan=3)
                elif param_type == "option": entry["widgets"]["opt_lbl"].grid(row=0, column=1, padx=(10, 2)); entry["widgets"]["opt_ent"].grid(row=0, column=2, columnspan=5, sticky="ew")
                if is_sampling: self._apply_sampling_state(entry, param_type)

            entry["widgets"]["info_lbl"].grid(row=0, column=8, padx=5, sticky="w")
            entry["widgets"]["remove_btn"].grid(row=0, column=9, rowspan=2, padx=10)
        self.update_total_cases()

    def update_total_cases(self, *_) -> None:
        dist_mode = self.distribution_var.get()
        total = 0
        try:
            if dist_mode == "grid_search":
                total = 1 if self.parameter_entries else 0
                for entry in self.parameter_entries:
                    param_type = entry["param_info"]["type"]
                    if param_type == "float": total *= entry["steps_var"].get()
                    elif param_type == "int": total *= entry["steps_var"].get() if entry["int_mode_var"].get() == "Range" else max(1, len([i for i in entry["list_var"].get().split(",") if i.strip()]))
                    elif param_type == "bool": total *= 2 if "Vary" in entry["bool_var"].get() else 1
                    elif param_type == "option": total *= max(1, len([opt for opt in entry["options_var"].get().split(",") if opt.strip()]))
            elif dist_mode == "csv_columnwise":
                if self.parameter_entries: total = len([i for i in self.parameter_entries[0]["csv_var"].get().split(",") if i.strip()])
        except (tk.TclError, ValueError, IndexError): total = 0

        if dist_mode not in self.SAMPLING_DISTRIBUTIONS: self.num_cases.set(total if total else 0)

    # -------------------------------------------------------------------------
    # END CLASS
    # -------------------------------------------------------------------------


def main() -> None:
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError):  # pragma: no cover - platform specific
        pass

    root = tk.Tk()
    app = OpenFASTTestCaseGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()