import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import pandas as pd
import json
import os
import shutil
import subprocess
import threading
import queue
import itertools
import re
import math
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
from math import cos, sin, radians
from typing import List, Tuple, Dict, Any, Optional
# --- NEW: Import the ADVANCED, AUTOMATIC geometry engine ---
try:
    import advanced_geometry_engine as engine
    GEOMETRY_ENGINE_AVAILABLE = True
except ImportError:
    GEOMETRY_ENGINE_AVAILABLE = False
# Suppress the deprecation warning from matplotlib about findfont
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

try:
    import matplotlib
    # CRITICAL FIX: Use a thread-safe, non-interactive backend for plotting.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# #############################################################################
# --- BEGIN: Standalone Helper Functions (Refactored for Robustness) ---
# #############################################################################

def _strip_quotes(s: str) -> str:
    s = s.strip()
    return s[1:-1] if (s.startswith('"') and s.endswith('"')) else s

def _read_lines(p: str, logger: logging.Logger) -> List[str]:
    logger.debug(f"Reading file: {p}")
    return open(p, 'r', encoding='utf-8', errors='ignore').readlines()

def _find_fst_refs(fst: str, logger: logging.Logger) -> Dict[str, str]:
    logger.info(f"Parsing FST for module references: {fst}")
    base = os.path.dirname(os.path.abspath(fst))
    refs = {}
    pat = re.compile(r'^\s*"?([^"\s]+)"?\s+([A-Za-z0-9_()]+)')
    for ln in _read_lines(fst, logger):
        m = pat.match(ln.strip())
        if m:
            refs[m.group(2)] = _strip_quotes(m.group(1))
    for k, v in list(refs.items()):
        if v.lower() in ['unused', 'none']:
            continue
        if not os.path.isabs(v):
            refs[k] = os.path.normpath(os.path.join(base, v))
    return refs

# #############################################################################
# --- END: Standalone Helper Functions ---
# #############################################################################


# #############################################################################
# --- BEGIN: UPDATED Code from Convert_OUT2CSV.txt ---
# #############################################################################

class ConverterRunner:
    def __init__(self, message_queue: queue.Queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def log(self, message: str):
        self.mq.put((self.log_type, f"[{self.case_name}][CSV] {message}"))

    def convert_openfast_to_csv_robust(self, input_file: str, output_file: str) -> Optional[pd.DataFrame]:
        self.log(f"Attempting to convert '{Path(input_file).name}'...")

        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except FileNotFoundError:
            self.log(f"Error: The input file was not found at '{input_file}'")
            return None
        except Exception as e:
            self.log(f"Error reading file: {e}")
            return None

        header_lines, column_names, column_units, data_start_index = [], [], [], -1

        for i, line in enumerate(lines):
            if 'Time' in line.split() and i + 1 < len(lines):
                potential_names = line.strip().split()
                potential_units = lines[i+1].strip().split()
                if len(potential_names) == len(potential_units) and len(potential_names) > 1:
                    column_names = potential_names
                    column_units = potential_units
                    data_start_index = i + 2
                    header_lines = lines[:i-1]
                    break

        if not column_names:
            self.log("Error: Could not find the header and unit lines. Please check the .out file format.")
            return None

        # Handle duplicate column names before creating the DataFrame
        seen = {}
        unique_columns = []
        for i, col in enumerate(column_names):
            if col in seen:
                seen[col] += 1
                new_col_name = f"{col}_{seen[col]}"
                self.log(f"Warning: Duplicate column '{col}' found. Renaming to '{new_col_name}'.")
                unique_columns.append(new_col_name)
            else:
                seen[col] = 1
                unique_columns.append(col)
        column_names = unique_columns

        data = []
        for i, line in enumerate(lines[data_start_index:]):
            line = line.strip()
            if not line: continue
            
            values = line.split()
            if len(values) == len(column_names):
                try:
                    float_values = [float(val.replace('D', 'E')) for val in values]
                    data.append(float_values)
                except ValueError:
                    self.log(f"Warning: Could not parse data on line {data_start_index + i + 1}. Skipping.")
            else:
                self.log(f"Warning: Mismatch in column count on line {data_start_index + i + 1}. Expected {len(column_names)}, found {len(values)}. Skipping.")

        if not data:
            self.log("Error: No data was successfully parsed from the file.")
            return None

        df = pd.DataFrame(data, columns=column_names)
        df.to_csv(output_file, index=False, float_format='%.6E')

        metadata_file = output_file.rsplit('.', 1)[0] + '_metadata.txt'
        with open(metadata_file, 'w') as f:
            f.write("OpenFAST Output File Metadata\n" + "=" * 60 + "\n\n")
            f.write(f"Source File: {Path(input_file).name}\n\n")
            description_written = any("Description:" in hline for hline in header_lines)
            if description_written:
                for hline in header_lines:
                    if "Description:" in hline: f.write(hline)
            else: f.write("No 'Description:' line found in the original file header.\n")
            f.write("\nColumn Information:\n" + "-" * 60 + "\n")
            f.write(f"{'Column Name':<25} {'Units'}\n" + "-" * 60 + "\n")
            original_names_for_meta = lines[data_start_index - 2].strip().split()
            for name, unit in zip(original_names_for_meta, column_units):
                f.write(f"{name:<25} {unit}\n")

        self.log("--- Conversion Summary ---")
        self.log(f"{'Input file:':<20} {Path(input_file).name}")
        self.log(f"{'Output CSV:':<20} {Path(output_file).name}")
        self.log(f"{'Rows/Cols:':<20} {len(df)} / {len(df.columns)}")
        
        return df

# #############################################################################
# --- END: UPDATED Code from Convert_OUT2CSV.txt ---
# #############################################################################

# #############################################################################
# --- BEGIN: Code from plot_analysis_unnormalized.txt ---
# #############################################################################
class PlottingRunner:
    def __init__(self, message_queue: queue.Queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def log(self, message: str): self.mq.put((self.log_type, f"[{self.case_name}][Plot] {message}"))
    def simplify_header(self, name: str) -> str: return re.sub(r'\s+', '', re.sub(r'\s*\(.*?\)\s*$', '', str(name))).lower()
    def strip_units(self, name: str) -> str: return re.sub(r'\s*\(.*?\)\s*$', '', str(name)).strip()
    def find_time_column(self, df: pd.DataFrame) -> Optional[str]: return next((c for c in df.columns if re.match(r'^\s*Time\b', str(c), re.IGNORECASE)), None)
    def extract_units_from_header(self, name: str) -> str: m = re.search(r'\((.*?)\)', str(name)); return m.group(1) if m else ""
    def build_units_map_from_csv(self, columns: List[str]) -> Dict[str, str]: return {col: self.extract_units_from_header(col) for col in columns}

    def find_vector_columns(self, df: pd.DataFrame, base_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        simplified_map = {col: self.simplify_header(col) for col in df.columns}
        base_simple = self.simplify_header(base_name)
        cols = {'x': None, 'y': None, 'z': None}
        
        for component in ['x', 'y', 'z']:
            # More robust pattern: case-insensitive, allows for standard suffixes like 't' or 'i'.
            pattern = re.compile(rf'^{re.escape(base_simple)}{component}[a-z0-9]*$', re.IGNORECASE)
            
            # Find a column that matches this pattern
            match_found = False
            for original_col, simplified_col in simplified_map.items():
                if pattern.match(simplified_col):
                    cols[component] = original_col
                    match_found = True
                    break # Take the first match
            if not match_found:
                self.log(f"Component search failed for base '{base_name}', component '{component}' with pattern '{pattern.pattern}'")

        return cols['x'], cols['y'], cols['z']
    
    def get_unit_with_fallback(self, column_name: str, units_map_csv: Dict[str, str]) -> str:
        u = units_map_csv.get(column_name, "");
        if u: return u
        METADATA_UNITS = {"time": "s", "twrbsfxt": "kN", "twrbsmxt": "kN-m", "ptfmroll": "deg", "ptfmsurge": "m", "hydrofxi": "N", "hydromxi": "N-m", "fairten1x": "N"}
        key = self.simplify_header(column_name)
        for k_meta, v_meta in METADATA_UNITS.items():
            if key.startswith(k_meta[:-1]) and key[-1] in 'xyz': return v_meta
            if key == k_meta: return v_meta
        return ""

    def compute_stats_after_threshold(self, t: pd.Series, y: pd.Series, t0: float) -> Tuple[float, float, float, float, float, bool]:
        t_num, y_num = pd.to_numeric(t, errors='coerce'), pd.to_numeric(y, errors='coerce')
        mask = (t_num >= t0) & y_num.notna()
        series, t_series, used_tail = (y_num[mask], t_num[mask], True) if mask.any() else (y_num[y_num.notna()], t_num[y_num.notna()], False)
        if series.empty: return (np.nan,) * 5 + (used_tail,)
        mean_val, idx_min, idx_max = series.mean(), series.idxmin(), series.idxmax()
        return mean_val, series.loc[idx_min], series.loc[idx_max], float(t_series.loc[idx_min]), float(t_series.loc[idx_max]), used_tail

    def draw_stats_for_series(self, ax: plt.Axes, color: str, time_col: str, df: pd.DataFrame, y_col: str, mean_start: float, time_unit: str, y_unit: str, label_prefix: str, always_minmax: bool, minmax_range_frac: float, minmax_abs: float):
        def format_eng(val: float) -> str: return f"{val:.3e}" if pd.notna(val) and (abs(val) >= 1e4 or (0 < abs(val) < 1e-2)) else (f"{val:.4f}" if pd.notna(val) else "N/A")
        def annotate_at_y_axis(ax: plt.Axes, y_value: float, text: str):
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            ax.annotate(text, xy=(0.0, y_value), xycoords=trans, xytext=(4, 0), textcoords='offset points', va='center', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))

        mean_val, ymin, ymax, t_min, t_max, used_tail = self.compute_stats_after_threshold(df[time_col], df[y_col], mean_start)
        desc_base = f"≥ {mean_start:g}{' '+time_unit if time_unit else ''}" if used_tail else "all data"
        mean_text = f"{format_eng(mean_val)}{' '+y_unit if y_unit else ''}"
        ax.axhline(y=mean_val, color=color, linestyle='--', linewidth=1.3, label=f"{label_prefix} -- Mean ({desc_base}): {mean_text}")
        annotate_at_y_axis(ax, mean_val, mean_text)
        rng = float(ymax - ymin) if pd.notna(ymax) and pd.notna(ymin) else np.nan
        show_minmax = always_minmax or (pd.notna(rng) and rng >= max(minmax_range_frac * max(abs(mean_val), 1e-12), minmax_abs))
        if show_minmax and pd.notna(ymin) and pd.notna(ymax):
            ax.axhline(y=ymin, color=color, linestyle=':', linewidth=1.2, label=f"{label_prefix} -- Min at t={t_min:.2f}: {format_eng(ymin)}")
            annotate_at_y_axis(ax, ymin, format_eng(ymin))
            ax.axhline(y=ymax, color=color, linestyle='-.', linewidth=1.2, label=f"{label_prefix} -- Max at t={t_max:.2f}: {format_eng(ymax)}")
            annotate_at_y_axis(ax, ymax, format_eng(ymax))

    def plot_group(self, time_col: str, df: pd.DataFrame, series_cols: List[str], series_labels: Dict[str, str], series_units: Dict[str, str], group_title: str, x_label: str, y_unit_hint: Optional[str], mean_start: float, time_unit: str, case_suffix: str, output_dir: str, file_stub: str, **kwargs):
        units = [series_units.get(c, "") for c in series_cols]
        chosen_unit = y_unit_hint or next((u for u in units if u), "")
        if any((u and chosen_unit and u != chosen_unit) for u in units): self.log(f"Warning: Mixed units in group '{group_title}'. Using '{chosen_unit}'.")
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in series_cols:
            if col not in df.columns: self.log(f"Warning: Column '{col}' missing for group '{group_title}'."); continue
            label = series_labels.get(col, self.strip_units(col))
            (line_handle,) = ax.plot(df[time_col], df[col], label=label, linewidth=1.4)
            self.draw_stats_for_series(ax, line_handle.get_color(), time_col, df, col, mean_start, time_unit, series_units.get(col, ""), label, **kwargs)
        ax.set_title(f"{group_title} vs. {self.strip_units(time_col)}{case_suffix}", fontsize=16); ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f"{group_title}{f' [{chosen_unit}]' if chosen_unit else ''}", fontsize=12); ax.grid(True)
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, framealpha=0.9)
        plt.tight_layout()
        save_path = os.path.join(output_dir, re.sub(r'[\\/*?:"<>|()\s]', "", str(file_stub)) + '.png')
        try: plt.savefig(save_path, dpi=150, bbox_inches='tight', bbox_extra_artists=(legend,)); self.log(f"Saved group plot: '{Path(save_path).name}'")
        except Exception as e: self.log(f"Error saving group plot '{group_title}': {e}")
        plt.close(fig)

    def run(self, csv_file: str, output_dir: str, case_name: Optional[str] = None, mean_start: float = 300.0, **kwargs):
        if not MATPLOTLIB_AVAILABLE: self.log("Matplotlib not found, skipping plotting."); return
        self.log(f"Reading data from '{Path(csv_file).name}'...");
        try: 
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
        except Exception as e: self.log(f"Error reading CSV file: {e}"); return
        time_col = self.find_time_column(df)
        if not time_col: self.log("Error: No 'Time' column found."); return

        csv_units_map = self.build_units_map_from_csv(list(df.columns))
        series_label, series_unit = {}, {}
        time_unit = self.get_unit_with_fallback(time_col, csv_units_map)
        x_label = f"{self.strip_units(time_col)}{f' [{time_unit}]' if time_unit else ''}"
        vector_bases = ['TwrBsF', 'TwrBsM', 'HydroF', 'HydroM', 'FAIRTEN1', 'FAIRTEN2', 'FAIRTEN3']
        scalar_channels = ['PtfmRoll', 'PtfmPitch', 'PtfmYaw', 'PtfmSurge', 'PtfmSway', 'PtfmHeave']
        channels_to_plot, fairten_mag_cols, found_scalars = [], {}, {}

        for base in vector_bases:
            self.log(f"Searching for vector components for base: '{base}'")
            x_col, y_col, z_col = self.find_vector_columns(df, base)
            self.log(f"  -> Found components: X='{x_col}', Y='{y_col}', Z='{z_col}'")
            
            if all([x_col, y_col, z_col]):
                try:
                    mag_vals = np.sqrt(pd.to_numeric(df[x_col],errors='coerce')**2 + pd.to_numeric(df[y_col],errors='coerce')**2 + pd.to_numeric(df[z_col],errors='coerce')**2)
                    mag_col = f"{base}_Magnitude"; df[mag_col] = mag_vals; channels_to_plot.append(mag_col)
                    series_label[mag_col], series_unit[mag_col] = mag_col, self.get_unit_with_fallback(x_col, csv_units_map)
                    if base.upper().startswith("FAIRTEN"): fairten_mag_cols[base.upper()] = mag_col
                except Exception as e: self.log(f"Warning: Failed to compute magnitude for '{base}': {e}")
            else:
                self.log(f"  -> FAILED to find all three components for '{base}'. Skipping magnitude calculation.")

        for channel in scalar_channels:
            matches = [c for c in df.columns if str(c).strip().lower().startswith(channel.lower())]
            if matches:
                col = matches[0]; channels_to_plot.append(col)
                series_label[col], series_unit[col] = self.strip_units(col), self.get_unit_with_fallback(col, csv_units_map)
                found_scalars[channel] = col

        if not channels_to_plot: self.log("No channels were found to plot."); return
        os.makedirs(output_dir, exist_ok=True); self.log(f"Generating plots...")
        plt.style.use('ggplot'); case_suffix = f" -- {case_name}" if case_name else ""

        for channel in channels_to_plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            try: (h,) = ax.plot(df[time_col], df[channel], label=series_label.get(channel, self.strip_units(channel)))
            except Exception as e: self.log(f"Error plotting channel '{channel}': {e}"); plt.close(fig); continue
            y_unit = series_unit.get(channel, "")
            ax.set_title(f'{series_label.get(channel, self.strip_units(channel))} vs. {self.strip_units(time_col)}{case_suffix}', fontsize=16)
            ax.set_xlabel(x_label, fontsize=12); ax.set_ylabel(f"{series_label.get(channel, self.strip_units(channel))}{f' [{y_unit}]' if y_unit else ''}", fontsize=12)
            self.draw_stats_for_series(ax, h.get_color(), time_col, df, channel, mean_start, time_unit, y_unit, series_label.get(channel, self.strip_units(channel)), **kwargs)
            ax.legend(); ax.grid(True); plt.tight_layout()
            save_path = os.path.join(output_dir, re.sub(r'[\\/*?:"<>|()\s]', "", series_label.get(channel, channel)) + '.png')
            try: plt.savefig(save_path, dpi=150)
            except Exception as e: self.log(f"Error saving plot for '{channel}': {e}")
            plt.close(fig)

        rpy_cols = [c for c in [found_scalars.get(k) for k in ['PtfmRoll', 'PtfmPitch', 'PtfmYaw']] if c]
        if rpy_cols: self.plot_group(time_col, df, rpy_cols, series_label, series_unit, "Platform Roll/Pitch/Yaw", x_label, "deg", mean_start, time_unit, case_suffix, output_dir, "Ptfm_RollPitchYaw", **kwargs)
        ssh_cols = [c for c in [found_scalars.get(k) for k in ['PtfmSurge', 'PtfmSway', 'PtfmHeave']] if c]
        if ssh_cols: self.plot_group(time_col, df, ssh_cols, series_label, series_unit, "Platform Surge/Sway/Heave", x_label, "m", mean_start, time_unit, case_suffix, output_dir, "Ptfm_SurgeSwayHeave", **kwargs)
        fair_cols = [c for c in [fairten_mag_cols.get(k) for k in ['FAIRTEN1', 'FAIRTEN2', 'FAIRTEN3']] if c]
        if fair_cols: self.plot_group(time_col, df, fair_cols, series_label, series_unit, "Fairlead Tension Magnitudes", x_label, "N", mean_start, time_unit, case_suffix, output_dir, "FAIRTEN_Magnitudes", **kwargs)
        self.log("Plotting complete.")

# #############################################################################
# --- END: Code from plot_analysis_unnormalized.txt ---
# #############################################################################

# #############################################################################
# --- BEGIN: NEW d'Alembert Implementation ---
# #############################################################################
class DalembertLogHandler(logging.Handler):
    def __init__(self, message_queue, case_name, log_type):
        super().__init__()
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type
    def emit(self, record):
        self.mq.put((self.log_type, f"[{self.case_name}][Dalembert] {self.format(record)}"))

class DalembertRunner:
    def __init__(self, message_queue: queue.Queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"dalembert_{self.case_name}_{id(self)}")
        logger.setLevel(logging.DEBUG) # Always capture debug, GUI can filter
        if not logger.handlers:
            handler = DalembertLogHandler(self.mq, self.case_name, self.log_type)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def run(self, fst: str, glue_out: str, outdir: str, analysis_start_time: float, **kwargs):
        self.logger.info("========== d'Alembert staticization: START ==========")
        try:
            # Create an args-like object from kwargs
            class Args:
                def __init__(self, d):
                    self.__dict__.update(d)
            
            args_dict = {
                'fst': fst, 'glue_out': glue_out, 'outdir': outdir,
                'outb': kwargs.get('outb', False),
                'moordyn_out': kwargs.get('moordyn_out'),
                'rotate_ed': kwargs.get('rotate_ed', True),
                'override_mass': kwargs.get('override_mass'),
                'override_com': kwargs.get('override_com'),
                'override_inertia': kwargs.get('override_inertia'),
                'verbose': True, # For full logging to GUI
                'log_step': kwargs.get('log_step', 100)
            }
            args = Args(args_dict)

            self.logger.info("Arguments: " + json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
            os.makedirs(args.outdir, exist_ok=True)

            builder = self.MassPropertyBuilder(args.fst, self.logger)
            auto_m, auto_com, auto_Icom = builder.compute()
            
            m = args.override_mass if args.override_mass is not None else auto_m
            r_com = np.array(args.override_com, float) if args.override_com is not None else auto_com
            Ixx, Iyy, Izz, Ixy, Ixz, Iyz = (args.override_inertia if args.override_inertia is not None else (0,0,0,0,0,0))
            I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]], float) if args.override_inertia is not None else auto_Icom

            self.logger.info(f"Mass properties in use: m={m:.6e} kg, CoM={r_com.tolist()}")

            refs = _find_fst_refs(args.fst, self.logger)
            geo = self._parse_elastodyn_geometry(refs['EDFile'])
            fairleads = self._parse_moordyn_points(refs['MooringFile'])
            PRP, yaw_xyz, twrbase_xyz = np.zeros(3), geo['YawBearing'], geo['TowerBase']

            df = self._parse_glue_text(args.glue_out)
            df = self._collapse_dupes(df)

            df_md = self._parse_glue_text(args.moordyn_out) if args.moordyn_out else None
            if df_md is not None and 'time' in df_md.columns:
                df_md = df_md.set_index('time')

            self._perform_dalembert_calculations(df, df_md, args, m, r_com, I, PRP, yaw_xyz, twrbase_xyz, fairleads, analysis_start_time)

        except Exception as e:
            self.logger.error(f"FATAL ERROR in d'Alembert analysis: {e}\n{traceback.format_exc()}")
        finally:
            self.logger.info("========== d'Alembert staticization: END ==========")
    
    def _perform_dalembert_calculations(self, df, df_md, args, m, r_com, I, PRP, yaw_xyz, twrbase_xyz, fairleads, analysis_start_time):
        hydro_cols=['hydrofxi','hydrofyi','hydrofzi','hydromxi','hydromyi','hydromzi']
        
        if all(c in df.columns for c in ['twrbsfxt','twrbsfyt','twrbsfzt','twrbsmxt','twrbsmyt','twrbsmzt']):
            edF, edM, ed_point, ed_name = ['twrbsfxt','twrbsfyt','twrbsfzt'], ['twrbsmxt','twrbsmyt','twrbsmzt'], twrbase_xyz, 'ed_towerbase_interface'
            self.logger.info("Using ED interface at Tower Base (TwrBs*) in platform axes")
        elif all(c in df.columns for c in ['yawbrfxp','yawbrfyp','yawbrfzp','yawbrmxp','yawbrmyp','yawbrmzp']):
            edF, edM, ed_point, ed_name = ['yawbrfxp','yawbrfyp','yawbrfzp'], ['yawbrmxp','yawbrmyp','yawbrmzp'], yaw_xyz, 'ed_yawbr_interface'
            self.logger.info("Using ED interface at Yaw Bearing (YawBr*) in platform axes")
        else:
            raise RuntimeError('Missing ED interface loads (TwrBs* or YawBr*).')

        rows=[]
        n=len(df)
        self.logger.info(f"Beginning time loop over {n} rows")
        
        for i in range(n):
            row=df.iloc[i]
            t=row['time']

            H_F=row[hydro_cols[:3]].values
            H_M=row[hydro_cols[3:]].values

            ED_F_loc=row[edF].values
            ED_M_loc=row[edM].values

            R = self._rotmat_from_rpy_deg(row['ptfmroll'], row['ptfmpitch'], row['ptfmyaw']) if args.rotate_ed else np.eye(3)
            ED_F = R @ ED_F_loc
            ED_M = R @ ED_M_loc
            ED_M_at_PRP = ED_M + np.cross((PRP - ed_point), ED_F)

            Moor_F=np.zeros(3); Moor_M=np.zeros(3); fair_entries=[]
            for k, rk in fairleads:
                Fk = np.zeros(3)
                md_cols = [f'fairten{k}x', f'fairten{k}y', f'fairten{k}z']
                if all(c in row.index for c in md_cols):
                    Fk = row[md_cols].values
                elif df_md is not None and all(c in df_md.columns for c in md_cols):
                    idx = (df_md.index - t).abs().argmin()
                    Fk = df_md.iloc[idx][md_cols].values
                Moor_F += Fk; Moor_M += np.cross(rk, Fk); fair_entries.append((k, rk, Fk))

            F_ext = H_F + Moor_F + ED_F
            M_ext_at_PRP = H_M + Moor_M + ED_M_at_PRP
            F_inert = -F_ext
            M_inert = -M_ext_at_PRP + np.cross((r_com - PRP), F_ext)

            def add(name, F, P, Mv=None):
                rows.append({'Time':t, 'LoadName':name, 'Px':P[0],'Py':P[1],'Pz':P[2], 'Fx':F[0],'Fy':F[1],'Fz':F[2], 'Mx':Mv[0] if Mv is not None else 0, 'My':Mv[1] if Mv is not None else 0, 'Mz':Mv[2] if Mv is not None else 0, 'F_norm':self._vnorm(F), 'M_norm':self._vnorm(Mv) if Mv is not None else 0})
            
            add('HydroDyn_Total_at_PRP', H_F, PRP, H_M)
            add(ed_name, ED_F, ed_point, ED_M)
            for k, rk, Fk in fair_entries: add(f'MoorDyn_Fairlead{k}', Fk, rk, None)
            add('Inertia_Trans_CoM', F_inert, r_com, None)
            add('Inertia_Rot_CoM', np.zeros(3), r_com, M_inert)
            
            Tot_F = F_ext + F_inert
            Tot_M = M_ext_at_PRP + M_inert + np.cross((r_com - PRP), F_inert)
            add('TOTAL_with_Inertia_at_PRP', Tot_F, PRP, Tot_M)

        loads_df=pd.DataFrame(rows)
        loads_csv=os.path.join(args.outdir, 'loads_timeseries_staticized.csv')
        loads_df.to_csv(loads_csv, index=False)
        self.logger.info(f"Wrote timeseries loads: {Path(loads_csv).name}")
        self._write_reports(loads_df, args, geo, fairleads, m, r_com, I, analysis_start_time)

    def _write_reports(self, loads_df, args, geo, fairleads, m, r_com, I, analysis_start_time):
        extrema_lines=[]
        total = loads_df[(loads_df['LoadName']=='TOTAL_with_Inertia_at_PRP') & (loads_df['Time'] >= analysis_start_time)].copy()
        if total.empty:
            extrema_lines.append(f"No TOTAL_with_Inertia_at_PRP samples at or after {analysis_start_time:.2f}s; extrema unavailable.")
        else:
            Fmag = np.sqrt(total['Fx']**2 + total['Fy']**2 + total['Fz']**2)
            Mmag = np.sqrt(total['Mx']**2 + total['My']**2 + total['Mz']**2)
            
            def pick_extrema(series): return series.idxmin(), series.idxmax(), (series - series.mean()).abs().idxmin()
            F_min_i, F_max_i, F_avg_i = pick_extrema(Fmag)
            M_min_i, M_max_i, M_avg_i = pick_extrema(Mmag)
            
            cases = {'F_min': F_min_i, 'F_max': F_max_i, 'F_avg': F_avg_i, 'M_min': M_min_i, 'M_max': M_max_i, 'M_avg': M_avg_i}
            extrema_data = [{'Case': name, **total.loc[idx][['Time','Fx','Fy','Fz','Mx','My','Mz']]} for name, idx in cases.items()]
            extrema_df = pd.DataFrame(extrema_data)
            extrema_csv = os.path.join(args.outdir, f'loads_extrema_after{int(analysis_start_time)}s.csv')
            extrema_df.to_csv(extrema_csv, index=False)
            self.logger.info(f"Wrote extrema CSV: {Path(extrema_csv).name}")
            extrema_lines = extrema_df.to_string(index=False).split('\n')

        rep = [f"Staticized snapshot report (d'Alembert) for {self.case_name}", "="*40, ""]
        rep.append(f"Mass properties: m={m:.6e} kg, CoM=({r_com[0]:.3f}, {r_com[1]:.3f}, {r_com[2]:.3f}) m")
        rep.append("Inertia tensor (CoM, inertial axes):\n" + np.array2string(I, prefix='    '))
        rep.append(f"\nLoad extrema summary (t >= {analysis_start_time:.2f} s):\n" + "\n".join(extrema_lines))
        report_path=os.path.join(args.outdir,'staticized_report.txt')
        with open(report_path,'w') as f: f.write("\n".join(rep))
        self.logger.info(f"Wrote report: {Path(report_path).name}")

    def _parse_glue_text(self, path):
        if path is None: return None
        self.logger.info(f"Parsing glue (text) output: {path}")
        lines = _read_lines(path, self.logger)
        h = next((i for i, ln in enumerate(lines) if ln.strip().startswith('Time')), None)
        if h is None: raise RuntimeError(f"Header 'Time' not found in {path}")
        
        cols = lines[h].strip().split()
        data = []
        bad = 0
        # FIX: Start from h+2 to skip the units line, which caused the ValueError
        for ln in lines[h+2:]:
            s = ln.strip()
            if not s: continue
            parts = s.split()
            if len(parts) == len(cols):
                try:
                    data.append([float(x.replace('D', 'E')) for x in parts])
                except ValueError:
                    bad += 1
        if bad > 0: self.logger.warning(f"Skipped {bad} malformed data rows in {path}")
        
        df = pd.DataFrame(data, columns=[c.lower() for c in cols])
        self.logger.debug(f"Glue columns: {list(df.columns)}; rows={len(df)}")
        return df

    def _collapse_dupes(self, df):
        if len(df.columns) == len(set(df.columns)): return df
        out = {}
        for col in dict.fromkeys(df.columns):
            same = df.loc[:, df.columns == col]
            if same.shape[1] > 1:
                self.logger.debug(f"Collapsing duplicate column '{col}' by averaging {same.shape[1]} copies.")
                out[col] = same.apply(pd.to_numeric, errors='coerce').mean(axis=1)
            else:
                out[col] = pd.to_numeric(same.iloc[:, 0], errors='coerce')
        return pd.DataFrame(out)
    
    def _parse_elastodyn_geometry(self, ed_path):
        lines=_read_lines(ed_path, self.logger)
        def fget(key):
            for ln in lines:
                if key in ln:
                    try: return float(ln.strip().split()[0])
                    except: pass
            return None
        return {'YawBearing': np.array([0.,0.,fget('TowerHt') or 0.0]), 'TowerBase': np.array([0.,0.,fget('TowerBsHt') or 0.0])}

    def _parse_moordyn_points(self, md_path):
        self.logger.info(f"Parsing MoorDyn fairlead points: {md_path}")
        lines=_read_lines(md_path, self.logger)
        pts=[]; in_points=False; in_lines=False; line_defs=[]
        for s in lines:
            s=s.strip().upper()
            if s.startswith('---'):
                in_points = 'POINTS' in s
                in_lines = 'LINES' in s
            elif in_points:
                tok=s.split()
                if len(tok)>=5 and tok[0].isdigit(): pts.append((int(tok[0]),tok[1],float(tok[2]),float(tok[3]),float(tok[4])))
            elif in_lines:
                tok=s.split()
                if len(tok)>=7 and tok[0].isdigit(): line_defs.append((int(tok[0]),int(tok[2]),int(tok[3])))
        
        vessel={pid:(x,y,z) for pid,att,x,y,z in pts if att=='VESSEL'}
        fair_pids = sorted([a if a in vessel else b for _,a,b in line_defs if a in vessel or b in vessel])
        return [(i, np.array(vessel[pid])) for i, pid in enumerate(fair_pids, 1)]

    def _vnorm(self, v): return float(np.linalg.norm(np.asarray(v,float)))
    def _rotmat_from_rpy_deg(self, r,p,y):
        rz,ry,rx = radians(y),radians(p),radians(r); cz,sz=cos(rz),sin(rz); cy,sy=cos(ry),sin(ry); cx,sx=cos(rx),sin(rx)
        Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]); Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]); Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        return Rz@Ry@Rx

    class MassPropertyBuilder:
        def __init__(self, fst_path, logger):
            self.logger = logger
            self.fst_path = fst_path
            # MODIFIED: Call standalone helper function, passing the valid logger
            self.refs = _find_fst_refs(fst_path, self.logger)
            self.ed_path = self.refs.get('EDFile')
            if not self.ed_path or not os.path.isfile(self.ed_path): raise RuntimeError('ElastoDyn file not found from FST.')
            self.ed = self._parse_elastodyn(self.ed_path)
            self.twr_path = self._find_tower_file_from_ed()
            if not self.twr_path or not os.path.isfile(self.twr_path): raise RuntimeError('ElastoDyn tower properties file not found.')
            self.tower_dist = self._parse_dist_prop(self.twr_path, 'HtFract', 'TMassDen')
            self.blade_mass_path = self._find_ed_blade_mass_file()
            if not self.blade_mass_path or not os.path.isfile(self.blade_mass_path): raise RuntimeError('ElastoDyn blade mass file not found.')
            self.bl_mass_dist = self._parse_dist_prop(self.blade_mass_path, 'BlFract', 'BMassDen', 3)
        
        def _read_lines(self, p): return _read_lines(p, self.logger)
        def _strip_quotes(self, s): return _strip_quotes(s)
        def _parse_kv_float(self, lines, key): return next((float(ln.strip().split()[0]) for ln in lines if key in ln), None)

        def _parse_elastodyn(self, ed_path):
            lines=self._read_lines(ed_path); kv=lambda k: self._parse_kv_float(lines,k); NumBl=int(kv('NumBl') or 3)
            return {'TowerHt':kv('TowerHt') or 0.0, 'TowerBsHt':kv('TowerBsHt') or 0.0, 'PtfmMass':kv('PtfmMass') or 0.0, 'PtfmI':(kv('PtfmRIner')or 0, kv('PtfmPIner')or 0, kv('PtfmYIner')or 0, kv('PtfmXYIner')or 0, kv('PtfmXZIner')or 0, kv('PtfmYZIner')or 0), 'PtfmCM':np.array([kv('PtfmCMxt')or 0, kv('PtfmCMyt')or 0, kv('PtfmCMzt')or 0]), 'NacMass':kv('NacMass') or 0.0, 'NacYIner':kv('NacYIner') or 0.0, 'NacCMn':np.array([kv('NacCMxn')or 0, kv('NacCMyn')or 0, kv('NacCMzn')or 0]), 'HubMass':kv('HubMass') or 0.0, 'HubIner':kv('HubIner') or 0.0, 'NumBl':NumBl, 'TipRad':kv('TipRad') or 0.0, 'HubRad':kv('HubRad') or 0.0, 'PreCone':[kv(f'PreCone({i})') or 0.0 for i in range(1,NumBl+1)], 'OverHang':kv('OverHang') or 0.0, 'ShftTilt':kv('ShftTilt') or 0.0, 'Twr2Shft':kv('Twr2Shft') or 0.0}

        def _find_tower_file_from_ed(self):
            base=os.path.dirname(os.path.abspath(self.ed_path))
            return next((os.path.normpath(os.path.join(base, self._strip_quotes(ln.split()[0]))) for ln in self._read_lines(self.ed_path) if 'TwrFile' in ln), self.refs.get('TwrFile'))

        def _find_ed_blade_mass_file(self):
            base=os.path.dirname(os.path.abspath(self.ed_path))
            return next((os.path.normpath(os.path.join(base, self._strip_quotes(ln.split()[0]))) for ln in self._read_lines(self.ed_path) if 'BldFile(1)' in ln or ('BldFile' in ln and 'ADBlFile' not in ln)), None)

        def _parse_dist_prop(self, path, key1, key2, val_idx=1):
            if not path or not os.path.isfile(path): self.logger.warning(f"Dist prop file not found: {path}"); return []
            lines=self._read_lines(path); data, started = [], False
            for ln in lines:
                s=ln.strip()
                if not s or s.startswith('!'): continue
                if key1 in s and key2 in s: started=True; continue
                if started:
                    parts=s.split()
                    if len(parts)>max(0,val_idx):
                        try: data.append((float(parts[0]), float(parts[val_idx])))
                        except: pass
            return sorted(data, key=lambda t:t[0])

        @staticmethod
        def parallel_axis(Ic, m, r): r=np.asarray(r).reshape(3); return Ic + m*((r@r)*np.eye(3) - np.outer(r,r))
        
        def compute(self):
            self.logger.info("Computing mass properties..."); Ms, Rs, Is = [], [], []
            m_ptfm, r_ptfm, I_ptfm = self.ed['PtfmMass'], self.ed['PtfmCM'], np.array([[self.ed['PtfmI'][0], self.ed['PtfmI'][3], self.ed['PtfmI'][4]], [self.ed['PtfmI'][3], self.ed['PtfmI'][1], self.ed['PtfmI'][5]], [self.ed['PtfmI'][4], self.ed['PtfmI'][5], self.ed['PtfmI'][2]]])
            Ms.append(m_ptfm); Rs.append(r_ptfm); Is.append(I_ptfm)
            Mt, Rt, It = self._tower_mass_properties(); Ms.extend(Mt); Rs.extend(Rt); Is.extend(It)
            m_nac, r_nac, I_nac = self.ed['NacMass'], np.array([0,0,self.ed['TowerHt']]) + self.ed['NacCMn'], np.diag([0.0, self.ed['NacYIner'], 0.0])
            Ms.append(m_nac); Rs.append(r_nac); Is.append(I_nac)
            r_hub, R_rotor = np.array([0,0,self.ed['TowerHt']]) + np.array([self.ed['OverHang'], 0, self.ed['Twr2Shft']]), DalembertRunner(None,None,None)._rotmat_from_rpy_deg(0, self.ed['ShftTilt'], 0)
            m_hub, I_hub = self.ed['HubMass'], R_rotor @ np.diag([0, self.ed['HubIner'], 0]) @ R_rotor.T
            Ms.append(m_hub); Rs.append(r_hub); Is.append(I_hub)
            Mb, Rb, Ib = self._blades_mass_properties(r_hub, R_rotor); Ms.extend(Mb); Rs.extend(Rb); Is.extend(Ib)
            Mtot = float(np.sum(Ms)); r_com = np.sum([m*np.asarray(r) for m,r in zip(Ms,Rs)], axis=0) / max(Mtot, 1e-16)
            I_origin = np.sum([self.parallel_axis(Ic, m, r) for m,r,Ic in zip(Ms,Rs,Is)], axis=0)
            I_com = I_origin - self.parallel_axis(np.zeros((3,3)), Mtot, r_com)
            return Mtot, r_com, I_com

        def _tower_mass_properties(self):
            z0, zTop, H = self.ed['TowerBsHt'], self.ed['TowerHt'], self.ed['TowerHt'] - self.ed['TowerBsHt']
            if H<=0: return [],[],[]
            z_list, md_list = [z0 + hf*H for hf,_ in self.tower_dist], [md for _,md in self.tower_dist]; Ms, Rs, Is = [], [], []
            for i in range(len(z_list)-1):
                L = z_list[i+1] - z_list[i];
                if L <= 0: continue
                m_seg = 0.5*(md_list[i]+md_list[i+1]) * L; r = np.array([0,0, 0.5*(z_list[i]+z_list[i+1])])
                Ic = np.diag([(1/12)*m_seg*L**2, (1/12)*m_seg*L**2, 0]); Ms.append(m_seg); Rs.append(r); Is.append(Ic)
            return Ms, Rs, Is

        def _blades_mass_properties(self, r_hub, R_rotor):
            NumBl, R_root, L_blade = self.ed['NumBl'], self.ed['HubRad'], self.ed['TipRad'] - self.ed['HubRad']
            bl_fracs, bl_mdens = [f for f,_ in self.bl_mass_dist], [md for _,md in self.bl_mass_dist]
            def mass_den_at_r(r): return np.interp(max(0, min(1, (r - R_root)/L_blade)), bl_fracs, bl_mdens) if bl_fracs else 0
            Ms, Rs, Is = [], [], []
            for ib, az in enumerate([i*360.0/NumBl for i in range(NumBl)]):
                r=radians(az); R_az = np.array([[cos(r),0,sin(r)],[0,1,0],[-sin(r),0,cos(r)]])
                R_cone = DalembertRunner(None,None,None)._rotmat_from_rpy_deg(0, self.ed['PreCone'][ib], 0)
                R_blade = R_rotor @ R_az @ R_cone
                spans = sorted(list(np.linspace(R_root, self.ed['TipRad'], 21)))
                for i in range(len(spans)-1):
                    r1, r2 = spans[i], spans[i+1]; L = r2-r1; m_seg = 0.5*(mass_den_at_r(r1)+mass_den_at_r(r2)) * L
                    r_global = r_hub + (0.5*(r1+r2)) * (R_blade @ np.array([1,0,0]))
                    Ic_local = np.diag([0, (1/12)*m_seg*L**2, (1/12)*m_seg*L**2])
                    Ms.append(m_seg); Rs.append(r_global); Is.append(R_blade @ Ic_local @ R_blade.T)
            return Ms, Rs, Is

# #############################################################################
# --- END: NEW d'Alembert Implementation ---
# #############################################################################


# #############################################################################
# --- Main GUI Application Class ---
# #############################################################################

class OpenFASTTestCaseGUI:
    def __init__(self, root):
        self.root = root; self.root.title("OpenFAST Test Case Workflow Manager"); self.root.geometry("1200x850")
        style = ttk.Style(); style.theme_use('clam'); style.configure("Accent.TButton", foreground="white", background="#0078D7"); style.map("Accent.TButton", background=[('active', '#005A9E')])
        self.notebook = ttk.Notebook(root); self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        self.setup_tab, self.run_tab, self.post_proc_tab = ttk.Frame(self.notebook), ttk.Frame(self.notebook), ttk.Frame(self.notebook)
        self.notebook.add(self.setup_tab, text="1. Setup Cases"); self.notebook.add(self.run_tab, text="2. Run Simulations"); self.notebook.add(self.post_proc_tab, text="3. Post-Process Results")
        self.tutorial_tab = ttk.Frame(self.notebook)
        self.notebook.insert(0, self.tutorial_tab, text="Tutorial") # Insert as the first tab
        self.base_fst_path, self.output_dir, self.openfast_exe = tk.StringVar(), tk.StringVar(value=str(Path.cwd() / "test_cases")), tk.StringVar()
        self.discovered_parameters, self.file_structure, self.message_queue = {}, {}, queue.Queue()
        self.num_threads = tk.IntVar(value=max(1, os.cpu_count() // 2))
        self.num_cases, self.parameter_entries = tk.IntVar(value=10), []
        self.run_button, self.run_job_queue, self.run_progress_lock, self.run_completed_cases, self.run_total_cases, self.run_cases = None, queue.Queue(), threading.Lock(), 0, 0, {}
        self.post_proc_button, self.post_proc_job_queue, self.post_proc_progress_lock, self.post_proc_completed_cases, self.post_proc_total_cases, self.post_proc_cases = None, queue.Queue(), threading.Lock(), 0, 0, {}
        self.run_convert_csv, self.run_dalembert, self.run_plotting = tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.geom_vary_height_enabled = tk.BooleanVar(value=False)
        self.geom_height_start_scale = tk.DoubleVar(value=0.9)
        self.geom_height_end_scale = tk.DoubleVar(value=1.1)
        self.geom_height_steps = tk.IntVar(value=3)

        self.geom_vary_diam_enabled = tk.BooleanVar(value=False)
        self.geom_diam_start_scale = tk.DoubleVar(value=0.9)
        self.geom_diam_end_scale = tk.DoubleVar(value=1.1)
        self.geom_diam_steps = tk.IntVar(value=3)
        # FIX: Add a dedicated lock for plotting to prevent multithreading issues
        self.plotting_lock = threading.Lock()

        self.create_tutorial_tab();self.create_setup_tab(); self.create_run_tab(); self.create_post_proc_tab(); self.process_queue()
        
    def create_setup_tab(self):
        canvas = tk.Canvas(self.setup_tab); scrollbar = ttk.Scrollbar(self.setup_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas); scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        self.create_file_selection_section(scrollable_frame); self.create_test_config_section(scrollable_frame)
        self.create_geometry_section(scrollable_frame)
        self.create_parameter_discovery_section(scrollable_frame); self.create_parameter_section(scrollable_frame)
        self.create_action_section(scrollable_frame); self.create_log_section(scrollable_frame, "setup_log")
        
    def create_run_tab(self):
        main_frame = ttk.Frame(self.run_tab, padding="10"); main_frame.pack(fill='both', expand=True)
        config_frame = ttk.LabelFrame(main_frame, text="Run Configuration", padding="10"); config_frame.pack(fill='x', pady=5)
        ttk.Label(config_frame, text="OpenFAST Path:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(config_frame, textvariable=self.openfast_exe, width=50).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_openfast_exe).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(config_frame, text="Parallel runs:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Spinbox(config_frame, from_=1, to=os.cpu_count() or 8, textvariable=self.num_threads, width=8).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        config_frame.columnconfigure(1, weight=1)
        case_frame = ttk.LabelFrame(main_frame, text="Test Cases to Run", padding="10"); case_frame.pack(fill='both', expand=True, pady=5)
        btn_frame = ttk.Frame(case_frame); btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Load Test Cases", command=self.load_run_cases).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Select All", command=lambda: self.select_all_cases(self.run_tree)).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Deselect All", command=lambda: self.deselect_all_cases(self.run_tree)).pack(side='left', padx=5)
        self.run_button = ttk.Button(btn_frame, text="Run Selected Simulations", command=self.run_selected_cases, style="Accent.TButton"); self.run_button.pack(side='left', padx=20)
        list_frame = ttk.Frame(case_frame); list_frame.pack(fill='both', expand=True)
        columns = ('Status', 'Parameters', 'Runtime', 'Result'); self.run_tree = ttk.Treeview(list_frame, columns=columns, show='headings', selectmode='extended')
        self.run_tree.heading('#0', text='Test Case'); self.run_tree.column('#0', width=150, anchor='w')
        for col in columns: self.run_tree.heading(col, text=col)
        self.run_tree.column('Status', width=180); self.run_tree.column('Parameters', width=300); self.run_tree.column('Runtime', width=100, anchor='center'); self.run_tree.column('Result', width=200)
        tree_scroll_y = ttk.Scrollbar(list_frame, orient="vertical", command=self.run_tree.yview); tree_scroll_x = ttk.Scrollbar(list_frame, orient="horizontal", command=self.run_tree.xview)
        self.run_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.run_tree.grid(row=0, column=0, sticky='nsew'); tree_scroll_y.grid(row=0, column=1, sticky='ns'); tree_scroll_x.grid(row=1, column=0, sticky='ew')
        list_frame.grid_rowconfigure(0, weight=1); list_frame.grid_columnconfigure(0, weight=1)
        self.run_tree.bind("<Button-3>", lambda e: self.show_case_context_menu(e, self.run_tree, self.run_cases))
        self.run_progress_var = tk.DoubleVar(); self.run_progress_bar = ttk.Progressbar(main_frame, variable=self.run_progress_var, maximum=100); self.run_progress_bar.pack(fill='x', pady=5)
        self.create_log_section(main_frame, "run_log", "Execution Log")

    def create_post_proc_tab(self):
        main_frame = ttk.Frame(self.post_proc_tab, padding="10"); main_frame.pack(fill='both', expand=True)
        top_frame = ttk.Frame(main_frame); top_frame.pack(fill='x', pady=5)
        config_frame = ttk.LabelFrame(top_frame, text="Configuration", padding="10"); config_frame.pack(fill='x', expand=True, side='left', padx=(0, 5))
        ttk.Label(config_frame, text="Results Directory:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(config_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=5, pady=2)
        config_frame.columnconfigure(1, weight=1)
        post_proc_frame = ttk.LabelFrame(top_frame, text="Tasks to Run", padding="10"); post_proc_frame.pack(fill='x', side='left')
        ttk.Checkbutton(post_proc_frame, text="Convert .out to .csv", variable=self.run_convert_csv).pack(anchor='w')
        ttk.Checkbutton(post_proc_frame, text="Run d'Alembert Analysis", variable=self.run_dalembert).pack(anchor='w')
        ttk.Checkbutton(post_proc_frame, text="Generate Plots", variable=self.run_plotting).pack(anchor='w')
        case_frame = ttk.LabelFrame(main_frame, text="Cases to Process", padding="10"); case_frame.pack(fill='both', expand=True, pady=5)
        btn_frame = ttk.Frame(case_frame); btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Load Results", command=self.load_post_proc_cases).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Select All", command=lambda: self.select_all_cases(self.post_proc_tree)).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Deselect All", command=lambda: self.deselect_all_cases(self.post_proc_tree)).pack(side='left', padx=5)
        self.post_proc_button = ttk.Button(btn_frame, text="Run Post-Processing", command=self.run_selected_post_proc, style="Accent.TButton"); self.post_proc_button.pack(side='left', padx=20)
        list_frame = ttk.Frame(case_frame); list_frame.pack(fill='both', expand=True)
        columns = ('Status', 'Parameters', 'Result'); self.post_proc_tree = ttk.Treeview(list_frame, columns=columns, show='headings', selectmode='extended')
        self.post_proc_tree.heading('#0', text='Test Case'); self.post_proc_tree.column('#0', width=150, anchor='w')
        for col in columns: self.post_proc_tree.heading(col, text=col)
        self.post_proc_tree.column('Status', width=120); self.post_proc_tree.column('Parameters', width=400); self.post_proc_tree.column('Result', width=200)
        tree_scroll_y = ttk.Scrollbar(list_frame, orient="vertical", command=self.post_proc_tree.yview); tree_scroll_x = ttk.Scrollbar(list_frame, orient="horizontal", command=self.post_proc_tree.xview)
        self.post_proc_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.post_proc_tree.grid(row=0, column=0, sticky='nsew'); tree_scroll_y.grid(row=0, column=1, sticky='ns'); tree_scroll_x.grid(row=1, column=0, sticky='ew')
        list_frame.grid_rowconfigure(0, weight=1); list_frame.grid_columnconfigure(0, weight=1)
        self.post_proc_tree.bind("<Button-3>", lambda e: self.show_case_context_menu(e, self.post_proc_tree, self.post_proc_cases))
        self.post_proc_progress_var = tk.DoubleVar(); self.post_proc_progress_bar = ttk.Progressbar(main_frame, variable=self.post_proc_progress_var, maximum=100); self.post_proc_progress_bar.pack(fill='x', pady=5)
        self.create_log_section(main_frame, "post_proc_log", "Post-Processing Log")

    def create_file_selection_section(self, parent):
        frame = ttk.LabelFrame(parent, text="File Selection", padding="10"); frame.pack(fill='x', pady=5, padx=5)
        ttk.Label(frame, text="Base FST File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame, textvariable=self.base_fst_path, width=60).grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse", command=self.browse_fst_file).grid(row=0, column=2, padx=5)
        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.output_dir, width=60).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        
    def create_test_config_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Test Configuration", padding="10"); frame.pack(fill='x', pady=5, padx=5)
        ttk.Label(frame, text="Number of Test Cases:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_cases_spinbox = ttk.Spinbox(frame, from_=2, to=10000, textvariable=self.num_cases, width=10)
        self.num_cases_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(frame, text="Distribution Type:").grid(row=0, column=2, sticky=tk.W, padx=20)
        self.distribution_var = tk.StringVar(value="grid_search")
        dist_combo = ttk.Combobox(frame, textvariable=self.distribution_var, values=["grid_search", "csv_columnwise", "latin_hypercube", "uniform", "normal"], width=15)
        dist_combo.grid(row=0, column=3, sticky=tk.W, padx=5); dist_combo.bind("<<ComboboxSelected>>", self.on_distribution_change)
    # --- NEW: Method to create the Geometry UI ---
    def create_geometry_section(self, parent):
        """Creates the UI section for configuring geometric variations."""
        geom_frame = ttk.LabelFrame(parent, text="Geometric Variations (Automatic)", padding="10")
        geom_frame.pack(fill='x', pady=5, padx=5)

        if not GEOMETRY_ENGINE_AVAILABLE:
            ttk.Label(geom_frame, text="Engine not found. Please ensure 'advanced_geometry_engine.py' is in the same directory.", foreground="red").pack()
            return

        # --- Height Variation Sub-section ---
        height_frame = ttk.LabelFrame(geom_frame, text="Height Scaling", padding="5")
        height_frame.pack(fill='x', expand=True, pady=5)

        enable_height_check = ttk.Checkbutton(height_frame, text="Enable Height Variation", variable=self.geom_vary_height_enabled, command=self.update_total_cases)
        enable_height_check.grid(row=0, column=0, columnspan=6, sticky='w', pady=(0, 5))

        ttk.Label(height_frame, text="Start Scale:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(height_frame, textvariable=self.geom_height_start_scale, width=8).grid(row=1, column=1, sticky='w', padx=5)
        ttk.Label(height_frame, text="End Scale:").grid(row=1, column=2, sticky='w', padx=5)
        ttk.Entry(height_frame, textvariable=self.geom_height_end_scale, width=8).grid(row=1, column=3, sticky='w', padx=5)
        ttk.Label(height_frame, text="Steps:").grid(row=1, column=4, sticky='w', padx=5)
        ttk.Spinbox(height_frame, from_=1, to=100, textvariable=self.geom_height_steps, width=5).grid(row=1, column=5, sticky='w', padx=5)

        # --- Diameter Variation Sub-section ---
        diam_frame = ttk.LabelFrame(geom_frame, text="Diameter Scaling", padding="5")
        diam_frame.pack(fill='x', expand=True, pady=5)

        enable_diam_check = ttk.Checkbutton(diam_frame, text="Enable Diameter Variation", variable=self.geom_vary_diam_enabled, command=self.update_total_cases)
        enable_diam_check.grid(row=0, column=0, columnspan=6, sticky='w', pady=(0, 5))

        ttk.Label(diam_frame, text="Start Scale:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(diam_frame, textvariable=self.geom_diam_start_scale, width=8).grid(row=1, column=1, sticky='w', padx=5)
        ttk.Label(diam_frame, text="End Scale:").grid(row=1, column=2, sticky='w', padx=5)
        ttk.Entry(diam_frame, textvariable=self.geom_diam_end_scale, width=8).grid(row=1, column=3, sticky='w', padx=5)
        ttk.Label(diam_frame, text="Steps:").grid(row=1, column=4, sticky='w', padx=5)
        ttk.Spinbox(diam_frame, from_=1, to=100, textvariable=self.geom_diam_steps, width=5).grid(row=1, column=5, sticky='w', padx=5)

        # Trace changes to update case count automatically
        self.geom_vary_height_enabled.trace_add("write", self.update_total_cases)
        self.geom_height_steps.trace_add("write", self.update_total_cases)
        self.geom_vary_diam_enabled.trace_add("write", self.update_total_cases)
        self.geom_diam_steps.trace_add("write", self.update_total_cases)
    # --- END NEW ---
    def create_parameter_discovery_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameter Discovery", padding="10"); frame.pack(fill='x', pady=5, padx=5)
        ttk.Button(frame, text="Discover Parameters", command=self.discover_parameters, style="Accent.TButton").pack(side='left', padx=5)
        self.discovery_status = ttk.Label(frame, text="Select a .fst file and click 'Discover Parameters'"); self.discovery_status.pack(side='left', padx=20)
        
    def create_parameter_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameter Configuration", padding="10"); frame.pack(fill='both', expand=True, pady=5, padx=5)
        control_frame = ttk.Frame(frame); control_frame.pack(fill='x', pady=5)
        ttk.Button(control_frame, text="Add from Discovery", command=self.show_parameter_selector).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_parameters).pack(side='left', padx=5)
        canvas = tk.Canvas(frame, height=250); scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.param_list_frame = ttk.Frame(canvas); self.param_list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.param_list_frame, anchor="nw"); canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        
    def create_action_section(self, parent):
        frame = ttk.Frame(parent, padding="5"); frame.pack(fill='x', pady=10)
        ttk.Button(frame, text="Generate Test Cases", command=self.generate_test_cases, style="Accent.TButton").pack(side='left', padx=5)
        ttk.Button(frame, text="Load Configuration", command=self.load_config).pack(side='left', padx=5)
        ttk.Button(frame, text="Save Configuration", command=self.save_config).pack(side='left', padx=5)
        ttk.Button(frame, text="View File Structure", command=self.show_file_structure).pack(side='left', padx=5)
        
    def create_log_section(self, parent, log_attr_name, title="Output Log"):
        frame = ttk.LabelFrame(parent, text=title, padding="10"); frame.pack(fill='both', expand=True, pady=5, padx=5)
        log_widget = scrolledtext.ScrolledText(frame, height=8, wrap=tk.WORD, bg="#f0f0f0", relief="sunken", borderwidth=1)
        log_widget.pack(fill='both', expand=True)
        setattr(self, log_attr_name, log_widget)
    def _copy_and_rewrite_paths(self, source_path: Path, dest_path: Path):
        """
        Copies a file and intelligently rewrites any internal file paths to be
        relative to the new case directory. This is the definitive fix for the path issue.
        """
        if source_path.suffix.lower() not in ['.fst', '.dat', '.twr', '.bld', '.ipt', '.txt', '.in']:
            shutil.copy2(source_path, dest_path)
            return

        try:
            content = source_path.read_text(encoding='utf-8', errors='ignore')
            original_content = content
            
            # --- DEFINITIVE FIX: Make the file extension optional in the regex by adding a '?' ---
            # This allows it to match both full filenames and root names like "marin_semi".
            pattern = re.compile(r'(["\'])((?:[a-zA-Z]:)?[a-zA-Z0-9_.\-\s\\/]+(\.\w{2,4})?)\1')

            for match in pattern.finditer(content):
                full_match = match.group(0)
                path_inside = match.group(2)
                
                if "/" not in path_inside and "\\" not in path_inside:
                    continue

                new_basename = Path(path_inside).name
                quote_char = match.group(1)
                new_path_str = f"{quote_char}{new_basename}{quote_char}"
                
                content = content.replace(full_match, new_path_str)

            dest_path.write_text(content, encoding='utf-8')

            if content != original_content:
                self.log(f"    Rewrote internal paths in {dest_path.name}")
        except Exception as e:
            self.log(f"    Warning: Could not read/rewrite {source_path.name}. Copying directly. Error: {e}")
            shutil.copy2(source_path, dest_path)
    def resolve_file_path(self, base_dir, filename):
        if not filename or filename.lower() in ['unused', 'none', '']: return None
        filename = filename.strip('"').strip("'")
        paths_to_check = [ base_dir / Path(filename), Path(filename), base_dir / Path(filename).name, Path(self.base_fst_path.get()).parent / Path(filename) ]
        for path in paths_to_check:
            try:
                if path.exists(): return path.resolve()
            except: continue
        self.log(f"Warning: Could not find file: {filename}"); return None
        
    # --- DEFINITIVE FIX: Rewritten discovery function to handle root names ---
    def _discover_and_parse_files_recursively(self, file_path: Path, file_info_by_path: Dict[Path, Dict], processed_paths: set):
        """
        Recursively scans files to find all dependencies, adding them to the
        file_info_by_path dictionary. This version correctly handles OpenFAST's
        root name convention for file families like WAMIT data.
        """
        if not file_path or not file_path.exists() or file_path in processed_paths:
            return

        self.log(f"  Scanning: {file_path.name}")
        processed_paths.add(file_path)

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            if file_path not in file_info_by_path:
                file_info_by_path[file_path] = {
                    'key': file_path.name,
                    'original_strings': set(),
                    'params': {}
                }
            
            params = self.extract_parameters_from_file(content.splitlines())
            if params:
                file_info_by_path[file_path]['params'] = params

            # Use the corrected regex that makes the file extension optional
            pattern = re.compile(r'(["\'])((?:[a-zA-Z]:)?[a-zA-Z0-9_.\-\s\\/]+(\.\w{2,4})?)\1')
            
            for match in pattern.finditer(content):
                full_match_str = match.group(0)
                path_inside_quotes = match.group(2)
                
                if not path_inside_quotes or path_inside_quotes.lower() in ['default', 'unused', 'none']:
                    continue
                
                # Resolve path relative to the current file's directory
                resolved_path = (file_path.parent / path_inside_quotes).resolve()
                
                # Case 1: The path points directly to an existing file.
                if resolved_path.is_file():
                    if resolved_path not in file_info_by_path:
                        file_info_by_path[resolved_path] = {'key': resolved_path.name, 'original_strings': set(), 'params': {}}
                    file_info_by_path[resolved_path]['original_strings'].add(path_inside_quotes)
                    
                    if resolved_path not in processed_paths:
                        self._discover_and_parse_files_recursively(resolved_path, file_info_by_path, processed_paths)

                # Case 2: The path does NOT point to a file, but might be a root name (e.g., "marin_semi").
                else:
                    parent_dir = resolved_path.parent
                    root_name = resolved_path.name
                    
                    if parent_dir.is_dir():
                        found_family = False
                        # Find all files in the directory that start with this root name.
                        for item in parent_dir.iterdir():
                            if item.is_file() and item.name.startswith(root_name + '.'):
                                found_family = True
                                self.log(f"  [Discovery] Found root name family member: {item.name}")
                                
                                # Add the found file to the structure, but don't recurse into it.
                                if item not in file_info_by_path:
                                    file_info_by_path[item] = {'key': item.name, 'original_strings': set(), 'params': {}}
                                # Associate the original root name string with ALL family members found
                                file_info_by_path[item]['original_strings'].add(path_inside_quotes)

                        if not found_family:
                             self.log(f"  [Discovery] Warning: Could not resolve path or find root name family for '{path_inside_quotes}'")

        except Exception as e:
            self.log(f"Could not process file {file_path.name}: {e}\n{traceback.format_exc()}")
        
    def discover_parameters(self):
        if not self.base_fst_path.get():
            messagebox.showerror("Error", "Please select a base FST file first")
            return
        
        self.log("Starting deep parameter discovery...")
        self.discovery_status.config(text="Scanning all referenced files...")
        self.root.update()

        file_info_by_path = {}
        processed_paths = set()
        
        try:
            initial_fst_path = Path(self.base_fst_path.get())
            self._discover_and_parse_files_recursively(initial_fst_path, file_info_by_path, processed_paths)
            
            self.file_structure = {}
            self.discovered_parameters = {}
            
            final_keys = set()
            for path, info in file_info_by_path.items():
                key = info['key']
                if key in final_keys:
                    i = 1
                    while f"{path.stem}_{i}{path.suffix}" in final_keys:
                        i += 1
                    key = f"{path.stem}_{i}{path.suffix}"
                final_keys.add(key)
                
                self.file_structure[key] = {'path': path, 'original_strings': info['original_strings']}
                if info['params']:
                    self.discovered_parameters[key] = info['params']
            
            total_params = sum(len(p) for p in self.discovered_parameters.values())
            self.discovery_status.config(text=f"Discovered {total_params} parameters across {len(self.file_structure)} files.")
            self.log(f"Discovery complete: Found {len(self.file_structure)} total files.")

        except Exception as e:
            self.log(f"Error during parameter discovery: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to discover parameters: {str(e)}")
        
    def extract_parameters_from_file(self, lines: List[str]):
        parameters = {}
        param_pattern = re.compile(r'^\s*([^\s!#"]+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[-!]')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith(('!', '#', '-', '=')): continue
            
            if line_stripped.lower().startswith(('joint', 'member', 'node', 'station')): continue

            match = param_pattern.match(line_stripped)
            if match:
                value_str, param_name = match.groups()
                if param_name.lower() in ['true', 'false', 'default', 'unused', 'none', 'end', 'echo'] or any(ext in value_str.lower() for ext in ['.dat', '.txt', '.csv', '.twr', '.bld', '.ipt', '.in']):
                    continue
                param_info = self.parse_parameter_value(value_str, line)
                if param_info:
                    parameters[param_name] = {'line_number': i, 'original_value': param_info['value'], 'type': param_info['type'], 'description': line.split('!', 1)[-1].split('-', 1)[-1].strip(), 'unit': self.extract_unit(line)}
        return parameters
        
    def parse_parameter_value(self, value_str, description):
        value_str = value_str.strip().strip('"\'')
        if value_str.upper() in ['DEFAULT']: return None
        try:
            value = float(value_str)
            if value == int(value) and '.' not in value_str and 'e' not in value_str.lower(): return {'value': int(value), 'type': 'int'}
            else: return {'value': value, 'type': 'float'}
        except ValueError: pass
        if value_str.lower() in ['true', 'false']: return {'value': value_str.lower() == 'true', 'type': 'bool'}
        if any(keyword in description.lower() for keyword in ['option', 'method', 'model', 'type', 'switch', 'code', 'name', 'file']): return {'value': value_str, 'type': 'option'}
        return None
        
    def extract_unit(self, description):
        match = re.search(r'\(([^)]+)\)', description)
        if match:
            unit = match.group(1)
            if not any(word in unit.lower() for word in ['flag', 'switch', 'quoted', 'string', 'option']): return unit
        return ''
        
    def show_parameter_selector(self):
        if not self.discovered_parameters: messagebox.showinfo("Info", "Run 'Discover Parameters' first."); return
        dialog = tk.Toplevel(self.root); dialog.title("Select Parameters to Vary"); dialog.geometry("900x700")
        search_frame = ttk.Frame(dialog); search_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side='left', padx=5)
        search_var = tk.StringVar(); search_entry = ttk.Entry(search_frame, textvariable=search_var, width=30); search_entry.pack(side='left', padx=5)
        tree_frame = ttk.Frame(dialog); tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        tree = ttk.Treeview(tree_frame, columns=('Type', 'Value', 'Unit', 'Description'), show='tree headings')
        tree.heading('#0', text='Parameter'); tree.heading('Type', text='Type'); tree.heading('Value', text='Current Value'); tree.heading('Unit', text='Unit'); tree.heading('Description', text='Description')
        tree.column('#0', width=200); tree.column('Type', width=80); tree.column('Value', width=100, anchor='e'); tree.column('Unit', width=80); tree.column('Description', width=350)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview); hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set); tree.grid(row=0, column=0, sticky='nsew'); vsb.grid(row=0, column=1, sticky='ns'); hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1); tree_frame.grid_columnconfigure(0, weight=1)
        all_items = []
        for file_type, params in sorted(self.discovered_parameters.items()):
            file_node = tree.insert('', 'end', text=file_type, open=False, tags=('file_node',))
            for param_name, param_info in sorted(params.items()):
                val_str = f"{param_info['original_value']:.4g}" if isinstance(param_info['original_value'], float) else str(param_info['original_value'])
                item = tree.insert(file_node, 'end', text=param_name, values=(param_info['type'], val_str, param_info.get('unit', ''), param_info['description'][:100]))
                all_items.append((item, file_type.lower(), param_name.lower(), param_info['description'].lower()))
        tree.tag_configure('file_node', font=('TkDefaultFont', 10, 'bold'))
        def search_params(*args):
            search_term = search_var.get().lower()
            for child in tree.get_children(): tree.item(child, open=False); tree.reattach(child, '', 'end')
            if not search_term: return
            for child in tree.get_children(): tree.detach(child)
            for item, file_type, param_name, desc in all_items:
                if search_term in param_name or search_term in desc or search_term in file_type:
                    parent = tree.parent(item); tree.reattach(parent, '', 'end'); tree.item(parent, open=True)
        search_var.trace('w', search_params)
        btn_frame = ttk.Frame(dialog); btn_frame.pack(fill='x', pady=10, padx=10)
        def add_selected():
            added_count = 0
            for item in tree.selection():
                parent = tree.parent(item)
                if parent:
                    file_type = tree.item(parent)['text']; param_name = tree.item(item)['text']
                    self.add_parameter_with_info(file_type, param_name, self.discovered_parameters[file_type][param_name])
                    added_count += 1
            dialog.destroy()
            if added_count > 0: self.log(f"Added {added_count} parameters for variation.")
        ttk.Button(btn_frame, text="Add Selected", command=add_selected, style="Accent.TButton").pack(side='right')
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='right', padx=5)

    def add_parameter_with_info(self, file_type, param_name, param_info):
        for entry in self.parameter_entries:
            if entry['file_type'] == file_type and entry['param_name'] == param_name:
                self.log(f"Parameter {file_type} - {param_name} is already added."); return

        row_frame = ttk.Frame(self.param_list_frame); row_frame.pack(fill='x', pady=4, padx=2)
        param_label = ttk.Label(row_frame, text=f"{file_type} - {param_name}", width=35, anchor='w', wraplength=220)
        param_label.grid(row=0, column=0, rowspan=2, padx=5, sticky='w')
        param_type, current_val = param_info['type'], param_info['original_value']
        entry_data = {'frame': row_frame, 'file_type': file_type, 'param_name': param_name, 'param_info': param_info, 'widgets': {}}
        csv_var = tk.StringVar(value=str(current_val))
        entry_data['widgets']['csv_lbl'] = ttk.Label(row_frame, text="CSV Values:")
        entry_data['widgets']['csv_ent'] = ttk.Entry(row_frame, textvariable=csv_var, width=40)
        entry_data.update({'csv_var': csv_var}); csv_var.trace_add("write", self.update_total_cases)

        if param_type == 'float':
            start_def, end_def = (current_val * 0.8, current_val * 1.2) if isinstance(current_val, (int, float)) and abs(current_val) > 1e-9 else (-1.0, 1.0)
            start_var, end_var, steps_var = tk.DoubleVar(value=start_def), tk.DoubleVar(value=end_def), tk.IntVar(value=5)
            entry_data['widgets'].update({'range_lbl_s': ttk.Label(row_frame, text="Start:"), 'range_ent_s': ttk.Entry(row_frame, textvariable=start_var, width=10),
                                          'range_lbl_e': ttk.Label(row_frame, text="End:"), 'range_ent_e': ttk.Entry(row_frame, textvariable=end_var, width=10),
                                          'range_lbl_st': ttk.Label(row_frame, text="Steps:"), 'range_spn_st': ttk.Spinbox(row_frame, from_=1, to=100, textvariable=steps_var, width=5)})
            entry_data.update({'start_var': start_var, 'end_var': end_var, 'steps_var': steps_var}); steps_var.trace_add("write", self.update_total_cases)
        elif param_type == 'int':
            mode_var, start_var, end_var, steps_var, list_var = tk.StringVar(value="Range"), tk.DoubleVar(value=current_val), tk.DoubleVar(value=current_val+4), tk.IntVar(value=5), tk.StringVar(value=str(current_val))
            def update_int_widgets():
                is_range = mode_var.get() == "Range"
                for name, w in entry_data['widgets'].items():
                    if name.startswith('range_'): w.grid() if is_range else w.grid_remove()
                    if name.startswith('list_'): w.grid() if not is_range else w.grid_remove()
                self.update_total_cases()
            entry_data['update_func'] = update_int_widgets
            entry_data['widgets'].update({'rad_range': ttk.Radiobutton(row_frame, text="Range", variable=mode_var, value="Range", command=update_int_widgets), 'rad_list': ttk.Radiobutton(row_frame, text="List", variable=mode_var, value="List", command=update_int_widgets),
                                          'range_lbl_s': ttk.Label(row_frame, text="Start:"), 'range_ent_s': ttk.Entry(row_frame, textvariable=start_var, width=8), 'range_lbl_e': ttk.Label(row_frame, text="End:"), 'range_ent_e': ttk.Entry(row_frame, textvariable=end_var, width=8),
                                          'range_lbl_st': ttk.Label(row_frame, text="Steps:"), 'range_spn_st': ttk.Spinbox(row_frame, from_=1, to=100, textvariable=steps_var, width=5),
                                          'list_lbl': ttk.Label(row_frame, text="List (CSV):"), 'list_ent': ttk.Entry(row_frame, textvariable=list_var, width=25)})
            entry_data.update({'int_mode_var': mode_var, 'start_var': start_var, 'end_var': end_var, 'steps_var': steps_var, 'list_var': list_var})
            steps_var.trace_add("write", self.update_total_cases); list_var.trace_add("write", self.update_total_cases)
        elif param_type == 'bool':
            bool_var = tk.StringVar(value="Vary (True & False)")
            entry_data['widgets'].update({'bool_lbl': ttk.Label(row_frame, text="Value:"), 'bool_combo': ttk.Combobox(row_frame, textvariable=bool_var, values=["Vary (True & False)", "True", "False"], width=20)})
            entry_data.update({'bool_var': bool_var}); bool_var.trace_add("write", self.update_total_cases)
        elif param_type == 'option':
            options_var = tk.StringVar(value=f'"{current_val}"')
            entry_data['widgets'].update({'opt_lbl': ttk.Label(row_frame, text="Options (CSV):"), 'opt_ent': ttk.Entry(row_frame, textvariable=options_var, width=30)})
            entry_data.update({'options_var': options_var}); options_var.trace_add("write", self.update_total_cases)

        info_text = f"[{param_info.get('unit', '')}] (Type: {param_type}, Current: {current_val})"
        entry_data['widgets']['info_lbl'] = ttk.Label(row_frame, text=info_text, foreground='gray')
        entry_data['widgets']['remove_btn'] = ttk.Button(row_frame, text="Remove", command=lambda e=entry_data: self.remove_parameter(e))
        row_frame.columnconfigure(8, weight=1)
        self.parameter_entries.append(entry_data)
        self.on_distribution_change()

    def remove_parameter(self, entry_to_remove):
        entry_to_remove['frame'].destroy(); self.parameter_entries.remove(entry_to_remove); self.update_total_cases()
        
    # --- REPLACED METHOD: Implements the combined geometry + standard parameter grid ---
    def generate_test_cases(self):
        # --- 1. Initial validation ---
        is_geom_height = self.geom_vary_height_enabled.get()
        is_geom_diam = self.geom_vary_diam_enabled.get()
        is_geom_active = is_geom_height or is_geom_diam

        if not self.base_fst_path.get():
            messagebox.showerror("Error", "Please select a base FST file."); return
        if not self.file_structure:
            messagebox.showerror("Error", "Please run 'Discover Parameters' before generating cases. The file map is missing.")
            return
        if not is_geom_active and not self.parameter_entries:
            messagebox.showerror("Error", "Please enable a geometric variation or add at least one standard parameter."); return
        if is_geom_active and not GEOMETRY_ENGINE_AVAILABLE:
            messagebox.showerror("Error", "Geometric variation is enabled, but the engine script ('advanced_geometry_engine.py') is missing."); return

        self.setup_log.delete(1.0, tk.END)
        self.log("Starting test case generation...")
        
        try:
            # --- 2. Setup paths and directories ---
            output_path = Path(self.output_dir.get())
            if output_path.exists() and any(output_path.iterdir()):
                if not messagebox.askyesno("Warning", f"Output directory '{output_path}' is not empty. Overwrite?"): return
            shutil.rmtree(output_path, ignore_errors=True)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # --- 3. Build the Combined Grid of All Variations ---
            all_param_steps = []
            
            height_scales = np.linspace(self.geom_height_start_scale.get(), self.geom_height_end_scale.get(), self.geom_height_steps.get()) if is_geom_height else [1.0]
            all_param_steps.append(height_scales)

            diam_scales = np.linspace(self.geom_diam_start_scale.get(), self.geom_diam_end_scale.get(), self.geom_diam_steps.get()) if is_geom_diam else [1.0]
            all_param_steps.append(diam_scales)

            standard_param_combinations = [()]
            if self.parameter_entries:
                if self.distribution_var.get() == "grid_search":
                    param_values_list = []
                    for entry in self.parameter_entries:
                        param_type = entry['param_info']['type']; values = []
                        if param_type == 'float':
                            start, end, steps = entry['start_var'].get(), entry['end_var'].get(), entry['steps_var'].get()
                            values = np.linspace(start, end, steps) if steps > 1 else [start]
                        elif param_type == 'int':
                            if entry['int_mode_var'].get() == 'Range':
                                start, end, steps = entry['start_var'].get(), entry['end_var'].get(), entry['steps_var'].get()
                                values = np.round(np.linspace(start, end, steps)).astype(int) if steps > 1 else [int(round(start))]
                            else: values = [int(i.strip()) for i in entry['list_var'].get().split(',') if i.strip()]
                        elif param_type == 'bool': values = [True, False] if "Vary" in entry['bool_var'].get() else [entry['bool_var'].get() == "True"]
                        elif param_type == 'option': values = [opt.strip().strip('"\'') for opt in entry['options_var'].get().split(',') if opt.strip()]
                        param_values_list.append(values if values else [entry['param_info']['original_value']])
                    if param_values_list:
                        standard_param_combinations = list(itertools.product(*param_values_list))
                else:
                    messagebox.showerror("Logic Error", "Geometric variation is only compatible with 'Grid Search' distribution type.")
                    return
            all_param_steps.append(standard_param_combinations)

            case_combinations = list(itertools.product(*all_param_steps))
            num_cases = len(case_combinations)
            self.log(f"Total combinations to generate: {num_cases}")
            test_summary = []
            
            # --- 4. Prepare for Generation ---
            model = None
            if is_geom_active:
                # Find the required file paths from the already successful discovery
                ed_key = next((k for k, v in self.file_structure.items() if 'elastodyn' in v['path'].name.lower()), None)
                hd_key = next((k for k, v in self.file_structure.items() if 'hydrodyn' in v['path'].name.lower()), None)
                md_key = next((k for k, v in self.file_structure.items() if 'moordyn' in v['path'].name.lower()), None)

                if not all([ed_key, hd_key, md_key]):
                    raise FileNotFoundError("Could not find ElastoDyn, HydroDyn, or MoorDyn files in the discovered file structure. Please re-run discovery.")

                ed_path = self.file_structure[ed_key]['path']
                hd_path = self.file_structure[hd_key]['path']
                md_path = self.file_structure[md_key]['path']
                
                # Pass the resolved paths directly to the engine
                model = engine.PlatformModel(ed_path=ed_path, hd_path=hd_path, md_path=md_path)

            if model:
                for msg in model.log: self.log(f"  [Engine Discovery] {msg}")

            # --- 5. Main Generation Loop ---
            for i, combination in enumerate(case_combinations):
                case_name = f"case_{i+1:04d}"
                case_dir = output_path / case_name
                self.log(f"Creating test case {i+1}/{num_cases}: {case_name}")
                case_dir.mkdir(exist_ok=True)
                
                # 5.A. Copy ALL discovered files and rewrite their internal paths using the new robust method
                self.log(f"  Copying and rewriting {len(self.file_structure)} model files...")
                for file_key, file_info in self.file_structure.items():
                    source_path = file_info['path']
                    dest_path = case_dir / source_path.name
                    # This single call replaces all the old complex logic
                    self._copy_and_rewrite_paths(source_path, dest_path)

                case_params = {}
                height_scale, diam_scale, standard_combo = combination

                # 5.B. Apply GEOMETRIC variation (if active)
                if is_geom_active:
                    self.log(f"  Applying geometric variation: H_scale={height_scale:.3f}, D_scale={diam_scale:.3f}")
                    case_params.update({'height_scale': float(height_scale), 'diameter_scale': float(diam_scale)})
                    variation_data = model.generate_variation(height_scale=height_scale, diameter_scale=diam_scale)
                    engine.update_files_for_case(case_dir, variation_data, model)

                # 5.C. Apply STANDARD parameter variations
                if self.parameter_entries and standard_combo:
                    for j, value in enumerate(standard_combo):
                        entry = self.parameter_entries[j]
                        file_key = entry['file_type']
                        param_name = entry['param_name']
                        p_info = self.discovered_parameters[file_key][param_name]
                        
                        if isinstance(value, np.integer): value = int(value)
                        elif isinstance(value, np.floating): value = float(value)
                        
                        case_params[f"{file_key}/{param_name}"] = value
                        self.modify_parameter_in_file(case_dir, file_key, param_name, value, p_info)

                # 5.D. Save case-specific metadata
                case_info_data = {'case_name': case_name, 'fst_file': Path(self.base_fst_path.get()).name, 'parameters': case_params}
                test_summary.append(case_info_data)
                with open(case_dir / 'case_info.json', 'w') as f: json.dump(case_info_data, f, indent=2)

            # --- 6. Finalize and Save Summary ---
            summary_file = output_path / "test_cases_summary.json"
            with open(summary_file, 'w') as f: json.dump({'generation_date': datetime.now().isoformat(), 'base_fst_file': self.base_fst_path.get(), 'num_cases': num_cases, 'test_cases': test_summary}, f, indent=4)
            
            self.log(f"Successfully generated {num_cases} test cases in '{output_path}'")
            if messagebox.askyesno("Success", f"Generated {num_cases} test cases.\nSwitch to 'Run Simulations' tab?"):
                self.notebook.select(self.run_tab); self.load_run_cases()

        except Exception as e:
            self.log(f"Error: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to generate test cases: {str(e)}")
    def modify_parameter_in_file(self, case_dir, file_type, param_name, value, param_info):
        original_path = self.file_structure[file_type]['path']
        file_path = case_dir / original_path.name
        if not file_path.exists(): self.log(f"Warning: File {file_path} not found for parameter {param_name}"); return
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: lines = f.readlines()
        modified, line_num = False, param_info.get('line_number', -1)
        if 0 <= line_num < len(lines) and param_name in lines[line_num]:
            lines[line_num] = self.format_parameter_line(lines[line_num], value, param_info); modified = True
        if not modified:
            for i, line in enumerate(lines):
                if re.search(r'\b' + re.escape(param_name) + r'\b', line) and not line.strip().startswith(('!', '#')):
                    lines[i] = self.format_parameter_line(line, value, param_info); modified = True; break
        if modified:
            with open(file_path, 'w', encoding='utf-8', errors='ignore') as f: f.writelines(lines)
        else: self.log(f"Warning: Parameter '{param_name}' not found in {file_path.name}")
            
    def format_parameter_line(self, line, new_value, param_info):
        param_type = param_info.get('type')
        if param_type == 'int': value_str = str(int(new_value))
        elif param_type == 'float': value_str = f"{new_value:.7f}".rstrip('0').rstrip('.') if 0.001 < abs(new_value) < 10000 else f"{new_value:.6e}"
        elif param_type == 'bool': value_str = str(bool(new_value))
        elif param_type == 'option': value_str = f'"{new_value}"' if ' ' in str(new_value) else str(new_value)
        else: value_str = str(new_value)
        parts = line.split()
        if not parts: return line
        old_value_pos = line.find(parts[0])
        return line[:old_value_pos] + value_str + line[old_value_pos + len(parts[0]):]
    
    def _rewrite_paths_in_file(self, target_file_path: Path, path_map: Dict[str, str]):
        """
        Reads a file and replaces all occurrences of old absolute/relative paths
        with their new local basenames.
        """
        if target_file_path.suffix.lower() not in ['.fst', '.dat', '.twr', '.bld', '.ipt', '.txt', '.in']:
            return # Skip binary or unknown files

        try:
            content = target_file_path.read_text(encoding='utf-8', errors='ignore')
            original_content = content

            # Create a regex that matches any of the original paths (escaped for regex)
            # Sort by length descending to match longer paths first (e.g., "Airfoils/Cylinder.dat" before "Cylinder.dat")
            sorted_paths = sorted(path_map.keys(), key=len, reverse=True)
            
            for old_path_str in sorted_paths:
                new_basename = path_map[old_path_str]
                # Replace if the path is found inside quotes
                content = content.replace(f'"{old_path_str}"', f'"{new_basename}"')
                content = content.replace(f"'{old_path_str}'", f"'{new_basename}'")

            if content != original_content:
                target_file_path.write_text(content, encoding='utf-8')
                self.log(f"    Rewrote internal paths in {target_file_path.name}")

        except Exception as e:
            self.log(f"    Warning: Could not read/rewrite paths in {target_file_path.name}. Error: {e}")    

    def generate_parameter_values(self, num_cases):
        dist = self.distribution_var.get()
        if dist == "grid_search":
            if not self.parameter_entries: return np.array([])
            param_steps = []
            for entry in self.parameter_entries:
                param_type = entry['param_info']['type']; values = []
                if param_type == 'float':
                    start, end, steps = entry['start_var'].get(), entry['end_var'].get(), entry['steps_var'].get()
                    values = np.linspace(start, end, steps) if steps > 1 else np.array([start])
                elif param_type == 'int':
                    if entry['int_mode_var'].get() == 'Range':
                        start, end, steps = entry['start_var'].get(), entry['end_var'].get(), entry['steps_var'].get()
                        values = np.round(np.linspace(start, end, steps) if steps > 1 else np.array([start])).astype(int)
                    else: values = [int(i.strip()) for i in entry['list_var'].get().split(',') if i.strip()]
                elif param_type == 'bool': values = [True, False] if "Vary" in entry['bool_var'].get() else [entry['bool_var'].get() == "True"]
                elif param_type == 'option': values = [opt.strip().strip('"\'') for opt in entry['options_var'].get().split(',') if opt.strip()]
                param_steps.append(values if len(values) > 0 else [entry['param_info']['original_value']])
            return np.array(list(itertools.product(*param_steps)), dtype=object).T
        elif dist == "csv_columnwise":
            if not self.parameter_entries: return np.array([])
            all_lists, param_names = [], []
            for entry in self.parameter_entries:
                param_info, param_name = entry['param_info'], entry['param_name']
                param_names.append(param_name)
                str_values = [item.strip() for item in entry['csv_var'].get().split(',') if item.strip()]
                try:
                    if param_info['type'] == 'float': typed_values = [float(v) for v in str_values]
                    elif param_info['type'] == 'int': typed_values = [int(v) for v in str_values]
                    elif param_info['type'] == 'bool': typed_values = [v.lower() in ['true', '1', 't', 'y', 'yes'] for v in str_values]
                    else: typed_values = [v.strip('"\'') for v in str_values]
                    all_lists.append(typed_values)
                except ValueError as e: messagebox.showerror("Input Error", f"Invalid value in CSV for '{param_name}' (type '{param_info['type']}'). Details: {e}"); return np.array([])
            if not all_lists or not all_lists[0]: return np.array([])
            first_len = len(all_lists[0])
            if any(len(lst) != first_len for lst in all_lists): messagebox.showerror("Input Error", f"All CSV inputs must have the same number of values (expected {first_len})."); return np.array([])
            return np.array(all_lists, dtype=object)
        else: # Sampling distributions
            numeric_params = [p for p in self.parameter_entries if p['param_info']['type'] in ['float', 'int']]
            if not numeric_params: self.log("Warning: Sampling distributions require numeric parameters."); return np.array([])
            try: from scipy.stats import qmc; sample = qmc.LatinHypercube(d=len(numeric_params)).sample(n=num_cases) if dist == "latin_hypercube" else np.random.rand(num_cases, len(numeric_params))
            except ImportError: self.log("Warning: 'scipy' not found. Falling back to uniform random."); sample = np.random.rand(num_cases, len(numeric_params))
            param_values = []
            for i, entry in enumerate(numeric_params):
                min_val, max_val = entry['start_var'].get(), entry['end_var'].get()
                scaled = min_val + (max_val - min_val) * sample[:, i]
                param_values.append(np.round(scaled).astype(int) if entry['param_info']['type'] == 'int' else scaled)
            return np.array(param_values)

    def on_distribution_change(self, event=None):
        dist_mode = self.distribution_var.get()
        is_grid, is_csv, is_sampling = dist_mode == "grid_search", dist_mode == "csv_columnwise", dist_mode not in ["grid_search", "csv_columnwise"]
        self.num_cases_spinbox.config(state='disabled' if is_grid or is_csv else 'normal')
        for entry in self.parameter_entries:
            for w in entry['widgets'].values():
                if hasattr(w, 'grid_remove'): w.grid_remove()
            param_type = entry['param_info']['type']
            if is_csv:
                entry['widgets']['csv_lbl'].grid(row=0, column=1, padx=(10, 2)); entry['widgets']['csv_ent'].grid(row=0, column=2, columnspan=5, sticky='ew')
            else:
                if param_type == 'float':
                    entry['widgets']['range_lbl_s'].grid(row=0, column=1, padx=(10, 2)); entry['widgets']['range_ent_s'].grid(row=0, column=2)
                    entry['widgets']['range_lbl_e'].grid(row=0, column=3, padx=5); entry['widgets']['range_ent_e'].grid(row=0, column=4)
                    entry['widgets']['range_lbl_st'].grid(row=0, column=5, padx=5); entry['widgets']['range_spn_st'].grid(row=0, column=6)
                elif param_type == 'int':
                    entry['widgets']['rad_range'].grid(row=0, column=1, sticky='w', padx=5); entry['widgets']['rad_list'].grid(row=1, column=1, sticky='w', padx=5)
                    if 'update_func' in entry: entry['update_func']()
                elif param_type == 'bool':
                    entry['widgets']['bool_lbl'].grid(row=0, column=1, padx=(10,2)); entry['widgets']['bool_combo'].grid(row=0, column=2, columnspan=3)
                elif param_type == 'option':
                    entry['widgets']['opt_lbl'].grid(row=0, column=1, padx=(10,2)); entry['widgets']['opt_ent'].grid(row=0, column=2, columnspan=5, sticky='ew')
                if is_sampling:
                    is_numeric = param_type in ['float', 'int']
                    for name, widget in entry['widgets'].items():
                        if hasattr(widget, 'config') and name not in ['info_lbl', 'remove_btn']: widget.config(state='disabled')
                    if is_numeric: entry['widgets']['range_ent_s'].config(state='normal'); entry['widgets']['range_ent_e'].config(state='normal')
            entry['widgets']['info_lbl'].grid(row=0, column=8, padx=5, sticky='w'); entry['widgets']['remove_btn'].grid(row=0, column=9, rowspan=2, padx=10)
        self.update_total_cases()

    def update_total_cases(self, *args):
        dist_mode = self.distribution_var.get()
        total = 1

        # Factor in geometric variations if enabled
        if self.geom_vary_height_enabled.get():
            try:
                total *= self.geom_height_steps.get()
            except (tk.TclError, ValueError): pass

        if self.geom_vary_diam_enabled.get():
            try:
                total *= self.geom_diam_steps.get()
            except (tk.TclError, ValueError): pass

        if dist_mode == "grid_search":
            is_geom_active = self.geom_vary_height_enabled.get() or self.geom_vary_diam_enabled.get()
            if not self.parameter_entries and not is_geom_active:
                total = 0
            
            for entry in self.parameter_entries:
                try:
                    if entry['param_info']['type'] == 'float': total *= entry['steps_var'].get()
                    elif entry['param_info']['type'] == 'int': total *= entry['steps_var'].get() if entry['int_mode_var'].get() == 'Range' else max(1, len([i for i in entry['list_var'].get().split(',') if i.strip()]))
                    elif entry['param_info']['type'] == 'bool': total *= 2 if "Vary" in entry['bool_var'].get() else 1
                    elif entry['param_info']['type'] == 'option': total *= max(1, len([o for o in entry['options_var'].get().split(',') if o.strip()]))
                except (tk.TclError, ValueError): pass
            self.num_cases.set(total)
        elif dist_mode == "csv_columnwise":
            total_csv = 0
            if self.parameter_entries:
                try: total_csv = len([i for i in self.parameter_entries[0]['csv_var'].get().split(',') if i.strip()])
                except (tk.TclError, IndexError): pass
            self.num_cases.set(total_csv * total) # Multiply by geometry steps
        else: # Sampling
            self.num_cases_spinbox.config(state='normal')
            
    def browse_fst_file(self):
        filename = filedialog.askopenfilename(title="Select base FST file", filetypes=[("FST files", "*.fst"), ("All files", "*.*")])
        if filename: self.base_fst_path.set(filename); self.log("Selected FST file: " + filename)
        if filename and messagebox.askyesno("Discover Parameters", "Discover parameters for this file now?"): self.discover_parameters()
        
    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory", initialdir=self.output_dir.get())
        if dirname: self.output_dir.set(dirname); self.log("Selected output directory: " + dirname)
        
    def browse_openfast_exe(self):
        filename = filedialog.askopenfilename(title="Select OpenFAST executable", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if filename: self.openfast_exe.set(filename); self.log_message('run_log', f"Selected OpenFAST executable: {filename}")
        
    def _load_cases_from_summary(self, tree, case_dict, log_attr):
        test_dir = self.output_dir.get() or filedialog.askdirectory(title="Select Test Case Directory")
        if not test_dir: return False
        self.output_dir.set(test_dir); tree.delete(*tree.get_children()); case_dict.clear()
        summary_file = Path(test_dir) / "test_cases_summary.json"
        if not summary_file.exists(): messagebox.showerror("Error", f"Could not find 'test_cases_summary.json' in {test_dir}"); return False
        with open(summary_file, 'r') as f: summary = json.load(f)
        for case_info in summary.get('test_cases', []):
            params_str = ', '.join([f"{k.split('/')[-1]}={v:.3g}" if isinstance(v, (int,float)) else f"{k.split('/')[-1]}={v}" for k, v in case_info['parameters'].items()])
            item_id = tree.insert('', 'end', text=case_info['case_name'], values=('Ready', params_str, '-', '-'))
            case_dict[item_id] = {'path': Path(test_dir) / case_info['case_name'], 'fst_file': case_info['fst_file'], 'name': case_info['case_name']}
        return True

    def load_run_cases(self):
        if self._load_cases_from_summary(self.run_tree, self.run_cases, 'run_log'):
            self.log_message('run_log', f"Loaded {len(self.run_cases)} cases to run from {self.output_dir.get()}"); self.select_all_cases(self.run_tree)

    def load_post_proc_cases(self):
        if self._load_cases_from_summary(self.post_proc_tree, self.post_proc_cases, 'post_proc_log'):
            self.log_message('post_proc_log', f"Loaded {len(self.post_proc_cases)} results to process from {self.output_dir.get()}"); self.select_all_cases(self.post_proc_tree)

    def select_all_cases(self, tree): tree.selection_set(tree.get_children())
    def deselect_all_cases(self, tree): tree.selection_set([])
    
    def run_selected_cases(self):
        if not self.openfast_exe.get() or not Path(self.openfast_exe.get()).exists(): messagebox.showerror("Error", "Please select a valid OpenFAST executable."); return
        selected_items = self.run_tree.selection()
        if not selected_items: messagebox.showwarning("Warning", "No test cases selected to run."); return
        if not messagebox.askyesno("Confirm", f"This will run {len(selected_items)} OpenFAST simulations. Continue?"): return
        self.run_progress_var.set(0); self.run_completed_cases = 0; self.run_total_cases = len(selected_items)
        while not self.run_job_queue.empty(): self.run_job_queue.get()
        for item_id in selected_items: self.run_job_queue.put(item_id)
        self.run_button.config(state='disabled'); threading.Thread(target=self.run_manager_thread, daemon=True).start()
        
    def run_manager_thread(self):
        num_workers = self.num_threads.get()
        self.message_queue.put(('run_log', f"Starting {self.run_total_cases} simulations with {num_workers} parallel workers..."))
        threads = [threading.Thread(target=self.run_worker, daemon=True) for _ in range(num_workers)]
        for t in threads: t.start()
        self.run_job_queue.join()
        self.message_queue.put(('run_log', "\n--- All simulations completed. ---")); self.message_queue.put(('enable_run_button', None))
        
    def run_worker(self):
        while True:
            try:
                item_id = self.run_job_queue.get_nowait()
            except queue.Empty:
                return

            case_data = self.run_cases[item_id]
            case_path, case_name = case_data['path'], case_data['name']
            self.message_queue.put(('run_tree_update', (item_id, 'Status', 'Running')))
            self.message_queue.put(('run_log', f"--- Running {case_name} ---"))
            start_time = datetime.now()
            
            has_error = False
            error_keywords = ["error:", "error ", "aborting", "failed", "fortran runtime error"]
            
            try:
                cmd = [self.openfast_exe.get(), case_data['fst_file']]
                process = subprocess.Popen(cmd, cwd=str(case_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
                
                for line in iter(process.stdout.readline, ''):
                    log_line = f"[{case_name}] {line.strip()}"
                    self.message_queue.put(('run_log', log_line))
                    if any(keyword in line.lower() for keyword in error_keywords):
                        has_error = True
                        self.message_queue.put(('run_log', f"[{case_name}] >>> Detected error keyword in output! <<<"))

                process.wait()
                runtime = (datetime.now() - start_time).total_seconds()
                
                if process.returncode != 0 or has_error:
                    result = f"Error (code {process.returncode})" if not has_error else "Error (detected in output)"
                    status = "Failed"
                else:
                    status = "Completed"
                    result = "Success (Ready for Post-Proc)"

            except Exception as e:
                runtime = (datetime.now() - start_time).total_seconds()
                result, status = f"Exception: {str(e)}", "Failed"
                self.message_queue.put(('run_log', f"FATAL ERROR launching {case_name}: {str(e)}\n{traceback.format_exc()}"))

            self.message_queue.put(('run_tree_update', (item_id, 'Status', status)))
            self.message_queue.put(('run_tree_update', (item_id, 'Result', result)))
            self.message_queue.put(('run_tree_update', (item_id, 'Runtime', f"{runtime:.1f}s")))
            with self.run_progress_lock:
                self.run_completed_cases += 1
                self.message_queue.put(('run_progress', (self.run_completed_cases / self.run_total_cases) * 100))
            self.run_job_queue.task_done()

    def run_selected_post_proc(self):
        selected_items = self.post_proc_tree.selection()
        if not selected_items: messagebox.showwarning("Warning", "No cases selected for post-processing."); return
        if not (self.run_convert_csv.get() or self.run_dalembert.get() or self.run_plotting.get()): messagebox.showwarning("Warning", "No post-processing tasks selected."); return
        if not messagebox.askyesno("Confirm", f"This will process {len(selected_items)} cases. Continue?"): return
        self.post_proc_progress_var.set(0); self.post_proc_completed_cases = 0; self.post_proc_total_cases = len(selected_items)
        while not self.post_proc_job_queue.empty(): self.post_proc_job_queue.get()
        for item_id in selected_items: self.post_proc_job_queue.put(item_id)
        self.post_proc_button.config(state='disabled'); threading.Thread(target=self.post_proc_manager_thread, daemon=True).start()

    def post_proc_manager_thread(self):
        num_workers = self.num_threads.get()
        self.message_queue.put(('post_proc_log', f"Starting post-processing for {self.post_proc_total_cases} cases with {num_workers} parallel workers..."))
        threads = [threading.Thread(target=self.post_proc_worker, daemon=True) for _ in range(num_workers)]
        for t in threads: t.start()
        self.post_proc_job_queue.join()
        self.message_queue.put(('post_proc_log', "\n--- All post-processing tasks completed. ---")); self.message_queue.put(('enable_post_proc_button', None))

    def post_proc_worker(self):
        while True:
            try: item_id = self.post_proc_job_queue.get_nowait()
            except queue.Empty: return
            case_data = self.post_proc_cases[item_id]
            self.message_queue.put(('post_proc_tree_update', (item_id, 'Status', 'Processing')))
            self.message_queue.put(('post_proc_log', f"--- Processing {case_data['name']} ---"))
            success = self.run_post_processing_steps(case_data)
            status, result = ("Completed", "Success") if success else ("Failed", "One or more tasks failed")
            self.message_queue.put(('post_proc_tree_update', (item_id, 'Status', status))); self.message_queue.put(('post_proc_tree_update', (item_id, 'Result', result)))
            with self.post_proc_progress_lock:
                self.post_proc_completed_cases += 1
                self.message_queue.put(('post_proc_progress', (self.post_proc_completed_cases / self.post_proc_total_cases) * 100))
            self.post_proc_job_queue.task_done()

    def run_post_processing_steps(self, case_data) -> bool:
        case_path, case_name = case_data['path'], case_data['name']
        self.message_queue.put(('post_proc_log', f"[{case_name}] Searching for main .out file in {case_path}"))
        
        main_out_file = None
        possible_out_files = list(case_path.glob('*.out'))
        candidate_files = [f for f in possible_out_files if 'MD.out' not in f.name and 'MoorDyn.out' not in f.name]
        
        if len(candidate_files) == 1:
            main_out_file = candidate_files[0]
            self.message_queue.put(('post_proc_log', f"[{case_name}] Found main output file: {main_out_file.name}"))
        elif len(candidate_files) > 1:
            self.message_queue.put(('post_proc_log', f"[{case_name}] WARNING: Found multiple candidate .out files. Using the first one found: {candidate_files[0].name}"))
            main_out_file = candidate_files[0]

        if not main_out_file or not main_out_file.exists() or main_out_file.stat().st_size < 200:
            self.message_queue.put(('post_proc_log', f"[{case_name}] ERROR: No suitable .out file found. The simulation may have failed or produced no output."))
            return False
            
        csv_path = main_out_file.with_suffix('.csv')
        overall_success = True
        
        # --- FIX: Dynamic analysis time based on TMax from FST file ---
        analysis_start_time = 300.0 # Default fallback
        fst_path = case_path / case_data['fst_file']
        try:
            with open(fst_path, 'r') as f:
                for line in f:
                    if "TMax" in line:
                        tmax = float(line.strip().split()[0])
                        analysis_start_time = tmax / 3.0
                        self.message_queue.put(('post_proc_log', f"[{case_name}] Found TMax={tmax}s. Setting analysis start time to {analysis_start_time:.2f}s."))
                        break
        except Exception as e:
            self.message_queue.put(('post_proc_log', f"[{case_name}] WARNING: Could not parse TMax from {fst_path.name}. Using default start time. Error: {e}"))
        # --- END FIX ---

        if self.run_convert_csv.get():
            try:
                converter = ConverterRunner(self.message_queue, case_name, 'post_proc_log')
                if converter.convert_openfast_to_csv_robust(str(main_out_file), str(csv_path)) is None:
                    self.message_queue.put(('post_proc_log', f"[{case_name}] CSV conversion failed. Halting subsequent tasks for this case.")); return False
            except Exception as e: 
                self.message_queue.put(('post_proc_log', f"[{case_name}] FATAL ERROR during CSV conversion: {e}\n{traceback.format_exc()}")); return False

        if self.run_dalembert.get():
            try:
                dalembert_dir = case_path / "dalembert_analysis"
                dalembert_dir.mkdir(exist_ok=True)
                runner = DalembertRunner(self.message_queue, case_name, 'post_proc_log')
                # Pass the dynamic start time to the runner
                runner.run(fst=str(fst_path), glue_out=str(main_out_file), outdir=str(dalembert_dir), analysis_start_time=analysis_start_time)
            except Exception as e: 
                self.message_queue.put(('post_proc_log', f"[{case_name}] ERROR in d'Alembert analysis: {e}\n{traceback.format_exc()}")); overall_success = False

        if self.run_plotting.get():
            if not csv_path.exists(): 
                self.message_queue.put(('post_proc_log', f"[{case_name}] Skipping plotting because CSV file '{csv_path.name}' not found."))
                overall_success = False
            else:
                # --- FIX: Use a lock to make plotting thread-safe ---
                with self.plotting_lock:
                    try:
                        plot_dir = case_path / "plots"
                        plot_dir.mkdir(exist_ok=True)
                        plot_runner = PlottingRunner(self.message_queue, case_name, 'post_proc_log')
                        # Pass the dynamic start time for stats calculation
                        plot_runner.run(csv_file=str(csv_path), output_dir=str(plot_dir), case_name=case_name, mean_start=analysis_start_time, always_minmax=False, minmax_range_frac=0.05, minmax_abs=0.0)
                    except Exception as e: 
                        self.message_queue.put(('post_proc_log', f"[{case_name}] ERROR in plotting: {e}\n{traceback.format_exc()}")); overall_success = False
        
        return overall_success

    def show_case_context_menu(self, event, tree, case_dict):
        item_id = tree.identify_row(event.y)
        if not item_id: return
        tree.selection_set(item_id)
        case_data = case_dict.get(item_id)
        if not case_data: return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label=f"Open Folder for '{case_data['name']}'", command=lambda p=case_data['path']: self.open_folder(p))
        menu.post(event.x_root, event.y_root)

    def open_folder(self, path):
        try:
            if sys.platform == "win32": os.startfile(path)
            elif sys.platform == "darwin": subprocess.Popen(["open", path])
            else: subprocess.Popen(["xdg-open", path])
        except Exception as e: messagebox.showerror("Error", f"Could not open folder: {e}")

    def show_file_structure(self):
        if not self.file_structure: messagebox.showinfo("Info", "Run 'Discover Parameters' first."); return
        dialog = tk.Toplevel(self.root); dialog.title("Discovered File Structure"); dialog.geometry("800x600")
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Consolas', 10)); text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert('end', "OpenFAST File Structure:\n" + "="*60 + "\n\n")
        for file_type, file_info in sorted(self.file_structure.items()):
            path = file_info.get('path')
            if path:
                text.insert('end', f"{file_type}:\n", 'heading'); text.insert('end', f"  Path: {path}\n")
                text.insert('end', f"  Parameters Found: {len(self.discovered_parameters.get(file_type, {}))}\n\n")
        text.tag_config('heading', font=('Consolas', 11, 'bold'), foreground='darkblue'); text.config(state='disabled')
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
        
    def save_config(self):
        if not self.parameter_entries: messagebox.showinfo("Info", "No parameters to save."); return
        config = {'base_fst_path': self.base_fst_path.get(), 'output_dir': self.output_dir.get(), 'num_cases': self.num_cases.get(), 'distribution': self.distribution_var.get(), 'parameters': []}
        for p in self.parameter_entries:
            p_data = {'file_type': p['file_type'], 'param_name': p['param_name'], 'csv_list': p['csv_var'].get()}
            if p['param_info']['type'] == 'float': p_data.update({'start': p['start_var'].get(), 'end': p['end_var'].get(), 'steps': p['steps_var'].get()})
            elif p['param_info']['type'] == 'int': p_data.update({'int_mode': p['int_mode_var'].get(), 'start': p['start_var'].get(), 'end': p['end_var'].get(), 'steps': p['steps_var'].get(), 'int_list': p['list_var'].get()})
            elif p['param_info']['type'] == 'bool': p_data.update({'bool_choice': p['bool_var'].get()})
            elif p['param_info']['type'] == 'option': p_data.update({'options_list': p['options_var'].get()})
            config['parameters'].append(p_data)
        filename = filedialog.asksaveasfilename(title="Save Configuration", defaultextension=".json", filetypes=[("JSON config", "*.json")])
        if filename:
            with open(filename, 'w') as f: json.dump(config, f, indent=4)
            self.log(f"Configuration saved to: {filename}")
            
    def load_config(self):
        filename = filedialog.askopenfilename(title="Load Configuration", filetypes=[("JSON config", "*.json")])
        if not filename: return
        try:
            with open(filename, 'r') as f: config = json.load(f)
            self.base_fst_path.set(config.get('base_fst_path', '')); self.output_dir.set(config.get('output_dir', 'test_cases'))
            self.num_cases.set(config.get('num_cases', 10)); self.distribution_var.set(config.get('distribution', 'grid_search'))
            self.clear_parameters()
            if self.base_fst_path.get() and not self.discovered_parameters: self.log("Base FST found, running discovery..."); self.discover_parameters()
            if not self.discovered_parameters: messagebox.showwarning("Warning", "Run parameter discovery before loading parameters."); return
            for param_config in config.get('parameters', []):
                file_type, param_name = param_config.get('file_type'), param_config.get('param_name')
                if file_type and param_name and file_type in self.discovered_parameters and param_name in self.discovered_parameters[file_type]:
                    param_info = self.discovered_parameters[file_type][param_name]
                    self.add_parameter_with_info(file_type, param_name, param_info)
                    entry = self.parameter_entries[-1]
                    if 'csv_list' in param_config: entry['csv_var'].set(param_config.get('csv_list', ''))
                    if entry['param_info']['type'] == 'float': entry['start_var'].set(param_config.get('start', 0)); entry['end_var'].set(param_config.get('end', 1)); entry['steps_var'].set(param_config.get('steps', 5))
                    elif entry['param_info']['type'] == 'int': entry['int_mode_var'].set(param_config.get('int_mode', 'Range')); entry['start_var'].set(param_config.get('start', 0)); entry['end_var'].set(param_config.get('end', 1)); entry['steps_var'].set(param_config.get('steps', 5)); entry['list_var'].set(param_config.get('int_list', '1,2,3'))
                    elif entry['param_info']['type'] == 'bool': entry['bool_var'].set(param_config.get('bool_choice', 'Vary (True & False)'))
                    elif entry['param_info']['type'] == 'option': entry['options_var'].set(param_config.get('options_list', ''))
                else: self.log(f"Warning: Could not find '{param_name}' in '{file_type}' from config.")
            self.log(f"Configuration loaded from: {filename}"); self.on_distribution_change()
        except Exception as e: messagebox.showerror("Error", f"Failed to load configuration: {str(e)}"); self.log(f"Error loading config: {e}")

    def clear_parameters(self):
        for entry in self.parameter_entries: entry['frame'].destroy()
        self.parameter_entries.clear(); self.update_total_cases()
        
    def log(self, message):
        self.setup_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"); self.setup_log.see(tk.END); self.root.update_idletasks()
        
    def log_message(self, log_attr_name, message):
        log_widget = getattr(self, log_attr_name); log_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"); log_widget.see(tk.END)
        
    def process_queue(self):
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                if msg_type == 'run_log': self.run_log.insert(tk.END, msg_data + '\n'); self.run_log.see(tk.END)
                elif msg_type == 'post_proc_log': self.post_proc_log.insert(tk.END, msg_data + '\n'); self.post_proc_log.see(tk.END)
                elif msg_type == 'run_tree_update': self.run_tree.set(*msg_data)
                elif msg_type == 'post_proc_tree_update': self.post_proc_tree.set(*msg_data)
                elif msg_type == 'run_progress': self.run_progress_bar['value'] = msg_data
                elif msg_type == 'post_proc_progress': self.post_proc_progress_bar['value'] = msg_data
                elif msg_type == 'enable_run_button': self.run_button.config(state='normal')
                elif msg_type == 'enable_post_proc_button': self.post_proc_button.config(state='normal')
        except queue.Empty: pass
        finally: self.root.after(100, self.process_queue)
    def create_tutorial_tab(self):
            """Creates the 'Tutorial' tab with instructions on how to use the application."""
            main_frame = ttk.Frame(self.tutorial_tab, padding="10")
            main_frame.pack(fill='both', expand=True)

            # Use a ScrolledText widget for the tutorial content
            text_widget = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, relief="flat", padx=10, pady=10)
            text_widget.pack(fill='both', expand=True)

            # --- Define text and formatting tags ---
            text_widget.tag_configure('h1', font=('TkDefaultFont', 16, 'bold'), spacing3=10, foreground='#003366')
            text_widget.tag_configure('h2', font=('TkDefaultFont', 12, 'bold'), spacing1=15, spacing3=5, foreground='#005A9E')
            text_widget.tag_configure('bold', font=('TkDefaultFont', 9, 'bold'))
            text_widget.tag_configure('code', font=('Consolas', 9), background='#f0f0f0', relief='raised', borderwidth=1, lmargin1=20, lmargin2=20, rmargin=20)
            text_widget.tag_configure('list_item', lmargin1=20, lmargin2=20)

            # --- Tutorial Content ---
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
                ("2. Parameter Discovery:", 'bold'),
                (" Click ", ''),
                ("Discover Parameters", 'code'),
                (". The application will scan your ", ''),
                (".fst", 'code'),
                (" file and all referenced input files (ElastoDyn, AeroDyn, etc.) to find numerical parameters that can be varied.\n", ''),
                ("3. Parameter Configuration:", 'bold'),
                (" Click ", ''),
                ("Add from Discovery", 'code'),
                (" to open a list of all found parameters. Select the ones you want to vary and click 'Add Selected'. For each added parameter, you must define how it will be varied based on the chosen 'Distribution Type'.\n", ''),
                ("   • ", 'list_item'),
                ("Grid Search:", 'bold'),
                (" Creates a test case for every possible combination of parameter values. You define the variation for each parameter (e.g., a range for floats/ints, a list for options).\n", 'list_item'),
                ("   • ", 'list_item'),
                ("CSV Column-wise:", 'bold'),
                (" Creates test cases based on columns of values. You provide a comma-separated list of values for each parameter. All lists must have the same length.\n", 'list_item'),
                ("   • ", 'list_item'),
                ("Sampling (LHS/Uniform):", 'bold'),
                (" Generates a specified number of random samples for numeric parameters within a defined start/end range.\n", 'list_item'),
                ("4. Generate Cases:", 'bold'),
                (" Click ", ''),
                ("Generate Test Cases", 'code'),
                (". This creates a subdirectory for each case, copies all necessary files, modifies the parameters, and saves a summary file (", ''),
                ("test_cases_summary.json", 'code'),
                (").\n\n", ''),
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
                (" is correct. Select the analysis tasks you want to perform:\n", ''),
                ("   • ", 'list_item'),
                ("Convert .out to .csv:", 'bold'),
                (" Converts the primary text output file to a more accessible CSV format.\n", 'list_item'),
                ("   • ", 'list_item'),
                ("Run d'Alembert Analysis:", 'bold'),
                (" Performs a static analysis to calculate system loads, including inertial effects. Generates reports and extrema files.\n", 'list_item'),
                ("   • ", 'list_item'),
                ("Generate Plots:", 'bold'),
                (" Automatically creates plots for key output channels (platform motion, tower loads, etc.) with statistical annotations.\n", 'list_item'),
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

            # Insert the text with the defined tags
            for text, tag in tutorial_text:
                text_widget.insert(tk.END, text, tag or ())

            # Make the text widget read-only
            text_widget.config(state='disabled')
def main():
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError): pass
    root = tk.Tk()
    app = OpenFASTTestCaseGUI(root)
    app.log("Welcome to the OpenFAST Workflow Manager!")
    app.log("=" * 60)
    app.log("1. Use 'Setup Cases' to discover parameters and generate cases.")
    app.log("2. Use 'Run Simulations' to execute the generated cases in parallel.")
    app.log("3. Use 'Post-Process Results' to analyze the completed simulations.")
    app.log("=" * 60)
    root.mainloop()

if __name__ == "__main__":
    main()