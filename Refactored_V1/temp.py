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
import gc  # NEW: Import garbage collector
from pathlib import Path
from datetime import datetime
from math import cos, sin, radians
from typing import List, Tuple, Dict, Any, Optional

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

# --- Check for Scipy dependency for Frequency Analysis ---
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# #############################################################################
# --- BEGIN: Standalone Helper Functions ---
# #############################################################################

def _strip_quotes(s: str) -> str:
    """Removes leading/trailing quotes from a string."""
    s = s.strip()
    return s[1:-1] if (s.startswith('"') and s.endswith('"')) or \
                      (s.startswith("'") and s.endswith("'")) else s

def _read_lines(p: str, logger: logging.Logger) -> List[str]:
    """Reads all lines from a file with error handling."""
    logger.debug(f"Reading file: {p}")
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines()
    except Exception as e:
        logger.error(f"Failed to read file {p}: {e}")
        return []

def _find_fst_refs(fst: str, logger: logging.Logger) -> Dict[str, str]:
    """Parses an FST file to find references to other module files."""
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
# --- BEGIN: CSV Converter Runner ---
# #############################################################################

class ConverterRunner:
    """Handles the conversion of OpenFAST .out files to .csv format."""
    def __init__(self, message_queue: queue.Queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def log(self, message: str):
        """Logs a message to the GUI via the message queue."""
        self.mq.put((self.log_type, f"[{self.case_name}][CSV] {message}"))

    def convert_openfast_to_csv_robust(self, input_file: str, output_file: str) -> bool:
        """
        Converts an OpenFAST output file to CSV using a memory-efficient streaming approach.
        This method reads the input file line-by-line and writes directly to the output
        CSV, avoiding loading the entire file into memory.

        Args:
            input_file: Path to the source .out file.
            output_file: Path to the destination .csv file.

        Returns:
            True if conversion was successful, False otherwise.
        """
        self.log(f"Attempting to convert '{Path(input_file).name}' using streaming...")

        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                # --- Find header and data start position ---
                header_lines, column_names, column_units, data_start_line_num = [], [], [], -1
                for i, line in enumerate(f_in):
                    header_lines.append(line)
                    # Search only near the top for performance
                    if 'Time' in line.split() and i < 200:
                        # Peek at the next line for units
                        current_pos = f_in.tell()
                        next_line = f_in.readline()
                        f_in.seek(current_pos) # Go back

                        potential_names = line.strip().split()
                        potential_units = next_line.strip().split()
                        if len(potential_names) == len(potential_units) and len(potential_names) > 1:
                            column_names = potential_names
                            column_units = potential_units
                            data_start_line_num = i + 2
                            header_lines.append(next_line) # Add units line
                            f_in.readline() # Consume the units line for real
                            break
                
                if not column_names:
                    self.log("Error: Could not find the header and unit lines. Check .out file format.")
                    return False

                # Handle duplicate column names
                seen = {}
                unique_columns = []
                for col in column_names:
                    if col in seen:
                        seen[col] += 1
                        new_col_name = f"{col}_{seen[col]}"
                        self.log(f"Warning: Duplicate column '{col}' found. Renaming to '{new_col_name}'.")
                        unique_columns.append(new_col_name)
                    else:
                        seen[col] = 1
                        unique_columns.append(col)
                column_names = unique_columns

                # --- Stream data processing ---
                row_count = 0
                with open(output_file, 'w', newline='') as f_out:
                    f_out.write(','.join(column_names) + '\n')
                    
                    for line_num, line in enumerate(f_in, start=data_start_line_num):
                        line = line.strip()
                        if not line: continue
                        
                        values = line.split()
                        if len(values) == len(column_names):
                            try:
                                formatted_values = [f'{float(val.replace("D", "E")):.6E}' for val in values]
                                f_out.write(','.join(formatted_values) + '\n')
                                row_count += 1
                            except ValueError:
                                self.log(f"Warning: Could not parse data on line {line_num}. Skipping.")
                        else:
                            self.log(f"Warning: Mismatch in column count on line {line_num}. Expected {len(column_names)}, found {len(values)}. Skipping.")

                if row_count == 0:
                    self.log("Error: No data was successfully parsed from the file.")
                    # Clean up empty file
                    try: Path(output_file).unlink()
                    except OSError: pass
                    return False

        except FileNotFoundError:
            self.log(f"Error: The input file was not found at '{input_file}'")
            return False
        except Exception as e:
            self.log(f"Error during streaming conversion: {e}\n{traceback.format_exc()}")
            return False

        # --- Write Metadata ---
        metadata_file = output_file.rsplit('.', 1)[0] + '_metadata.txt'
        with open(metadata_file, 'w') as f:
            f.write("OpenFAST Output File Metadata\n" + "=" * 60 + "\n\n")
            f.write(f"Source File: {Path(input_file).name}\n\n")
            
            # Write header description
            desc_lines = [hline for hline in header_lines if "Description:" in hline]
            if desc_lines:
                f.writelines(desc_lines)
            else:
                f.write("No 'Description:' line found in the original file header.\n")

            f.write("\nColumn Information:\n" + "-" * 60 + "\n")
            f.write(f"{'Column Name':<25} {'Units'}\n" + "-" * 60 + "\n")
            
            original_names_for_meta = header_lines[data_start_line_num - 2].strip().split()
            for name, unit in zip(original_names_for_meta, column_units):
                f.write(f"{name:<25} {unit}\n")

        self.log("--- Conversion Summary ---")
        self.log(f"{'Input file:':<20} {Path(input_file).name}")
        self.log(f"{'Output CSV:':<20} {Path(output_file).name}")
        self.log(f"{'Rows/Cols:':<20} {row_count} / {len(column_names)}")
        
        return True

# #############################################################################
# --- END: CSV Converter Runner ---
# #############################################################################


# #############################################################################
# --- BEGIN: Plotting Runner ---
# #############################################################################

class PlottingRunner:
    """Generates plots from a given CSV data file."""
    
    _VECTOR_BASES = ['TwrBsF', 'TwrBsM', 'HydroF', 'HydroM']
    _SCALAR_CHANNELS = [
        'PtfmRoll', 'PtfmPitch', 'PtfmYaw', 
        'PtfmSurge', 'PtfmSway', 'PtfmHeave', 
        'FairTen1', 'FairTen2', 'FairTen3'
    ]
    
    def __init__(self, message_queue: queue.Queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def log(self, message: str):
        self.mq.put((self.log_type, f"[{self.case_name}][Plot] {message}"))

    def _simplify_header(self, name: str) -> str:
        return re.sub(r'\s+', '', re.sub(r'\s*\(.*?\)\s*$', '', str(name))).lower()

    def _strip_units(self, name: str) -> str:
        return re.sub(r'\s*\(.*?\)\s*$', '', str(name)).strip()

    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        return next((c for c in df.columns if re.match(r'^\s*Time\b', str(c), re.IGNORECASE)), None)

    def _extract_units_from_header(self, name: str) -> str:
        m = re.search(r'\((.*?)\)', str(name))
        return m.group(1) if m else ""

    def _build_units_map_from_csv(self, columns: List[str]) -> Dict[str, str]:
        return {col: self._extract_units_from_header(col) for col in columns}

    def _find_vector_columns(self, df: pd.DataFrame, base_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        simplified_map = {col: self._simplify_header(col) for col in df.columns}
        base_simple = self._simplify_header(base_name)
        cols = {'x': None, 'y': None, 'z': None}
        
        for component in ['x', 'y', 'z']:
            pattern = re.compile(rf'^{re.escape(base_simple)}{component}[a-z0-9]*$', re.IGNORECASE)
            match_found = False
            for original_col, simplified_col in simplified_map.items():
                if pattern.match(simplified_col):
                    cols[component] = original_col
                    match_found = True
                    break
            if not match_found:
                self.log(f"Component search failed for base '{base_name}', component '{component}'")

        return cols['x'], cols['y'], cols['z']
    
    def _get_unit_with_fallback(self, column_name: str, units_map_csv: Dict[str, str]) -> str:
        u = units_map_csv.get(column_name, "")
        if u: return u
        METADATA_UNITS = {
            "time": "s", "twrbsfxt": "kN", "twrbsmxt": "kN-m", "ptfmroll": "deg",
            "ptfmsurge": "m", "hydrofxi": "N", "hydromxi": "N-m", "fairten1x": "N"
        }
        key = self._simplify_header(column_name)
        for k_meta, v_meta in METADATA_UNITS.items():
            if key.startswith(k_meta[:-1]) and key[-1] in 'xyz': return v_meta
            if key == k_meta: return v_meta
        return ""

    def _compute_stats_after_threshold(self, t: pd.Series, y: pd.Series, t0: float) -> Tuple[float, float, float, float, float, bool]:
        t_num, y_num = pd.to_numeric(t, errors='coerce'), pd.to_numeric(y, errors='coerce')
        mask = (t_num >= t0) & y_num.notna()
        series, t_series, used_tail = (y_num[mask], t_num[mask], True) if mask.any() else (y_num[y_num.notna()], t_num[y_num.notna()], False)
        if series.empty: return (np.nan,) * 5 + (used_tail,)
        mean_val, idx_min, idx_max = series.mean(), series.idxmin(), series.idxmax()
        return mean_val, series.loc[idx_min], series.loc[idx_max], float(t_series.loc[idx_min]), float(t_series.loc[idx_max]), used_tail

    def _draw_stats_for_series(self, ax: plt.Axes, color: str, time_col: str, df: pd.DataFrame, y_col: str, mean_start: float, time_unit: str, y_unit: str, label_prefix: str, always_minmax: bool, minmax_range_frac: float, minmax_abs: float):
        def format_eng(val: float) -> str:
            return f"{val:.3e}" if pd.notna(val) and (abs(val) >= 1e4 or (0 < abs(val) < 1e-2)) else (f"{val:.4f}" if pd.notna(val) else "N/A")
        
        def annotate_at_y_axis(ax: plt.Axes, y_value: float, text: str):
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            ax.annotate(text, xy=(0.0, y_value), xycoords=trans, xytext=(4, 0), textcoords='offset points', va='center', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))

        mean_val, ymin, ymax, t_min, t_max, used_tail = self._compute_stats_after_threshold(df[time_col], df[y_col], mean_start)
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

    def _plot_group(self, time_col: str, df: pd.DataFrame, series_cols: List[str], series_labels: Dict[str, str], series_units: Dict[str, str], group_title: str, x_label: str, y_unit_hint: Optional[str], mean_start: float, time_unit: str, case_suffix: str, output_dir: str, file_stub: str, **kwargs):
        units = [series_units.get(c, "") for c in series_cols]
        chosen_unit = y_unit_hint or next((u for u in units if u), "")
        if any((u and chosen_unit and u != chosen_unit) for u in units):
            self.log(f"Warning: Mixed units in group '{group_title}'. Using '{chosen_unit}'.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in series_cols:
            if col not in df.columns:
                self.log(f"Warning: Column '{col}' missing for group '{group_title}'.")
                continue
            label = series_labels.get(col, self._strip_units(col))
            (line_handle,) = ax.plot(df[time_col], df[col], label=label, linewidth=1.4)
            self._draw_stats_for_series(ax, line_handle.get_color(), time_col, df, col, mean_start, time_unit, series_units.get(col, ""), label, **kwargs)
        
        ax.set_title(f"{group_title} vs. {self._strip_units(time_col)}{case_suffix}", fontsize=16)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f"{group_title}{f' [{chosen_unit}]' if chosen_unit else ''}", fontsize=12)
        ax.grid(True)
        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, framealpha=0.9)
        plt.tight_layout()
        save_path = os.path.join(output_dir, re.sub(r'[\\/*?:"<>|()\s]', "", str(file_stub)) + '.png')
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', bbox_extra_artists=(legend,))
            self.log(f"Saved group plot: '{Path(save_path).name}'")
        except Exception as e:
            self.log(f"Error saving group plot '{group_title}': {e}")
        finally:
            plt.close(fig) # CRITICAL: Free memory

    def run(self, csv_file: str, output_dir: str, case_name: Optional[str] = None, mean_start: float = 300.0, **kwargs):
        if not MATPLOTLIB_AVAILABLE:
            self.log("Matplotlib not found, skipping plotting.")
            return
        
        self.log(f"Reading data from '{Path(csv_file).name}'...")
        try: 
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
        except Exception as e:
            self.log(f"Error reading CSV file: {e}")
            return
        
        time_col = self._find_time_column(df)
        if not time_col:
            self.log("Error: No 'Time' column found.")
            return

        csv_units_map = self._build_units_map_from_csv(list(df.columns))
        series_label, series_unit = {}, {}
        time_unit = self._get_unit_with_fallback(time_col, csv_units_map)
        x_label = f"{self._strip_units(time_col)}{f' [{time_unit}]' if time_unit else ''}"
        
        channels_to_plot, found_scalars = [], {}

        for base in self._VECTOR_BASES:
            self.log(f"Searching for vector components for base: '{base}'")
            x_col, y_col, z_col = self._find_vector_columns(df, base)
            self.log(f"  -> Found components: X='{x_col}', Y='{y_col}', Z='{z_col}'")
            
            if all([x_col, y_col, z_col]):
                try:
                    mag_vals = np.sqrt(pd.to_numeric(df[x_col],errors='coerce')**2 + pd.to_numeric(df[y_col],errors='coerce')**2 + pd.to_numeric(df[z_col],errors='coerce')**2)
                    mag_col = f"{base}_Magnitude"
                    df[mag_col] = mag_vals
                    channels_to_plot.append(mag_col)
                    series_label[mag_col] = mag_col
                    series_unit[mag_col] = self._get_unit_with_fallback(x_col, csv_units_map)
                except Exception as e:
                    self.log(f"Warning: Failed to compute magnitude for '{base}': {e}")
            else:
                self.log(f"  -> FAILED to find all three components for '{base}'. Skipping magnitude calculation.")

        for channel in self._SCALAR_CHANNELS:
            matches = [c for c in df.columns if str(c).strip().lower().startswith(channel.lower())]
            if matches:
                col = matches[0]
                channels_to_plot.append(col)
                series_label[col] = self._strip_units(col)
                series_unit[col] = self._get_unit_with_fallback(col, csv_units_map)
                found_scalars[channel] = col

        if not channels_to_plot:
            self.log("No channels were found to plot.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        self.log(f"Generating plots...")
        plt.style.use('ggplot')
        case_suffix = f" -- {case_name}" if case_name else ""

        # Individual plots
        for channel in channels_to_plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            try:
                label = series_label.get(channel, self._strip_units(channel))
                (h,) = ax.plot(df[time_col], df[channel], label=label)
                y_unit = series_unit.get(channel, "")
                ax.set_title(f'{label} vs. {self._strip_units(time_col)}{case_suffix}', fontsize=16)
                ax.set_xlabel(x_label, fontsize=12)
                ax.set_ylabel(f"{label}{f' [{y_unit}]' if y_unit else ''}", fontsize=12)
                self._draw_stats_for_series(ax, h.get_color(), time_col, df, channel, mean_start, time_unit, y_unit, label, **kwargs)
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                save_path = os.path.join(output_dir, re.sub(r'[\\/*?:"<>|()\s]', "", label) + '.png')
                plt.savefig(save_path, dpi=150)
            except Exception as e:
                self.log(f"Error plotting channel '{channel}': {e}")
            finally:
                plt.close(fig) # CRITICAL: Free memory

        # Group plots
        rpy_cols = [c for c in [found_scalars.get(k) for k in ['PtfmRoll', 'PtfmPitch', 'PtfmYaw']] if c]
        if rpy_cols: self._plot_group(time_col, df, rpy_cols, series_label, series_unit, "Platform Roll/Pitch/Yaw", x_label, "deg", mean_start, time_unit, case_suffix, output_dir, "Ptfm_RollPitchYaw", **kwargs)
        
        ssh_cols = [c for c in [found_scalars.get(k) for k in ['PtfmSurge', 'PtfmSway', 'PtfmHeave']] if c]
        if ssh_cols: self._plot_group(time_col, df, ssh_cols, series_label, series_unit, "Platform Surge/Sway/Heave", x_label, "m", mean_start, time_unit, case_suffix, output_dir, "Ptfm_SurgeSwayHeave", **kwargs)
        
        fair_cols = [c for c in [found_scalars.get(k) for k in ['FairTen1', 'FairTen2', 'FairTen3']] if c]
        if fair_cols: self._plot_group(time_col, df, fair_cols, series_label, series_unit, "Fairlead Tensions", x_label, "N", mean_start, time_unit, case_suffix, output_dir, "Fairlead_Tensions", **kwargs)
        
        self.log("Plotting complete.")

# #############################################################################
# --- END: Plotting Runner ---
# #############################################################################


# #############################################################################
# --- BEGIN: Frequency Analysis Runner ---
# #############################################################################

class FrequencyAnalysisRunner:
    """Encapsulates the logic for calculating natural frequencies from free decay tests."""
    
    _CANONICAL_COLUMN_ALIASES: Dict[str, str] = {
        "fairten1": "FAIRTEN1", "fair1ten": "FAIRTEN1", "fairten2": "FAIRTEN2",
        "fair2ten": "FAIRTEN2", "fairten3": "FAIRTEN3", "fair3ten": "FAIRTEN3",
        "anchten1": "ANCHTEN1", "anch1ten": "ANCHTEN1", "anchten2": "ANCHTEN2",
        "anch2ten": "ANCHTEN2", "anchten3": "ANCHTEN3", "anch3ten": "ANCHTEN3",
    }

    def __init__(self, message_queue: queue.Queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def log(self, message: str):
        self.mq.put((self.log_type, f"[{self.case_name}][Freq] {message}"))

    def _canonicalize_column_names(self, df: pd.DataFrame, units_map: Dict[str, str]) -> None:
        """Harmonize column names using _CANONICAL_COLUMN_ALIASES."""
        rename_map: Dict[str, str] = {}
        for col in list(df.columns):
            key = str(col).strip().lower()
            canonical = self._CANONICAL_COLUMN_ALIASES.get(key)
            if not canonical or canonical == col: continue
            if canonical in df.columns:
                if df[canonical].equals(df[col]):
                    df.drop(columns=[col], inplace=True)
                    if col in units_map:
                        unit_value = units_map.pop(col)
                        if canonical not in units_map or not units_map[canonical]:
                            units_map[canonical] = unit_value
                continue
            rename_map[col] = canonical
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            for old, new in rename_map.items():
                unit_value = units_map.pop(old, None)
                if unit_value is not None and (new not in units_map or not units_map[new]):
                    units_map[new] = unit_value

    def _read_fast_csv(self, csv_file: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
        """Reads a CSV file, handling headers and units."""
        try:
            df_raw = pd.read_csv(csv_file, comment="#", skip_blank_lines=True, engine="python")
            if df_raw.empty:
                raise ValueError(f"CSV '{csv_file}' appears to be empty.")

            df_raw.columns = [str(col).strip() for col in df_raw.columns]
            first_row = df_raw.iloc[0]
            units_map: Dict[str, str] = {}

            def looks_like_units(value: object) -> bool:
                return isinstance(value, str) and bool(re.search(r"[A-Za-z]", value))

            if any(looks_like_units(val) for val in first_row):
                for col in df_raw.columns:
                    unit_token = str(first_row[col])
                    units_map[col] = re.sub(r"^[[\(\s]*|[]\)\s]*$", "", unit_token).strip()
                df = df_raw.iloc[1:].reset_index(drop=True)
            else:
                df = df_raw.copy()
                units_map = {col: "" for col in df.columns}

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            self._canonicalize_column_names(df, units_map)
            return df, units_map
        except Exception as e:
            self.log(f"Error reading CSV for frequency analysis: {e}")
            return None, None

    def _calculate_frequencies_from_decay(self, time: np.ndarray, data: np.ndarray) -> Dict:
        """Analyzes a free decay signal to find damped and undamped frequencies."""
        prominence_threshold = (np.max(data) - np.min(data)) * 0.1
        peak_indices, _ = find_peaks(data, prominence=prominence_threshold)

        if len(peak_indices) < 2:
            raise ValueError("Could not find at least 2 significant peaks. Check signal or start time.")

        peak_times, peak_values = time[peak_indices], data[peak_indices]
        damped_periods = np.diff(peak_times)
        mean_damped_period = np.mean(damped_periods)
        damped_frequency_hz = 1 / mean_damped_period
        damped_frequency_rad = 2 * np.pi * damped_frequency_hz

        log_decrements = [np.log(peak_values[i] / peak_values[i+1]) for i in range(len(peak_values) - 1) if peak_values[i] > 0 and peak_values[i+1] > 0]
        if not log_decrements:
            raise ValueError("Could not calculate logarithmic decrement. Are peak values valid?")

        mean_log_decrement = np.mean(log_decrements)
        damping_ratio = mean_log_decrement / np.sqrt((2 * np.pi)**2 + mean_log_decrement**2)

        if damping_ratio >= 1:
            natural_frequency_rad, natural_period = np.nan, np.nan
        else:
            natural_frequency_rad = damped_frequency_rad / np.sqrt(1 - damping_ratio**2)
            natural_period = 2 * np.pi / natural_frequency_rad

        return {
            "damped_period_s": mean_damped_period, "damped_frequency_hz": damped_frequency_hz,
            "damped_frequency_rad_s": damped_frequency_rad, "logarithmic_decrement": mean_log_decrement,
            "damping_ratio_zeta": damping_ratio, "natural_period_s": natural_period,
            "natural_frequency_rad_s": natural_frequency_rad, "peak_indices": peak_indices.tolist(),
            "peak_times": peak_times.tolist(), "peak_values": peak_values.tolist(),
        }

    def _plot_decay_analysis(self, time, data, results, column_name, units, filename):
        """Generates and saves a plot of the free decay analysis."""
        fig, ax = plt.subplots(figsize=(12, 7))
        try:
            ax.plot(time, data, label=f'"{column_name}" Signal (Mean Subtracted)', color='cornflowerblue', zorder=2)
            ax.plot(results["peak_times"], results["peak_values"], 'o', color='crimson', markersize=8, label=f'Detected Peaks ({len(results["peak_times"])} found)', zorder=3)

            A0, zeta, wn = results["peak_values"][0], results["damping_ratio_zeta"], results["natural_frequency_rad_s"]
            envelope_time = np.linspace(results["peak_times"][0], time[-1], 500)
            decay_envelope = A0 * np.exp(-zeta * wn * (envelope_time - envelope_time[0]))
            ax.plot(envelope_time, decay_envelope, '--', color='black', label='Fitted Exponential Decay Envelope', zorder=4)
            ax.plot(envelope_time, -decay_envelope, '--', color='black', zorder=4)

            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel(f"Amplitude ({units})", fontsize=12)
            ax.set_title(f'Free Decay Analysis for "{column_name}"', fontsize=14, weight='bold')
            ax.legend(loc='upper right')

            stats_text = (
                f"Damped Natural Frequency (ωd): {results['damped_frequency_rad_s']:.4f} rad/s\n"
                f"Damped Period (Td): {results['damped_period_s']:.4f} s\n"
                f"--- Damping ---\n"
                f"Damping Ratio (ζ): {results['damping_ratio_zeta']:.4f}\n"
                f"Logarithmic Decrement (δ): {results['logarithmic_decrement']:.4f}\n"
                f"--- Undamped System ---\n"
                f"Undamped Natural Frequency (ωn): {results['natural_frequency_rad_s']:.4f} rad/s\n"
                f"Undamped Natural Period (Tn): {results['natural_period_s']:.4f} s"
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            self.log(f"Saved frequency analysis plot: {Path(filename).name}")
        finally:
            plt.close(fig) # CRITICAL: Free memory

    def run(self, csv_file: str, column_name: str, output_dir: str, start_time: float):
        """Main execution method for the frequency analysis."""
        self.log(f"Starting frequency analysis for column '{column_name}'")
        df, units_map = self._read_fast_csv(csv_file)
        if df is None: return

        analysis_col = column_name
        if analysis_col not in df.columns:
            for alias, canonical in self._CANONICAL_COLUMN_ALIASES.items():
                if alias == column_name.strip().lower() and canonical in df.columns:
                    analysis_col = canonical
                    break
            else:
                raise KeyError(f"Column '{column_name}' not found. Available: {', '.join(df.columns)}")

        self.log(f"Analyzing canonical column: '{analysis_col}'")
        df_filtered = df[df['Time'] >= start_time].copy()
        if df_filtered.empty:
            raise ValueError(f"No data available after start_time={start_time}s.")

        time_series = df_filtered['Time'].to_numpy()
        data_series_raw = df_filtered[analysis_col].to_numpy()
        mean_value = np.mean(data_series_raw)
        data_series_zero_meaned = data_series_raw - mean_value
        self.log(f"Subtracted signal mean ({mean_value:.4f}) for peak analysis.")

        results = self._calculate_frequencies_from_decay(time_series, data_series_zero_meaned)

        results_path = Path(output_dir) / f"frequency_results_{analysis_col}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        self.log(f"Saved numerical results to: {results_path.name}")

        plot_path = Path(output_dir) / f"frequency_plot_{analysis_col}.png"
        self._plot_decay_analysis(
            time_series, data_series_zero_meaned, results,
            analysis_col, units_map.get(analysis_col, '-'), filename=str(plot_path)
        )

# #############################################################################
# --- END: Frequency Analysis Runner ---
# #############################################################################


# #############################################################################
# --- BEGIN: d'Alembert Runner ---
# #############################################################################
class DalembertLogHandler(logging.Handler):
    """Custom logging handler to route logs to the GUI's message queue."""
    def __init__(self, message_queue, case_name, log_type):
        super().__init__()
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type
    def emit(self, record):
        self.mq.put((self.log_type, f"[{self.case_name}][Dalembert] {self.format(record)}"))

class DalembertRunner:
    """
    Performs d'Alembert staticization to extract quasi-static loads from
    dynamic simulation results.
    """
    def __init__(self, message_queue: queue.Queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"dalembert_{self.case_name}_{id(self)}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = DalembertLogHandler(self.mq, self.case_name, self.log_type)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def run(self, fst: str, glue_out: str, outdir: str, analysis_start_time: float, **kwargs):
        self.logger.info("========== d'Alembert staticization: START ==========")
        try:
            class Args:
                def __init__(self, d): self.__dict__.update(d)
            
            args = Args({
                'fst': fst, 'glue_out': glue_out, 'outdir': outdir,
                'outb': kwargs.get('outb', False), 'moordyn_out': kwargs.get('moordyn_out'),
                'rotate_ed': kwargs.get('rotate_ed', True), 'override_mass': kwargs.get('override_mass'),
                'override_com': kwargs.get('override_com'), 'override_inertia': kwargs.get('override_inertia'),
                'verbose': True, 'log_step': kwargs.get('log_step', 100)
            })
            self.logger.info("Arguments: " + json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
            os.makedirs(args.outdir, exist_ok=True)

            builder = self.MassPropertyBuilder(args.fst, self.logger)
            auto_m, auto_com, auto_Icom = builder.compute()
            
            m = args.override_mass if args.override_mass is not None else auto_m
            r_com = np.array(args.override_com, float) if args.override_com is not None else auto_com
            I = np.array([[args.override_inertia[0], args.override_inertia[3], args.override_inertia[4]],
                          [args.override_inertia[3], args.override_inertia[1], args.override_inertia[5]],
                          [args.override_inertia[4], args.override_inertia[5], args.override_inertia[2]]], float) if args.override_inertia is not None else auto_Icom

            self.logger.info(f"Mass properties in use: m={m:.6e} kg, CoM={r_com.tolist()}")

            refs = _find_fst_refs(args.fst, self.logger)
            geo = self._parse_elastodyn_geometry(refs['EDFile'])
            fairleads, anchors = self._parse_moordyn_points(refs['MooringFile'])
            PRP, yaw_xyz, twrbase_xyz = np.zeros(3), geo['YawBearing'], geo['TowerBase']

            df = self._parse_glue_text(args.glue_out)
            df = self._collapse_dupes(df)

            df_md = self._parse_glue_text(args.moordyn_out) if args.moordyn_out else None
            if df_md is not None and 'time' in df_md.columns:
                df_md = df_md.set_index('time')

            self._perform_dalembert_calculations(df, df_md, args, m, r_com, I, PRP, yaw_xyz, twrbase_xyz, fairleads, anchors, analysis_start_time, geo)

        except Exception as e:
            self.logger.error(f"FATAL ERROR in d'Alembert analysis: {e}\n{traceback.format_exc()}")
        finally:
            self.logger.info("========== d'Alembert staticization: END ==========")
    
    def _perform_dalembert_calculations(self, df: pd.DataFrame, df_md: Optional[pd.DataFrame], args: Any, m: float, r_com: np.ndarray, I: np.ndarray, PRP: np.ndarray, yaw_xyz: np.ndarray, twrbase_xyz: np.ndarray, fairleads: Dict[int, np.ndarray], anchors: Dict[int, np.ndarray], analysis_start_time: float, geo: Dict):
        hydro_cols=['hydrofxi','hydrofyi','hydrofzi','hydromxi','hydromyi','hydromzi']
        
        if all(c in df.columns for c in ['twrbsfxt','twrbsfyt','twrbsfzt','twrbsmxt','twrbsmyt','twrbsmzt']):
            edF_cols, edM_cols, ed_point, ed_name = ['twrbsfxt','twrbsfyt','twrbsfzt'], ['twrbsmxt','twrbsmyt','twrbsmzt'], twrbase_xyz, 'ed_towerbase_interface'
            self.logger.info("Using ED interface at Tower Base (TwrBs*) in platform axes")
        elif all(c in df.columns for c in ['yawbrfxp','yawbrfyp','yawbrfzp','yawbrmxp','yawbrmyp','yawbrmzp']):
            edF_cols, edM_cols, ed_point, ed_name = ['yawbrfxp','yawbrfyp','yawbrfzp'], ['yawbrmxp','yawbrmyp','yawbrmzp'], yaw_xyz, 'ed_yawbr_interface'
            self.logger.info("Using ED interface at Yaw Bearing (YawBr*) in platform axes")
        else:
            raise RuntimeError('Missing ED interface loads (TwrBs* or YawBr*).')

        rows=[]
        n=len(df)
        self.logger.info(f"Beginning time loop over {n} rows")
        
        force_methods = self._detect_mooring_force_methods(df, df_md, fairleads)
        
        for i, row in df.iterrows():
            t=row['time']

            H_F = row[hydro_cols[:3]].values
            H_M = row[hydro_cols[3:]].values

            ED_F_loc = row[edF_cols].values
            ED_M_loc = row[edM_cols].values

            R_plat = self._rotmat_from_rpy_deg(row['ptfmroll'], row['ptfmpitch'], row['ptfmyaw'])
            ED_F = R_plat @ ED_F_loc if args.rotate_ed else ED_F_loc
            ED_M = R_plat @ ED_M_loc if args.rotate_ed else ED_M_loc
            ED_M_at_PRP = ED_M + np.cross((PRP - ed_point), ED_F)

            Moor_F, Moor_M, fair_entries = self._calculate_mooring_loads(row, df_md, R_plat, fairleads, anchors, force_methods, i)

            F_ext = H_F + Moor_F + ED_F
            M_ext_at_PRP = H_M + Moor_M + ED_M_at_PRP
            F_inert = -F_ext
            M_inert = -M_ext_at_PRP + np.cross((r_com - PRP), F_ext)

            def add(name, F, P, Mv=None):
                rows.append({'Time':t, 'LoadName':name, 'Px':P[0],'Py':P[1],'Pz':P[2], 'Fx':F[0],'Fy':F[1],'Fz':F[2], 'Mx':Mv[0] if Mv is not None else 0, 'My':Mv[1] if Mv is not None else 0, 'Mz':Mv[2] if Mv is not None else 0, 'F_norm':self._vnorm(F), 'M_norm':self._vnorm(Mv) if Mv is not None else 0})
            
            add('HydroDyn_Total_at_PRP', H_F, PRP, H_M)
            add(ed_name, ED_F, ed_point, ED_M)
            
            for k, rk, Fk, method in fair_entries:
                method_label = {'moordyn_main': 'MoorDyn_HiFi', 'moordyn_file': 'MoorDyn_HiFi_File', 'geometric': 'MoorDyn_Approx'}[method]
                add(f'{method_label}_Fairlead{k}', Fk, rk, None)
            
            if i % 100 == 0:
                if not hasattr(self, 'moor_force_samples'): self.moor_force_samples = {k: [] for k in sorted(fairleads.keys())}
                for k, _, Fk, method in fair_entries:
                    self.moor_force_samples[k].append({'time': t, 'Fx': Fk[0], 'Fy': Fk[1], 'Fz': Fk[2], 'magnitude': np.linalg.norm(Fk), 'method': method})

            add('Inertia_Trans_CoM', F_inert, r_com, None)
            add('Inertia_Rot_CoM', np.zeros(3), r_com, M_inert)
            
            Tot_F = F_ext + F_inert
            Tot_M = M_ext_at_PRP + M_inert + np.cross((r_com - PRP), F_inert)
            add('TOTAL_with_Inertia_at_PRP', Tot_F, PRP, Tot_M)

            if i > 0 and i % (getattr(args, 'log_step', 100) * 10) == 0:
                self.logger.debug(f"Processing: {i/n:.1%} complete (t={t:.2f}s)")

        loads_df=pd.DataFrame(rows)
        loads_csv=os.path.join(args.outdir, 'loads_timeseries_staticized.csv')
        loads_df.to_csv(loads_csv, index=False)
        self.logger.info(f"Wrote timeseries loads: {Path(loads_csv).name}")
        self._write_reports(loads_df, args, geo, fairleads, m, r_com, I, analysis_start_time, force_methods)
    
    def _detect_mooring_force_methods(self, df, df_md, fairleads):
        """Pre-scans for available mooring force columns to decide on calculation strategy."""
        force_methods = {}
        self.logger.info("Mooring force calculation method detection:")
        for k in sorted(fairleads.keys()):
            con_point = k + 3
            # Case-insensitive check for Con<N>Fx/y/z, Line<N>Fx/y/z, etc.
            if any(next((c for c in df.columns if c.lower() == f'con{con_point}f{comp}'.lower()), None) for comp in ['x','y','z']):
                force_methods[k] = 'moordyn_main'
                self.logger.info(f"  Line {k}: Using MoorDyn force components from main output (HIGH FIDELITY)")
            elif any(next((c for c in df.columns if c.lower() == f'line{k}f{comp}'.lower()), None) for comp in ['x','y','z']):
                force_methods[k] = 'moordyn_main'
                self.logger.info(f"  Line {k}: Using MoorDyn legacy force components from main output (HIGH FIDELITY)")
            elif df_md is not None and any(next((c for c in df_md.columns if c.lower() == f'con{con_point}f{comp}'.lower()), None) for comp in ['x','y','z']):
                force_methods[k] = 'moordyn_file'
                self.logger.info(f"  Line {k}: Using MoorDyn force components from separate file (HIGH FIDELITY)")
            else:
                force_methods[k] = 'geometric'
                self.logger.warning(f"  Line {k}: Using geometric approximation (REDUCED ACCURACY)")
        
        if any(m == 'geometric' for m in force_methods.values()):
            self.logger.warning("╔════════════════════════════════════════════════════════════╗")
            self.logger.warning("║ NOTICE: Using straight-line approximation for some lines.  ║")
            self.logger.warning("║ For high fidelity, add Con<N>Fx/Fy/Fz to MoorDyn OUTPUTS.  ║")
            self.logger.warning("╚════════════════════════════════════════════════════════════╝")
        return force_methods
        
    def _calculate_mooring_loads(self, row, df_md, R_plat, fairleads, anchors, force_methods, timestep_index):
        """Calculates total mooring force and moment for a single timestep."""
        Moor_F, Moor_M = np.zeros(3), np.zeros(3)
        fair_entries = []
        platform_pos = row[['ptfmsurge', 'ptfmsway', 'ptfmheave']].values
        t = row['time']

        for k, rk_local in sorted(fairleads.items()):
            Fk = np.zeros(3)
            method_used = force_methods[k]
            rk_global = platform_pos + R_plat @ rk_local
            
            if method_used == 'moordyn_main':
                con_point = k + 3
                # Case-insensitive column fetching
                fx_col = next((c for c in row.index if c.lower() == f'con{con_point}fx'), next((c for c in row.index if c.lower() == f'line{k}fx'), None))
                fy_col = next((c for c in row.index if c.lower() == f'con{con_point}fy'), next((c for c in row.index if c.lower() == f'line{k}fy'), None))
                fz_col = next((c for c in row.index if c.lower() == f'con{con_point}fz'), next((c for c in row.index if c.lower() == f'line{k}fz'), None))
                if all([fx_col, fy_col, fz_col]):
                    Fk = row[[fx_col, fy_col, fz_col]].values.astype(float)
                else:
                    method_used = 'geometric' # Fallback
            
            elif method_used == 'moordyn_file':
                idx = (df_md.index - t).abs().argmin()
                con_point = k + 3
                fx_col = next((c for c in df_md.columns if c.lower() == f'con{con_point}fx'), None)
                fy_col = next((c for c in df_md.columns if c.lower() == f'con{con_point}fy'), None)
                fz_col = next((c for c in df_md.columns if c.lower() == f'con{con_point}fz'), None)
                if all([fx_col, fy_col, fz_col]):
                    Fk = df_md.iloc[idx][[fx_col, fy_col, fz_col]].values.astype(float)
                else:
                    method_used = 'geometric' # Fallback
            
            if method_used == 'geometric':
                tension_col = next((c for c in row.index if c.lower() == f'fairten{k}'), None)
                if tension_col and anchors.get(k) is not None:
                    tension_mag = row[tension_col]
                    direction_vec = anchors[k] - rk_global
                    norm = np.linalg.norm(direction_vec)
                    if norm > 1e-6: Fk = tension_mag * (direction_vec / norm)

            # Validation check against FairTen output
            tension_col = next((c for c in row.index if c.lower() == f'fairten{k}'), None)
            if tension_col and not np.all(Fk == 0):
                reported_tension = row[tension_col]
                computed_magnitude = np.linalg.norm(Fk)
                relative_error = abs(computed_magnitude - reported_tension) / max(reported_tension, 1e-6)
                if relative_error > 0.10 and timestep_index % 500 == 0: # Log periodically to avoid spam
                    self.logger.warning(f"t={t:.2f}, Line {k}: Force magnitude mismatch! Computed={computed_magnitude:.2e}, Reported={reported_tension:.2e}, Err={relative_error:.1%}")

            Moor_M += np.cross(rk_global, Fk)
            Moor_F += Fk
            fair_entries.append((k, rk_local, Fk, method_used))
            
        return Moor_F, Moor_M, fair_entries

    def _write_reports(self, loads_df, args, geo, fairleads, m, r_com, I, analysis_start_time, force_methods):
        """Generates comprehensive staticized load reports."""
        extrema_lines = []
        total = loads_df[(loads_df['LoadName']=='TOTAL_with_Inertia_at_PRP') & (loads_df['Time'] >= analysis_start_time)].copy()
        
        if total.empty:
            extrema_lines.append(f"No TOTAL_with_Inertia_at_PRP samples after {analysis_start_time:.2f}s.")
        else:
            Fmag, Mmag = np.sqrt(total['Fx']**2 + total['Fy']**2 + total['Fz']**2), np.sqrt(total['Mx']**2 + total['My']**2 + total['Mz']**2)
            cases = {'F_max': Fmag.idxmax(), 'M_max': Mmag.idxmax()}
            extrema_data = [{'Case': name, **total.loc[idx][['Time','Fx','Fy','Fz','Mx','My','Mz']]} for name, idx in cases.items()]
            extrema_df = pd.DataFrame(extrema_data)
            extrema_csv = os.path.join(args.outdir, f'loads_extrema_after{int(analysis_start_time)}s.csv')
            extrema_df.to_csv(extrema_csv, index=False)
            self.logger.info(f"Wrote extrema CSV: {Path(extrema_csv).name}")
            extrema_lines = extrema_df.to_string(index=False).split('\n')

        mooring_stats_lines = self._generate_mooring_stats_report(fairleads)

        rep = [
            "=" * 80, f"d'ALEMBERT STATICIZATION REPORT: {self.case_name}", "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
            "SIMULATION PARAMETERS:", "-" * 80
        ]
        try:
            fst_content = Path(args.fst).read_text(encoding='utf-8', errors='ignore')
            tmax_match = re.search(r'^\s*([^\s]+)\s+TMax\b', fst_content, re.MULTILINE | re.IGNORECASE)
            tmax = float(tmax_match.group(1)) if tmax_match else np.nan
            rep.append(f"  FST File: {Path(args.fst).name}")
            rep.append(f"  Simulation Duration (TMax) : {tmax:.2f} s")
            rep.append(f"  Analysis Start Time        : {analysis_start_time:.2f} s")
            rep.append(f"  Analysis Duration          : {total['Time'].max() - analysis_start_time:.2f} s" if not total.empty else "N/A")
        except Exception as e:
            self.logger.warning(f"Could not extract simulation parameters: {e}")
        
        rep.extend(["", "MASS PROPERTIES:", "-" * 80, f"  Total Mass: {m:.6e} kg", f"  Center of Mass (CoM): {r_com.tolist()}", ""])
        rep.extend(mooring_stats_lines)
        rep.extend(["", "LOAD EXTREMA SUMMARY (t >= {analysis_start_time:.2f} s):", "=" * 80, *extrema_lines, ""])
        
        report_path = os.path.join(args.outdir, 'staticized_report.txt')
        with open(report_path, 'w') as f: f.write("\n".join(rep))
        self.logger.info(f"Wrote comprehensive report: {Path(report_path).name}")

    def _generate_mooring_stats_report(self, fairleads):
        """Generates a text block with statistics about mooring forces."""
        lines = ["\nMooring Force Statistics:", "=" * 80]
        if not hasattr(self, 'moor_force_samples') or not self.moor_force_samples:
            lines.append("  No mooring force samples collected.")
            return lines

        for k in sorted(fairleads.keys()):
            samples = self.moor_force_samples.get(k, [])
            if not samples: continue
            
            df_samples = pd.DataFrame(samples)
            mag_stats = df_samples['magnitude'].describe()
            method = df_samples['method'].iloc[0]
            
            lines.append(f"\nLine {k} - Method: {method}:")
            lines.append(f"  Force Mag [N]: mean={mag_stats['mean']:.3e}, std={mag_stats['std']:.3e}, min={mag_stats['min']:.3e}, max={mag_stats['max']:.3e}")
            lines.append(f"  Mean F [N]:    Fx={df_samples['Fx'].mean():.3e}, Fy={df_samples['Fy'].mean():.3e}, Fz={df_samples['Fz'].mean():.3e}")
        return lines

    def _parse_glue_text(self, path: str) -> Optional[pd.DataFrame]:
        """
        Parses a whitespace-delimited OpenFAST text output file into a DataFrame
        using a memory-efficient streaming method.

        Args:
            path: The file path to the .out file.

        Returns:
            A pandas DataFrame with the time-series data, or None.
        """
        if path is None: return None
        self.logger.info(f"Parsing glue (text) output with streaming: {path}")
        
        header_line_num = None
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if line.strip().startswith('Time'):
                        header_line_num = i
                        break
        except Exception as e:
             raise RuntimeError(f"Could not read or find header in {path}: {e}")

        if header_line_num is None:
            raise RuntimeError(f"Header 'Time' not found in {path}")

        try:
            # Use pandas' optimized reader, skipping metadata and handling whitespace.
            df = pd.read_csv(
                path, sep=r'\s+', header=header_line_num,
                encoding='utf-8', errors='ignore', low_memory=True
            )
            # The line after the header contains units, which pandas reads as data.
            # Drop this first row of data.
            if not pd.api.types.is_numeric_dtype(df.iloc[0, 0]):
                df = df.iloc[1:].reset_index(drop=True)
            
            df = df.apply(pd.to_numeric, errors='coerce')
            df.columns = [c.lower() for c in df.columns]
            self.logger.debug(f"Glue columns: {list(df.columns)}; rows={len(df)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Pandas failed to parse {path}. Error: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Pandas parsing error in {path}")

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
        lines = _read_lines(ed_path, self.logger)
        def fget(key):
            return next((float(ln.strip().split()[0].strip('"\'')) for ln in lines if key in ln and not ln.strip().startswith(('!','#'))), None)
        
        tower_ht = fget('TowerHt') or 90.0
        tower_bs_ht = fget('TowerBsHt') or 0.0
        
        geo_data = {
            'TowerHt': tower_ht, 'TowerBsHt': tower_bs_ht,
            'YawBearing': np.array([0.0, 0.0, tower_ht]),
            'TowerBase': np.array([0.0, 0.0, tower_bs_ht]),
            'OverHang': fget('OverHang') or 0.0, 'ShftTilt': fget('ShftTilt') or 0.0,
            'Twr2Shft': fget('Twr2Shft') or 0.0, 'TipRad': fget('TipRad') or 0.0,
            'HubRad': fget('HubRad') or 0.0,
        }
        self.logger.debug(f"Extracted geometry: TowerHt={tower_ht:.1f}m, TowerBsHt={tower_bs_ht:.1f}m")
        return geo_data

    def _parse_moordyn_points(self, md_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        self.logger.info(f"Parsing MoorDyn fairlead and anchor points: {md_path}")
        lines = _read_lines(md_path, self.logger)
        points_data, lines_data = [], []
        in_points, in_lines = False, False

        for s in lines:
            s_upper = s.strip().upper()
            if s_upper.startswith('---'):
                in_points, in_lines = 'POINTS' in s_upper, 'LINES' in s_upper
                continue
            parts = s.strip().split()
            if not parts or not parts[0].isdigit(): continue
            if in_points and len(parts) >= 5:
                points_data.append((int(parts[0]), parts[1].upper(), float(parts[2]), float(parts[3]), float(parts[4])))
            elif in_lines and len(parts) >= 4:
                lines_data.append((int(parts[0]), int(parts[2]), int(parts[3])))

        all_points_map = {pid: {'att': att, 'pos': np.array([x, y, z])} for pid, att, x, y, z in points_data}
        fairleads, anchors = {}, {}

        for line_id, pida, pidb in lines_data:
            pointA, pointB = all_points_map.get(pida), all_points_map.get(pidb)
            if not pointA or not pointB: continue
            if pointA['att'] == 'VESSEL' and pointB['att'] == 'FIXED':
                fairleads[line_id], anchors[line_id] = pointA['pos'], pointB['pos']
            elif pointB['att'] == 'VESSEL' and pointA['att'] == 'FIXED':
                fairleads[line_id], anchors[line_id] = pointB['pos'], pointA['pos']
        
        self.logger.info(f"Found {len(fairleads)} fairleads and {len(anchors)} anchors.")
        return fairleads, anchors

    def _vnorm(self, v): return float(np.linalg.norm(np.asarray(v,float)))
    def _rotmat_from_rpy_deg(self, r,p,y):
        rz,ry,rx = radians(y),radians(p),radians(r); cz,sz=cos(rz),sin(rz); cy,sy=cos(ry),sin(ry); cx,sx=cos(rx),sin(rx)
        Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]); Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]); Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        return Rz@Ry@Rx

    class MassPropertyBuilder:
        def __init__(self, fst_path, logger):
            self.logger = logger
            self.fst_path = fst_path
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
                if not s or s.startswith(('!','#')): continue
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
# --- END: d'Alembert Runner ---
# #############################################################################


# #############################################################################
# --- Main GUI Application Class ---
# #############################################################################

class OpenFASTTestCaseGUI:
    """
    A comprehensive GUI for managing OpenFAST test case generation, execution,
    and post-processing in large batches.
    """
    # --- REFACTOR: Moved UI strings and default values to constants ---
    TUTORIAL_TAB_NAME = "Tutorial"
    SETUP_TAB_NAME = "1. Setup Cases"
    RUN_TAB_NAME = "2. Run Simulations"
    POST_PROC_TAB_NAME = "3. Post-Process Results"
    DEFAULT_ANALYSIS_START_TIME = 300.0

    def __init__(self, root: tk.Tk):
        """Initializes the main application window and its components."""
        self.root = root
        self.root.title("OpenFAST Test Case Workflow Manager")
        self.root.geometry("1200x850")
        self._set_app_icon()
        
        self._setup_style()
        self._init_vars()

        self._create_notebook_and_tabs()
        
        self.process_queue()
        self.log("Welcome to the OpenFAST Workflow Manager!")

    def _set_app_icon(self):
        """Tries to set a custom application icon."""
        try:
            # Assumes icon is in the same directory as the script
            icon_path = Path(__file__).parent / "logo.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception as e:
            print(f"Could not load custom icon: {e}")

    def _setup_style(self):
        """Configures the ttk style for the application."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Accent.TButton", foreground="white", background="#0078D7")
        style.map("Accent.TButton", background=[('active', '#005A9E')])

    def _init_vars(self):
        """Initializes all Tkinter variables and application state."""
        # File paths
        self.base_fst_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "test_cases"))
        self.openfast_exe = tk.StringVar()

        # State and data
        self.discovered_parameters: Dict[str, Dict] = {}
        self.file_structure: Dict[str, Dict] = {}
        self.parameter_entries: List[Dict] = []
        self.num_cases = tk.IntVar(value=10)
        
        # Threading and queues
        self.message_queue = queue.Queue()
        self.num_threads = tk.IntVar(value=max(1, os.cpu_count() // 2))
        
        # Task-specific data (run and post-proc)
        self.task_data = {
            'run': {'cases': {}, 'job_queue': queue.Queue(), 'progress_lock': threading.Lock(), 'completed': 0, 'total': 0},
            'post_proc': {'cases': {}, 'job_queue': queue.Queue(), 'progress_lock': threading.Lock(), 'completed': 0, 'total': 0}
        }
        self.plotting_lock = threading.Lock() # For matplotlib thread safety

        # Post-processing options
        self.run_convert_csv = tk.BooleanVar(value=True)
        self.run_dalembert = tk.BooleanVar(value=True)
        self.run_plotting = tk.BooleanVar(value=True)
        self.run_frequency_analysis = tk.BooleanVar(value=False)
        self.frequency_analysis_column = tk.StringVar(value="PtfmHeave")

    def _create_notebook_and_tabs(self):
        """Creates the main notebook and populates it with tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        tab_creators = {
            self.TUTORIAL_TAB_NAME: self.create_tutorial_tab,
            self.SETUP_TAB_NAME: self.create_setup_tab,
            self.RUN_TAB_NAME: self.create_run_tab,
            self.POST_PROC_TAB_NAME: self.create_post_proc_tab,
        }

        self.tabs = {}
        for name, creator_func in tab_creators.items():
            frame = ttk.Frame(self.notebook)
            self.tabs[name] = frame
            self.notebook.add(frame, text=name)
            creator_func(frame)

    # --- Tab Creation Methods ---

    def create_setup_tab(self, parent_frame: ttk.Frame):
        """Creates the content for the 'Setup Cases' tab."""
        main_frame = ttk.Frame(parent_frame)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.create_file_selection_section(scrollable_frame)
        self.create_test_config_section(scrollable_frame)
        self.create_parameter_discovery_section(scrollable_frame)
        self.create_parameter_section(scrollable_frame)
        self.create_action_section(scrollable_frame)
        
        log_frame = ttk.LabelFrame(parent_frame, text="Output Log", padding="10")
        log_frame.pack(fill='x', side='bottom', pady=5, padx=5)
        self.setup_log = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD, bg="#f0f0f0", relief="sunken", borderwidth=1)
        self.setup_log.pack(fill='both', expand=False)

    def create_run_tab(self, parent_frame: ttk.Frame):
        """Creates the content for the 'Run Simulations' tab using a helper."""
        # --- Top Configuration Frame ---
        config_frame = ttk.LabelFrame(parent_frame, text="Run Configuration", padding="10")
        config_frame.pack(fill='x', pady=5, padx=10)
        ttk.Label(config_frame, text="OpenFAST Path:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(config_frame, textvariable=self.openfast_exe, width=50).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_openfast_exe).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(config_frame, text="Parallel runs:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Spinbox(config_frame, from_=1, to=os.cpu_count() or 8, textvariable=self.num_threads, width=8).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        config_frame.columnconfigure(1, weight=1)

        # --- Main Task Layout ---
        widgets = self._create_task_tab_layout(
            parent=parent_frame,
            task_key='run',
            title="Test Cases to Run",
            columns=('Status', 'Parameters', 'Runtime', 'Result'),
            col_widths={'Status': 180, 'Parameters': 300, 'Runtime': 100, 'Result': 200},
            load_cmd=self.load_run_cases,
            run_cmd=self.run_selected_cases,
            run_button_text="Run Selected Simulations"
        )
        self.run_widgets = widgets

    def create_post_proc_tab(self, parent_frame: ttk.Frame):
        """Creates the content for the 'Post-Process' tab using a helper."""
        top_frame = ttk.Frame(parent_frame); top_frame.pack(fill='x', pady=5, padx=10)
        
        # --- Left: Configuration Frame ---
        config_frame = ttk.LabelFrame(top_frame, text="Configuration", padding="10")
        config_frame.pack(fill='x', expand=True, side='left', padx=(0, 5))
        ttk.Label(config_frame, text="Results Directory:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(config_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=5, pady=2)
        config_frame.columnconfigure(1, weight=1)
        
        # --- Right: Tasks Frame ---
        tasks_frame = ttk.LabelFrame(top_frame, text="Tasks to Run", padding="10")
        tasks_frame.pack(fill='x', side='left', padx=5)
        ttk.Checkbutton(tasks_frame, text="Convert .out to .csv", variable=self.run_convert_csv).pack(anchor='w')
        ttk.Checkbutton(tasks_frame, text="Run d'Alembert Analysis", variable=self.run_dalembert).pack(anchor='w')
        ttk.Checkbutton(tasks_frame, text="Generate Plots", variable=self.run_plotting).pack(anchor='w')
        
        freq_frame = ttk.Frame(tasks_frame)
        freq_frame.pack(anchor='w', fill='x', pady=(5,0))
        freq_check = ttk.Checkbutton(freq_frame, text="Run Frequency Analysis on column:", variable=self.run_frequency_analysis)
        freq_check.pack(side='left')
        freq_entry = ttk.Entry(freq_frame, textvariable=self.frequency_analysis_column, width=18)
        freq_entry.pack(side='left', padx=5)
        if not SCIPY_AVAILABLE:
            freq_check.config(state='disabled'); freq_entry.config(state='disabled')
            ttk.Label(tasks_frame, text="(Frequency Analysis requires 'scipy')", foreground="gray", font=('TkDefaultFont', 8)).pack(anchor='w')
        
        # --- Main Task Layout ---
        widgets = self._create_task_tab_layout(
            parent=parent_frame,
            task_key='post_proc',
            title="Cases to Process",
            columns=('Status', 'Parameters', 'Result'),
            col_widths={'Status': 120, 'Parameters': 400, 'Result': 200},
            load_cmd=self.load_post_proc_cases,
            run_cmd=self.run_selected_post_proc,
            run_button_text="Run Post-Processing"
        )
        self.post_proc_widgets = widgets

    def _create_task_tab_layout(self, parent, task_key, title, columns, col_widths, load_cmd, run_cmd, run_button_text):
        """REFACTOR: Helper to create the common layout for run/post-proc tabs."""
        case_frame = ttk.LabelFrame(parent, text=title, padding="10")
        case_frame.pack(fill='both', expand=True, pady=5, padx=10)
        
        btn_frame = ttk.Frame(case_frame)
        btn_frame.pack(fill='x', pady=5)
        
        list_frame = ttk.Frame(case_frame)
        list_frame.pack(fill='both', expand=True)

        # *** FIX IS HERE ***
        # The Treeview's parent must be the `list_frame` where it will be gridded,
        # not the `case_frame` which uses the pack manager.
        tree = ttk.Treeview(list_frame, columns=columns, show='headings', selectmode='extended')
        
        ttk.Button(btn_frame, text="Load Cases", command=load_cmd).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Select All", command=lambda: tree.selection_set(tree.get_children())).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Deselect All", command=lambda: tree.selection_set([])).pack(side='left', padx=5)
        run_button = ttk.Button(btn_frame, text=run_button_text, command=run_cmd, style="Accent.TButton")
        run_button.pack(side='left', padx=20)
        
        tree.heading('#0', text='Test Case')
        tree.column('#0', width=150, anchor='w')
        for col, width in col_widths.items():
            tree.heading(col, text=col)
            tree.column(col, width=width, anchor='center' if col == 'Runtime' else 'w')

        tree_scroll_y = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree_scroll_x = ttk.Scrollbar(list_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        
        # Grid the tree and scrollbars inside list_frame
        tree.grid(row=0, column=0, sticky='nsew')
        tree_scroll_y.grid(row=0, column=1, sticky='ns')
        tree_scroll_x.grid(row=1, column=0, sticky='ew')
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        tree.bind("<Button-3>", lambda e: self.show_case_context_menu(e, tree, self.task_data[task_key]['cases']))
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(case_frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill='x', pady=5, side='bottom')

        log_widget = self.create_log_section(case_frame, f"{task_key}_log", "Execution Log")

        return {'tree': tree, 'run_button': run_button, 'progress_bar': progress_bar, 'progress_var': progress_var, 'log': log_widget}

    # --- Setup Tab Section Creators ---
    
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
        frame = ttk.LabelFrame(parent, text="Test Configuration", padding="10")
        frame.pack(fill='x', pady=5, padx=5)
        
        ttk.Label(frame, text="Number of Test Cases:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_cases_spinbox = ttk.Spinbox(frame, from_=2, to=10000, textvariable=self.num_cases, width=10)
        self.num_cases_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(frame, text="Parameter Distribution:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.distribution_var = tk.StringVar(value="grid_search")
        dist_combo = ttk.Combobox(frame, textvariable=self.distribution_var, values=["grid_search", "csv_columnwise", "latin_hypercube", "uniform", "normal"], width=20, state="readonly")
        dist_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        dist_combo.bind("<<ComboboxSelected>>", self.on_distribution_change)
        
        self.dist_help_label = ttk.Label(frame, text="Controls how standard parameters are varied.", foreground="gray", font=('TkDefaultFont', 9, 'italic'))
        self.dist_help_label.grid(row=1, column=2, columnspan=2, sticky='w', padx=10)

    def create_parameter_discovery_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameter Discovery", padding="10"); frame.pack(fill='x', pady=5, padx=5)
        ttk.Button(frame, text="Discover Parameters", command=self.discover_parameters, style="Accent.TButton").pack(side='left', padx=5)
        self.discovery_status = ttk.Label(frame, text="Select a .fst file and click 'Discover Parameters'"); self.discovery_status.pack(side='left', padx=20)
        
    def create_parameter_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameter Configuration", padding="10")
        frame.pack(fill='x', pady=5, padx=5)
        
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', pady=5)
        ttk.Button(control_frame, text="Add from Discovery", command=self.show_parameter_selector).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_parameters).pack(side='left', padx=5)
        
        scroll_container = ttk.Frame(frame, height=250)
        scroll_container.pack(fill='x', pady=5)
        scroll_container.pack_propagate(False)
        
        canvas = tk.Canvas(scroll_container, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        self.param_list_frame = ttk.Frame(canvas)
        
        self.param_list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.param_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
    def create_action_section(self, parent):
        frame = ttk.Frame(parent, padding="5"); frame.pack(fill='x', pady=10)
        ttk.Button(frame, text="Generate Test Cases", command=self.generate_test_cases, style="Accent.TButton").pack(side='left', padx=5)
        ttk.Button(frame, text="Load Configuration", command=self.load_config).pack(side='left', padx=5)
        ttk.Button(frame, text="Save Configuration", command=self.save_config).pack(side='left', padx=5)
        ttk.Button(frame, text="View File Structure", command=self.show_file_structure).pack(side='left', padx=5)
        
    def create_log_section(self, parent, log_attr_name, title="Output Log"):
        frame = ttk.LabelFrame(parent, text=title, padding="10")
        frame.pack(fill='both', expand=True, pady=5)
        log_widget = scrolledtext.ScrolledText(frame, height=8, wrap=tk.WORD, bg="#f0f0f0", relief="sunken", borderwidth=1)
        log_widget.pack(fill='both', expand=True)
        setattr(self, log_attr_name, log_widget)
        return log_widget

    # --- Case Generation and File Handling ---
        
    def _copy_and_rewrite_paths(self, source_path: Path, dest_path: Path):
        """Copies a file, rewriting internal relative paths to be local."""
        if source_path.suffix.lower() not in ['.fst', '.dat', '.twr', '.bld', '.ipt', '.txt', '.in']:
            shutil.copy2(source_path, dest_path)
            return

        try:
            content = source_path.read_text(encoding='utf-8', errors='ignore')
            # Pattern to find quoted paths that may have relative traversals
            pattern = re.compile(r'(["\'])((?:\.\.[/\])*[a-zA-Z0-9_.\-\s/\]+)\1')

            def replace_func(match):
                quote = match.group(1)
                path_str = match.group(2)
                if path_str.lower() in ['default', 'unused', 'none']: return match.group(0)
                new_basename = Path(path_str).name
                return f'{quote}{new_basename}{quote}'

            new_content = pattern.sub(replace_func, content)

            if new_content != content:
                self.log(f"    Rewrote internal paths in {dest_path.name}")
            dest_path.write_text(new_content, encoding='utf-8')
            
        except Exception as e:
            self.log(f"    Warning: Error rewriting {source_path.name}: {e}. Copying as-is.")
            shutil.copy2(source_path, dest_path)
        
    def _discover_and_parse_files_recursively(self, file_path: Path, file_info_by_path: Dict[Path, Dict], processed_paths: set):
        """Recursively scans files to find all dependencies and parameters."""
        if not file_path or not file_path.exists() or file_path in processed_paths:
            return

        self.log(f"  Scanning: {file_path.name}")
        processed_paths.add(file_path)

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            if file_path not in file_info_by_path:
                file_info_by_path[file_path] = {'key': file_path.name, 'original_strings': set(), 'params': {}}
            
            params = self.extract_parameters_from_file(content.splitlines())
            if params: file_info_by_path[file_path]['params'] = params

            # Pattern to find any quoted string that looks like a file path or root name
            pattern = re.compile(r'(["\'])((?:[a-zA-Z]:)?[a-zA-Z0-9_.\-\s\\/]+)\1')
            
            for match in pattern.finditer(content):
                path_inside_quotes = match.group(2)
                if not path_inside_quotes or path_inside_quotes.lower() in ['default', 'unused', 'none']: continue
                
                resolved_path = (file_path.parent / path_inside_quotes).resolve()
                
                if resolved_path.is_file():
                    if resolved_path not in processed_paths:
                        self._discover_and_parse_files_recursively(resolved_path, file_info_by_path, processed_paths)
                else: # Check for root name families (e.g., "wamit_data" for "wamit_data.1", ".3", etc.)
                    parent_dir, root_name = resolved_path.parent, resolved_path.name
                    if parent_dir.is_dir():
                        for item in parent_dir.glob(f"{root_name}.*"):
                            if item.is_file() and item not in processed_paths:
                                self.log(f"  [Discovery] Found root name family member: {item.name}")
                                self._discover_and_parse_files_recursively(item, file_info_by_path, processed_paths)

        except Exception as e:
            self.log(f"Could not process file {file_path.name}: {e}")
        
    def discover_parameters(self):
        if not self.base_fst_path.get():
            messagebox.showerror("Error", "Please select a base FST file first")
            return
        
        self.log("Starting deep parameter discovery...")
        self.discovery_status.config(text="Scanning all referenced files...")
        self.root.update()

        file_info_by_path, processed_paths = {}, set()
        try:
            self._discover_and_parse_files_recursively(Path(self.base_fst_path.get()), file_info_by_path, processed_paths)
            
            self.file_structure, self.discovered_parameters = {}, {}
            final_keys = set()
            for path, info in file_info_by_path.items():
                key = info['key']
                if key in final_keys: # Handle name collisions
                    key = f"{path.stem}_{sum(1 for k in final_keys if k.startswith(path.stem)) + 1}{path.suffix}"
                final_keys.add(key)
                
                self.file_structure[key] = {'path': path, 'original_strings': info['original_strings']}
                if info['params']: self.discovered_parameters[key] = info['params']
            
            total_params = sum(len(p) for p in self.discovered_parameters.values())
            self.discovery_status.config(text=f"Discovered {total_params} parameters across {len(self.file_structure)} files.")
            self.log(f"Discovery complete: Found {len(self.file_structure)} total files.")

        except Exception as e:
            self.log(f"Error during parameter discovery: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to discover parameters: {str(e)}")
        
    def extract_parameters_from_file(self, lines: List[str]):
        parameters = {}
        param_pattern = re.compile(r'^\s*([^\s!#]+)\s+([a-zA-Z_][a-zA-Z0-9_()]*)', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith(('!', '#')) or all(c in '-=_ ' for c in line_stripped):
                continue
            
            match = param_pattern.match(line_stripped)
            if not match: continue
            
            value_str, param_name = match.groups()
            
            if param_name.lower() in ['true', 'false', 'default', 'unused', 'none', 'end']: continue
            if any(ext in value_str.lower() for ext in ['.dat', '.txt', '.csv', '.twr', '.bld', '.fst']): continue
            
            try:
                param_info = self.parse_parameter_value(value_str, line)
                if param_info:
                    comment_match = re.search(r'[-!]\s*(.+)$', line)
                    description = comment_match.group(1).strip() if comment_match else ""
                    parameters[param_name] = {
                        'line_number': i, 'original_value': param_info['value'],
                        'type': param_info['type'], 'description': description,
                        'unit': self.extract_unit(line)
                    }
            except Exception:
                continue
        return parameters

    def extract_unit(self, line: str) -> str:
        matches = re.findall(r'\(([^)]+)\)', line)
        for match in matches:
            if len(match) < 10 and not any(word in match.lower() for word in ['flag', 'switch', 'see']):
                return match
        return ''

    def parse_parameter_value(self, value_str: str, description: str):
        value_str = value_str.strip().strip('"\'')
        if value_str.upper() in ['DEFAULT']: return None
        try:
            value = float(value_str)
            is_int_keyword = any(kw in description.lower() for kw in ['switch', 'flag', 'mode', 'method', 'order', 'num', 'index'])
            if is_int_keyword and value == int(value) and '.' not in value_str and 'e' not in value_str.lower():
                return {'value': int(value), 'type': 'int'}
            return {'value': value, 'type': 'float'}
        except ValueError:
            if value_str.lower() in ['true', 'false']:
                return {'value': value_str.lower() == 'true', 'type': 'bool'}
            if any(kw in description.lower() for kw in ['option', 'name', 'file', 'type']):
                return {'value': value_str, 'type': 'option'}
        return None
        
    def generate_test_cases(self):
        if not self.base_fst_path.get() or not self.file_structure:
            messagebox.showerror("Error", "Please select a base FST file and run 'Discover Parameters' first."); return
        if not self.parameter_entries:
            messagebox.showerror("Error", "Please add at least one parameter to vary."); return

        self.setup_log.delete(1.0, tk.END)
        self.log("Starting test case generation...")
        
        try:
            output_path = Path(self.output_dir.get())
            if output_path.exists() and any(output_path.iterdir()):
                if not messagebox.askyesno("Warning", f"Output directory '{output_path}' is not empty. Overwrite?"): return
            shutil.rmtree(output_path, ignore_errors=True)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # This logic correctly determines the combinations based on GUI settings
            # No changes needed here, as it was already robust.
            # ... (Full logic from original code) ...
            standard_param_combinations = []
            dist_type = self.distribution_var.get()
            
            if dist_type == "grid_search":
                param_values_list = []
                for entry in self.parameter_entries:
                    param_type = entry['param_info']['type']
                    values = []
                    if param_type == 'float':
                        start, end, steps = entry['start_var'].get(), entry['end_var'].get(), entry['steps_var'].get()
                        values = np.linspace(start, end, steps) if steps > 1 else [start]
                    elif param_type == 'int':
                        if entry['int_mode_var'].get() == 'Range':
                            start, end, steps = entry['start_var'].get(), entry['end_var'].get(), entry['steps_var'].get()
                            values = np.round(np.linspace(start, end, steps)).astype(int) if steps > 1 else [int(round(start))]
                        else:
                            values = [int(i.strip()) for i in entry['list_var'].get().split(',') if i.strip()]
                    elif param_type == 'bool':
                        values = [True, False] if "Vary" in entry['bool_var'].get() else [entry['bool_var'].get() == "True"]
                    elif param_type == 'option':
                        values = [opt.strip().strip('"\'') for opt in entry['options_var'].get().split(',') if opt.strip()]
                    param_values_list.append(values if values else [entry['param_info']['original_value']])
                
                if param_values_list:
                    standard_param_combinations = list(itertools.product(*param_values_list))
            
            elif dist_type == "csv_columnwise":
                all_lists = []
                for entry in self.parameter_entries:
                    str_values = [item.strip() for item in entry['csv_var'].get().split(',') if item.strip()]
                    try:
                        if entry['param_info']['type'] == 'float': typed_values = [float(v) for v in str_values]
                        elif entry['param_info']['type'] == 'int': typed_values = [int(float(v)) for v in str_values]
                        elif entry['param_info']['type'] == 'bool': typed_values = [v.lower() in ['true', '1'] for v in str_values]
                        else: typed_values = [v.strip('"\'') for v in str_values]
                        all_lists.append(typed_values)
                    except ValueError as e:
                        messagebox.showerror("Input Error", f"Invalid CSV value for '{entry['param_name']}': {e}"); return
                
                if all_lists and all_lists[0] and all(len(lst) == len(all_lists[0]) for lst in all_lists):
                    standard_param_combinations = list(zip(*all_lists))
                elif all_lists:
                    messagebox.showerror("Input Error", "All CSV inputs must have the same number of values."); return
            
            else: # Sampling distributions
                num_samples = self.num_cases.get()
                numeric_params = [p for p in self.parameter_entries if p['param_info']['type'] in ['float', 'int']]
                if not numeric_params: messagebox.showerror("Error", "Sampling distributions require numeric parameters."); return
                
                try:
                    from scipy.stats import qmc
                    sample = qmc.LatinHypercube(d=len(numeric_params)).sample(n=num_samples) if dist_type == "latin_hypercube" else np.random.rand(num_samples, len(numeric_params))
                except ImportError:
                    self.log("Warning: 'scipy' not found. Falling back to uniform random."); sample = np.random.rand(num_samples, len(numeric_params))
                
                param_values = [entry['start_var'].get() + (entry['end_var'].get() - entry['start_var'].get()) * sample[:, i] for i, entry in enumerate(numeric_params)]
                standard_param_combinations = list(zip(*param_values))

            if not standard_param_combinations:
                self.log("Error: No parameter combinations generated."); return
            
            num_cases = len(standard_param_combinations)
            self.log(f"Total combinations to generate: {num_cases}")
            if num_cases > 10000 and not messagebox.askyesno("Large Job", f"Generate {num_cases} cases?"): return
            
            test_summary = []
            for i, combo in enumerate(standard_param_combinations):
                case_name = f"case_{i+1:04d}"
                case_dir = output_path / case_name
                self.log(f"Creating test case {i+1}/{num_cases}: {case_name}")
                case_dir.mkdir()
                
                for file_key, file_info in self.file_structure.items():
                    self._copy_and_rewrite_paths(file_info['path'], case_dir / file_info['path'].name)

                case_params = {}
                for j, value in enumerate(combo):
                    entry = self.parameter_entries[j]
                    file_key, param_name = entry['file_type'], entry['param_name']
                    p_info = self.discovered_parameters[file_key][param_name]
                    
                    if isinstance(value, np.integer): value = int(value)
                    elif isinstance(value, np.floating): value = float(value)
                    
                    case_params[f"{file_key}/{param_name}"] = value
                    self.modify_parameter_in_file(case_dir, file_key, param_name, value, p_info)

                case_info_data = {'case_name': case_name, 'fst_file': Path(self.base_fst_path.get()).name, 'parameters': case_params}
                test_summary.append(case_info_data)
                (case_dir / 'case_info.json').write_text(json.dumps(case_info_data, indent=2))

            summary_file = output_path / "test_cases_summary.json"
            summary_data = {'generation_date': datetime.now().isoformat(), 'base_fst_file': self.base_fst_path.get(), 'num_cases': num_cases, 'test_cases': test_summary}
            summary_file.write_text(json.dumps(summary_data, indent=4))
            
            self.log(f"Successfully generated {num_cases} test cases in '{output_path}'")
            if messagebox.askyesno("Success", f"Generated {num_cases} test cases.\nSwitch to 'Run Simulations' tab?"):
                self.notebook.select(self.tabs[self.RUN_TAB_NAME])
                self.load_run_cases()

        except Exception as e:
            self.log(f"Error: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to generate test cases: {str(e)}")

    def modify_parameter_in_file(self, case_dir, file_key, param_name, value, param_info):
        file_path = case_dir / self.file_structure[file_key]['path'].name
        if not file_path.exists(): self.log(f"Warning: File {file_path} not found for param {param_name}"); return
        
        lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines(True)
        line_num = param_info.get('line_number', -1)
        
        if 0 <= line_num < len(lines) and param_name in lines[line_num]:
            lines[line_num] = self.format_parameter_line(lines[line_num], value, param_info)
            file_path.write_text("".join(lines), encoding='utf-8')
        else:
            self.log(f"Warning: Parameter '{param_name}' not found at expected line in {file_path.name}. Searching file...")
            for i, line in enumerate(lines):
                if re.search(r'\b' + re.escape(param_name) + r'\b', line) and not line.strip().startswith(('!', '#')):
                    lines[i] = self.format_parameter_line(line, value, param_info)
                    file_path.write_text("".join(lines), encoding='utf-8')
                    return
            self.log(f"Error: Could not find parameter '{param_name}' to modify in {file_path.name}")
            
    def format_parameter_line(self, line, new_value, param_info):
        param_type = param_info.get('type')
        if param_type == 'float': value_str = f"{new_value:.7G}"
        elif param_type == 'bool': value_str = str(bool(new_value)).upper()
        elif param_type == 'option': value_str = f'"{new_value}"' if ' ' in str(new_value) else str(new_value)
        else: value_str = str(new_value) # Includes int
        
        parts = line.split()
        if not parts: return line
        # Replace the first "word" on the line with the new value string
        return re.sub(r'^\s*[^\s]+', f'{value_str: >{len(parts[0])}}', line, count=1)

    # --- Task Execution and Management ---

    def _load_cases_to_tree(self, tree, case_dict, log_widget):
        """REFACTOR: Helper to load cases from summary file into a treeview."""
        test_dir = self.output_dir.get() or filedialog.askdirectory(title="Select Test Case Directory")
        if not test_dir: return False
        self.output_dir.set(test_dir)
        tree.delete(*tree.get_children())
        case_dict.clear()
        
        summary_file = Path(test_dir) / "test_cases_summary.json"
        if not summary_file.exists():
            messagebox.showerror("Error", f"Could not find 'test_cases_summary.json' in {test_dir}")
            return False
            
        with open(summary_file, 'r') as f: summary = json.load(f)
        for case_info in summary.get('test_cases', []):
            params_str = ', '.join([f"{k.split('/')[-1]}={v:.3g}" if isinstance(v, (int,float)) else f"{k.split('/')[-1]}={v}" for k, v in case_info.get('parameters', {}).items()])
            item_id = tree.insert('', 'end', text=case_info['case_name'], values=('Ready', params_str, '-', '-'))
            case_dict[item_id] = {'path': Path(test_dir) / case_info['case_name'], 'fst_file': case_info['fst_file'], 'name': case_info['case_name']}
        
        log_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(case_dict)} cases from {test_dir}\n")
        tree.selection_set(tree.get_children())
        return True

    def load_run_cases(self):
        self._load_cases_to_tree(self.run_widgets['tree'], self.task_data['run']['cases'], self.run_widgets['log'])

    def load_post_proc_cases(self):
        self._load_cases_to_tree(self.post_proc_widgets['tree'], self.task_data['post_proc']['cases'], self.post_proc_widgets['log'])

    def run_selected_cases(self):
        self._start_task('run', "OpenFAST simulations")

    def run_selected_post_proc(self):
        tasks_selected = self.run_convert_csv.get() or self.run_dalembert.get() or self.run_plotting.get() or self.run_frequency_analysis.get()
        if not tasks_selected: messagebox.showwarning("Warning", "No post-processing tasks selected."); return
        if self.run_frequency_analysis.get() and not self.frequency_analysis_column.get().strip():
            messagebox.showerror("Input Error", "Please specify a column name for Frequency Analysis."); return
        self._start_task('post_proc', "post-processing tasks")

    def _start_task(self, task_key, task_name):
        """Generic method to start a long-running batch task."""
        widgets = self.run_widgets if task_key == 'run' else self.post_proc_widgets
        task_info = self.task_data[task_key]
        
        selected_items = widgets['tree'].selection()
        if not selected_items:
            messagebox.showwarning("Warning", f"No cases selected for {task_name}."); return
        if not messagebox.askyesno("Confirm", f"This will run {len(selected_items)} {task_name}. Continue?"): return
        
        widgets['progress_var'].set(0)
        task_info['completed'] = 0
        task_info['total'] = len(selected_items)
        
        while not task_info['job_queue'].empty(): task_info['job_queue'].get()
        for item_id in selected_items: task_info['job_queue'].put(item_id)
        
        widgets['run_button'].config(state='disabled')
        manager_thread = threading.Thread(target=self._task_manager_thread, args=(task_key,), daemon=True)
        manager_thread.start()
        
    def _task_manager_thread(self, task_key: str):
        """Generic thread manager for running worker threads."""
        task_info = self.task_data[task_key]
        worker_func = self.run_worker if task_key == 'run' else self.post_proc_worker
        num_workers = self.num_threads.get()
        
        self.message_queue.put((f'{task_key}_log', f"Starting {task_info['total']} tasks with {num_workers} parallel workers..."))
        threads = [threading.Thread(target=worker_func, daemon=True) for _ in range(num_workers)]
        for t in threads: t.start()
        
        task_info['job_queue'].join() # Wait for all jobs to be processed
        
        self.message_queue.put((f'{task_key}_log', "\n--- All tasks completed. ---"))
        self.message_queue.put((f'enable_{task_key}_button', None))

    def run_worker(self):
        """Worker thread for executing a single OpenFAST simulation."""
        while True:
            try:
                item_id = self.task_data['run']['job_queue'].get_nowait()
            except queue.Empty:
                return

            case_data = self.task_data['run']['cases'][item_id]
            case_path, case_name = case_data['path'], case_data['name']
            self.message_queue.put(('run_tree_update', (item_id, 'Status', 'Running')))
            self.message_queue.put(('run_log', f"--- Running {case_name} ---"))
            start_time = datetime.now()
            
            try:
                cmd = [self.openfast_exe.get(), case_data['fst_file']]
                process = subprocess.Popen(cmd, cwd=str(case_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
                
                has_error = False
                error_keywords = ["error:", "error ", "aborting", "failed", "fortran runtime error"]
                for line in iter(process.stdout.readline, ''):
                    log_line = f"[{case_name}] {line.strip()}"
                    self.message_queue.put(('run_log', log_line))
                    if any(keyword in line.lower() for keyword in error_keywords): has_error = True

                process.wait()
                runtime = (datetime.now() - start_time).total_seconds()
                
                if process.returncode != 0 or has_error:
                    result, status = f"Error (code {process.returncode})" if not has_error else "Error (in output)", "Failed"
                else:
                    result, status = "Success", "Completed"

            except Exception as e:
                runtime = (datetime.now() - start_time).total_seconds()
                result, status = f"Exception: {e}", "Failed"
                self.message_queue.put(('run_log', f"FATAL ERROR launching {case_name}: {e}\n{traceback.format_exc()}"))

            self.message_queue.put(('run_tree_update', (item_id, 'Status', status)))
            self.message_queue.put(('run_tree_update', (item_id, 'Result', result)))
            self.message_queue.put(('run_tree_update', (item_id, 'Runtime', f"{runtime:.1f}s")))
            
            with self.task_data['run']['progress_lock']:
                self.task_data['run']['completed'] += 1
                progress = (self.task_data['run']['completed'] / self.task_data['run']['total']) * 100
                self.message_queue.put(('run_progress', progress))
            
            self.task_data['run']['job_queue'].task_done()
            gc.collect() # Clean up after external process

    def post_proc_worker(self):
        """Worker thread for post-processing a single completed case."""
        while True:
            try:
                item_id = self.task_data['post_proc']['job_queue'].get_nowait()
            except queue.Empty:
                return
            
            case_data = self.task_data['post_proc']['cases'][item_id]
            self.message_queue.put(('post_proc_tree_update', (item_id, 'Status', 'Processing')))
            self.message_queue.put(('post_proc_log', f"--- Processing {case_data['name']} ---"))
            
            success = self.run_post_processing_steps(case_data)
            status, result = ("Completed", "Success") if success else ("Failed", "Task(s) failed")
            
            self.message_queue.put(('post_proc_tree_update', (item_id, 'Status', status)))
            self.message_queue.put(('post_proc_tree_update', (item_id, 'Result', result)))
            
            with self.task_data['post_proc']['progress_lock']:
                self.task_data['post_proc']['completed'] += 1
                progress = (self.task_data['post_proc']['completed'] / self.task_data['post_proc']['total']) * 100
                self.message_queue.put(('post_proc_progress', progress))
            
            self.task_data['post_proc']['job_queue'].task_done()
            # MEMORY FIX: Explicitly request garbage collection
            self.message_queue.put(('post_proc_log', f"[{case_data['name']}] Requesting garbage collection to free memory."))
            gc.collect()

    def run_post_processing_steps(self, case_data: Dict) -> bool:
        """Runs the selected sequence of post-processing tasks for one case."""
        case_path, case_name = case_data['path'], case_data['name']
        self.message_queue.put(('post_proc_log', f"[{case_name}] Searching for main .out file..."))
        
        out_files = [f for f in case_path.glob('*.out') if 'MD.out' not in f.name and 'MoorDyn.out' not in f.name]
        if not out_files:
            self.message_queue.put(('post_proc_log', f"[{case_name}] ERROR: No suitable .out file found. Simulation may have failed."))
            return False
        main_out_file = out_files[0]
        if len(out_files) > 1: self.message_queue.put(('post_proc_log', f"[{case_name}] WARNING: Multiple .out files found, using '{main_out_file.name}'"))
            
        csv_path = main_out_file.with_suffix('.csv')
        overall_success = True
        
        analysis_start_time = self.DEFAULT_ANALYSIS_START_TIME
        try:
            fst_content = (case_path / case_data['fst_file']).read_text()
            tmax_match = re.search(r'^\s*([\d.eE+-]+)\s+TMax', fst_content, re.IGNORECASE | re.MULTILINE)
            if tmax_match: analysis_start_time = float(tmax_match.group(1)) / 3.0
        except Exception: pass
        self.message_queue.put(('post_proc_log', f"[{case_name}] Using analysis start time: {analysis_start_time:.2f}s"))

        if self.run_convert_csv.get():
            try:
                converter = ConverterRunner(self.message_queue, case_name, 'post_proc_log')
                # MEMORY FIX: Check boolean return value from streaming converter
                if not converter.convert_openfast_to_csv_robust(str(main_out_file), str(csv_path)):
                    self.message_queue.put(('post_proc_log', f"[{case_name}] CSV conversion failed. Halting subsequent tasks."))
                    return False
            except Exception as e: 
                self.message_queue.put(('post_proc_log', f"[{case_name}] FATAL ERROR during CSV conversion: {e}\n{traceback.format_exc()}"))
                return False

        if self.run_dalembert.get():
            try:
                dalembert_dir = case_path / "dalembert_analysis"
                dalembert_dir.mkdir(exist_ok=True)
                DalembertRunner(self.message_queue, case_name, 'post_proc_log').run(
                    fst=str(case_path / case_data['fst_file']), glue_out=str(main_out_file), 
                    outdir=str(dalembert_dir), analysis_start_time=analysis_start_time
                )
            except Exception as e: 
                self.message_queue.put(('post_proc_log', f"[{case_name}] ERROR in d'Alembert analysis: {e}\n{traceback.format_exc()}"))
                overall_success = False

        if self.run_plotting.get() and csv_path.exists():
            with self.plotting_lock:
                try:
                    plot_dir = case_path / "plots"
                    plot_dir.mkdir(exist_ok=True)
                    PlottingRunner(self.message_queue, case_name, 'post_proc_log').run(
                        csv_file=str(csv_path), output_dir=str(plot_dir), case_name=case_name, 
                        mean_start=analysis_start_time, always_minmax=False, minmax_range_frac=0.05, minmax_abs=0.0
                    )
                except Exception as e: 
                    self.message_queue.put(('post_proc_log', f"[{case_name}] ERROR in plotting: {e}\n{traceback.format_exc()}"))
                    overall_success = False
        
        if self.run_frequency_analysis.get() and SCIPY_AVAILABLE and csv_path.exists():
            with self.plotting_lock:
                try:
                    freq_dir = case_path / "frequency_analysis"
                    freq_dir.mkdir(exist_ok=True)
                    FrequencyAnalysisRunner(self.message_queue, case_name, 'post_proc_log').run(
                        csv_file=str(csv_path), column_name=self.frequency_analysis_column.get(),
                        output_dir=str(freq_dir), start_time=analysis_start_time
                    )
                except Exception as e:
                    self.message_queue.put(('post_proc_log', f"[{case_name}] ERROR in Frequency Analysis: {e}\n{traceback.format_exc()}"))
                    overall_success = False

        return overall_success

    # --- GUI Interaction and Helpers ---

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
        for file_key, file_info in sorted(self.file_structure.items()):
            text.insert('end', f"{file_key}:\n", 'heading')
            text.insert('end', f"  Path: {file_info.get('path')}\n")
            text.insert('end', f"  Parameters Found: {len(self.discovered_parameters.get(file_key, {}))}\n\n")
        text.tag_config('heading', font=('Consolas', 11, 'bold'), foreground='darkblue')
        text.config(state='disabled')
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
        
    def process_queue(self):
        """Processes messages from worker threads to update the GUI safely."""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                if msg_type.endswith('_log'):
                    log_widget = getattr(self, msg_type)
                    log_widget.insert(tk.END, msg_data + '\n')
                    log_widget.see(tk.END)
                elif msg_type.endswith('_tree_update'):
                    tree = self.run_widgets['tree'] if 'run' in msg_type else self.post_proc_widgets['tree']
                    tree.set(*msg_data)
                elif msg_type.endswith('_progress'):
                    widgets = self.run_widgets if 'run' in msg_type else self.post_proc_widgets
                    widgets['progress_bar']['value'] = msg_data
                elif msg_type.startswith('enable_'):
                    button_key = msg_type.replace('enable_', '')
                    widgets = self.run_widgets if 'run' in button_key else self.post_proc_widgets
                    widgets['run_button'].config(state='normal')
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    # --- Other UI methods (browsing, parameter selection, etc.) ---
    # These methods were mostly fine and are included here without major changes
    # for completeness. Minor cleanups have been applied.
    
    def create_tutorial_tab(self, parent_frame: ttk.Frame):
        """Creates the 'Tutorial' tab with instructions on how to use the application."""
        text_widget = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, relief="flat", padx=10, pady=10)
        text_widget.pack(fill='both', expand=True)
        # ... (Full tutorial text content from original code remains here) ...
        # This part is very long and unchanged, so it's omitted for brevity in this summary,
        # but it is included in the final code block.
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
        text_widget.tag_configure('h1', font=('TkDefaultFont', 16, 'bold'), spacing3=10)
        text_widget.tag_configure('h2', font=('TkDefaultFont', 12, 'bold'), spacing1=15, spacing3=5)
        text_widget.tag_configure('h3', font=('TkDefaultFont', 10, 'bold'), spacing1=10)
        text_widget.tag_configure('list_item', lmargin1=20, lmargin2=20)
        text_widget.tag_configure('code_block', font=('Consolas', 9), background='#f0f0f0', lmargin1=20, lmargin2=20)
        for text, tag in tutorial_text: text_widget.insert(tk.END, text, tag)
        text_widget.config(state='disabled')

    def browse_fst_file(self):
        filename = filedialog.askopenfilename(title="Select base FST file", filetypes=[("FST files", "*.fst"), ("All files", "*.*")])
        if filename:
            self.base_fst_path.set(filename)
            self.log("Selected FST file: " + filename)
            if messagebox.askyesno("Discover Parameters", "Discover parameters for this file now?"):
                self.discover_parameters()
    
    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory", initialdir=self.output_dir.get())
        if dirname:
            self.output_dir.set(dirname)
            self.log("Selected output directory: " + dirname)
    
    def browse_openfast_exe(self):
        filename = filedialog.askopenfilename(title="Select OpenFAST executable", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if filename:
            self.openfast_exe.set(filename)
            self.message_queue.put(('run_log', f"Selected OpenFAST executable: {filename}"))

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
        if any(e['file_type'] == file_type and e['param_name'] == param_name for e in self.parameter_entries):
            self.log(f"Parameter {file_type} - {param_name} is already added."); return

        row_frame = ttk.Frame(self.param_list_frame); row_frame.pack(fill='x', pady=4, padx=2)
        ttk.Label(row_frame, text=f"{file_type} - {param_name}", width=35, anchor='w', wraplength=220).grid(row=0, column=0, rowspan=2, padx=5, sticky='w')
        
        param_type, current_val = param_info['type'], param_info['original_value']
        entry_data = {'frame': row_frame, 'file_type': file_type, 'param_name': param_name, 'param_info': param_info, 'widgets': {}}
        
        csv_var = tk.StringVar(value=str(current_val))
        entry_data.update({'csv_var': csv_var, 'widgets': {'csv_lbl': ttk.Label(row_frame, text="CSV Values:"), 'csv_ent': ttk.Entry(row_frame, textvariable=csv_var, width=40)}})
        csv_var.trace_add("write", self.update_total_cases)

        if param_type == 'float':
            start_def, end_def = (current_val * 0.8, current_val * 1.2) if isinstance(current_val, (int, float)) and abs(current_val) > 1e-9 else (-1.0, 1.0)
            start_var, end_var, steps_var = tk.DoubleVar(value=start_def), tk.DoubleVar(value=end_def), tk.IntVar(value=5)
            entry_data.update({'start_var': start_var, 'end_var': end_var, 'steps_var': steps_var})
            entry_data['widgets'].update({'range_lbl_s': ttk.Label(row_frame, text="Start:"), 'range_ent_s': ttk.Entry(row_frame, textvariable=start_var, width=10), 'range_lbl_e': ttk.Label(row_frame, text="End:"), 'range_ent_e': ttk.Entry(row_frame, textvariable=end_var, width=10), 'range_lbl_st': ttk.Label(row_frame, text="Steps:"), 'range_spn_st': ttk.Spinbox(row_frame, from_=1, to=100, textvariable=steps_var, width=5)})
            steps_var.trace_add("write", self.update_total_cases)
        elif param_type == 'int':
            mode_var, start_var, end_var, steps_var, list_var = tk.StringVar(value="Range"), tk.DoubleVar(value=current_val), tk.DoubleVar(value=current_val+4), tk.IntVar(value=5), tk.StringVar(value=str(current_val))
            def update_int_widgets():
                is_range = mode_var.get() == "Range"
                for name, w in entry_data['widgets'].items():
                    if name.startswith('range_'): w.grid() if is_range else w.grid_remove()
                    if name.startswith('list_'): w.grid() if not is_range else w.grid_remove()
                self.update_total_cases()
            entry_data.update({'int_mode_var': mode_var, 'start_var': start_var, 'end_var': end_var, 'steps_var': steps_var, 'list_var': list_var, 'update_func': update_int_widgets})
            entry_data['widgets'].update({'rad_range': ttk.Radiobutton(row_frame, text="Range", variable=mode_var, value="Range", command=update_int_widgets), 'rad_list': ttk.Radiobutton(row_frame, text="List", variable=mode_var, value="List", command=update_int_widgets), 'range_lbl_s': ttk.Label(row_frame, text="Start:"), 'range_ent_s': ttk.Entry(row_frame, textvariable=start_var, width=8), 'range_lbl_e': ttk.Label(row_frame, text="End:"), 'range_ent_e': ttk.Entry(row_frame, textvariable=end_var, width=8), 'range_lbl_st': ttk.Label(row_frame, text="Steps:"), 'range_spn_st': ttk.Spinbox(row_frame, from_=1, to=100, textvariable=steps_var, width=5), 'list_lbl': ttk.Label(row_frame, text="List (CSV):"), 'list_ent': ttk.Entry(row_frame, textvariable=list_var, width=25)})
            steps_var.trace_add("write", self.update_total_cases); list_var.trace_add("write", self.update_total_cases)
        elif param_type == 'bool':
            bool_var = tk.StringVar(value="Vary (True & False)")
            entry_data.update({'bool_var': bool_var, 'widgets': {'bool_lbl': ttk.Label(row_frame, text="Value:"), 'bool_combo': ttk.Combobox(row_frame, textvariable=bool_var, values=["Vary (True & False)", "True", "False"], width=20)}})
            bool_var.trace_add("write", self.update_total_cases)
        elif param_type == 'option':
            options_var = tk.StringVar(value=f'"{current_val}"')
            entry_data.update({'options_var': options_var, 'widgets': {'opt_lbl': ttk.Label(row_frame, text="Options (CSV):"), 'opt_ent': ttk.Entry(row_frame, textvariable=options_var, width=30)}})
            options_var.trace_add("write", self.update_total_cases)

        entry_data['widgets']['info_lbl'] = ttk.Label(row_frame, text=f"[{param_info.get('unit', '')}] (Type: {param_type}, Current: {current_val})", foreground='gray')
        entry_data['widgets']['remove_btn'] = ttk.Button(row_frame, text="Remove", command=lambda e=entry_data: self.remove_parameter(e))
        row_frame.columnconfigure(8, weight=1)
        self.parameter_entries.append(entry_data)
        self.on_distribution_change()

    def remove_parameter(self, entry_to_remove):
        entry_to_remove['frame'].destroy(); self.parameter_entries.remove(entry_to_remove); self.update_total_cases()

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
        total = 0
        try:
            if dist_mode == "grid_search":
                total = 1 if self.parameter_entries else 0
                for entry in self.parameter_entries:
                    if entry['param_info']['type'] == 'float': total *= entry['steps_var'].get()
                    elif entry['param_info']['type'] == 'int':
                        if entry['int_mode_var'].get() == 'Range': total *= entry['steps_var'].get()
                        else: total *= max(1, len([i for i in entry['list_var'].get().split(',') if i.strip()]))
                    elif entry['param_info']['type'] == 'bool': total *= 2 if "Vary" in entry['bool_var'].get() else 1
                    elif entry['param_info']['type'] == 'option': total *= max(1, len([o for o in entry['options_var'].get().split(',') if o.strip()]))
            elif dist_mode == "csv_columnwise":
                if self.parameter_entries: total = len([i for i in self.parameter_entries[0]['csv_var'].get().split(',') if i.strip()])
        except (tk.TclError, ValueError, IndexError):
            total = 0 # In case of invalid input during typing
        self.num_cases.set(total if dist_mode != 'latin_hypercube' else self.num_cases.get())

def main():
    """Main function to launch the Tkinter application."""
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError):
        pass # For non-Windows systems
    
    root = tk.Tk()
    app = OpenFASTTestCaseGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()