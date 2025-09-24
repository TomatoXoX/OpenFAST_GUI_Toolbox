# In file: advanced_geometry_engine.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import traceback

class PlatformModel:
    """
    A class to parse, analyze, and generate variations of an OpenFAST
    semi-submersible platform's geometry.
    """
    def __init__(self, ed_path: Path, hd_path: Path, md_path: Path):
        self.log: List[str] = []
        self._log(f"Initializing geometry engine with specific module files.")

        if not all([ed_path.exists(), hd_path.exists(), md_path.exists()]):
            raise FileNotFoundError(f"One or more engine input files not found: ED={ed_path.exists()}, HD={hd_path.exists()}, MD={md_path.exists()}")

        self.ed_file_name = ed_path.name
        self.hd_file_name = hd_path.name
        self.md_file_name = md_path.name

        self.ed_data = self._parse_file_to_dict_robust(ed_path)
        self.hd_data = self._parse_file_to_dict_robust(hd_path)
        self.md_lines = md_path.read_text(encoding='utf-8', errors='ignore').splitlines()

        self.initial_mass = self._get_val(self.ed_data, 'PtfmMass')
        self.initial_roll_inertia = self._get_val(self.ed_data, 'PtfmRIner')
        self.initial_pitch_inertia = self._get_val(self.ed_data, 'PtfmPIner')
        self.initial_yaw_inertia = self._get_val(self.ed_data, 'PtfmYIner')
        self.initial_volume = self._get_val(self.hd_data, 'PtfmVol0')
        self.initial_tower_base_z = self._get_val(self.ed_data, 'TowerBsHt')
        self.initial_platform_ref_z = self._get_val(self.hd_data, 'PtfmRefzt')

        # --- THIS IS WHERE THE ERROR OCCURRED ---
        self.initial_joints = self._parse_table_from_lines(self.hd_data, 'NJoints', ['id', 'x', 'y', 'z', 'type', 'overlap'])
        self.initial_fairleads = self._parse_fairleads_from_md()

        self._log("Initial discovery complete.")
        self._log(f"  - Initial Mass: {self.initial_mass:.2f}, Volume: {self.initial_volume:.2f}")
        self._log(f"  - Initial Tower Base Z: {self.initial_tower_base_z:.2f}")

    def generate_variation(self, height_scale: float = 1.0, diameter_scale: float = 1.0) -> Dict[str, Any]:
        self._log(f"Generating variation: H_scale={height_scale:.3f}, D_scale={diameter_scale:.3f}")
        scaled_joints = self.initial_joints.copy()
        scaled_joints['x'] *= diameter_scale; scaled_joints['y'] *= diameter_scale; scaled_joints['z'] *= height_scale
        scaled_fairleads = self.initial_fairleads.copy()
        scaled_fairleads[:, 0] *= diameter_scale; scaled_fairleads[:, 1] *= diameter_scale; scaled_fairleads[:, 2] *= height_scale
        scaled_tower_base_z = self.initial_tower_base_z * height_scale
        scaled_platform_ref_z = self.initial_platform_ref_z * height_scale
        volume_scale_factor = (diameter_scale ** 2) * height_scale
        new_volume = self.initial_volume * volume_scale_factor
        new_mass = self.initial_mass * volume_scale_factor
        roll_pitch_inertia_scale = volume_scale_factor * ((diameter_scale**2 + height_scale**2) / 2)
        yaw_inertia_scale = volume_scale_factor * (diameter_scale**2)
        new_roll_inertia = self.initial_roll_inertia * roll_pitch_inertia_scale
        new_pitch_inertia = self.initial_pitch_inertia * roll_pitch_inertia_scale
        new_yaw_inertia = self.initial_yaw_inertia * yaw_inertia_scale
        self._log(f"  - New Mass: {new_mass:.2f}, New Volume: {new_volume:.2f}")
        self._log(f"  - New Tower Base Z: {scaled_tower_base_z:.2f}")
        return {"mass": new_mass, "roll_inertia": new_roll_inertia, "pitch_inertia": new_pitch_inertia, "yaw_inertia": new_yaw_inertia, "volume": new_volume, "joints": scaled_joints, "fairleads": scaled_fairleads, "tower_base_z": scaled_tower_base_z, "platform_ref_z": scaled_platform_ref_z}

    def _log(self, msg: str): self.log.append(msg)
    def _get_val(self, data: Dict[str, str], key: str) -> float:
        line = data.get(key)
        if line: return float(line.strip().split()[0])
        self._log(f"WARNING: Key '{key}' not found."); return 0.0

    def _parse_file_to_dict_robust(self, file_path: Path) -> Dict[str, str]:
        data = {}
        if not file_path.exists(): self._log(f"ERROR: Cannot find file to parse: {file_path}"); return data
        lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        pat = re.compile(r'^\s*"?([^"\s]+)"?\s+([A-Za-z_][A-Za-z0-9_()]*)(?:\s|$)')
        for line in lines:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith(('!', '#', '-', '=')): continue
            m = pat.match(line_strip)
            if m: data[m.group(2)] = line
        return data

    # --- DEFINITIVE FIX: A more robust table parser that knows when to stop ---
    def _parse_table_from_lines(self, data_dict: Dict[str, str], count_key: str, columns: List[str]) -> pd.DataFrame:
        """
        Parses a table from the raw file lines, stopping when it hits the next section.
        """
        all_lines = list(data_dict.values())
        line_with_count = data_dict.get(count_key)
        if not line_with_count:
            self._log(f"ERROR: Count key '{count_key}' not found for table parsing.")
            return pd.DataFrame(columns=columns)
        
        num_rows = int(line_with_count.strip().split()[0])
        
        try:
            start_search_index = all_lines.index(line_with_count) + 1
        except ValueError:
            self._log(f"CRITICAL: Count key line not found in its own dict for '{count_key}'.")
            return pd.DataFrame(columns=columns)
        
        table_start_index = -1
        for i in range(start_search_index, len(all_lines)):
            if '(-)' in all_lines[i] or '(m)' in all_lines[i] or 'JointID' in all_lines[i]:
                table_start_index = i + 1
                break
        
        if table_start_index == -1:
            self._log(f"ERROR: Could not find table header for '{count_key}'.")
            return pd.DataFrame(columns=columns)

        table_lines_data = []
        for i in range(table_start_index, len(all_lines)):
            line = all_lines[i].strip()
            
            # STOP CONDITION 1: We found the next section header.
            if line.startswith('---') or 'MEMBER CROSS-SECTION' in line.upper():
                self._log(f"  -> Found end of table for '{count_key}' at next section header.")
                break
            
            # STOP CONDITION 2: We have already found all the rows we need.
            if len(table_lines_data) >= num_rows:
                break

            # VALIDATION: Check if it looks like a valid data line for *this* table.
            if line and line.split()[0].isdigit():
                parts = line.split()
                # This explicitly rejects lines that don't have the correct number of columns.
                if len(parts) == len(columns):
                    table_lines_data.append(parts)
                else:
                    self._log(f"  -> Skipping malformed line in table '{count_key}': '{line}' (Expected {len(columns)} parts, found {len(parts)})")

        if len(table_lines_data) != num_rows:
            self._log(f"WARNING: Found {len(table_lines_data)} rows but expected {num_rows} for table '{count_key}'.")

        return pd.DataFrame(table_lines_data, columns=columns).apply(pd.to_numeric)

    def _parse_fairleads_from_md(self) -> np.ndarray:
        fairleads = []
        in_points_section = False
        for line in self.md_lines:
            line_upper = line.strip().upper()
            if line_upper.startswith('---'): in_points_section = 'POINTS' in line_upper; continue
            if in_points_section and "VESSEL" in line_upper:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try: fairleads.append([float(p) for p in parts[2:5]])
                    except ValueError: continue
        return np.array(fairleads)

def update_files_for_case(case_dir: Path, data: Dict[str, Any], model: 'PlatformModel'):
    """
    Writes the new geometric data to ElastoDyn, HydroDyn, and MoorDyn files
    using robust, content-aware methods that preserve file formatting.
    """
    # --- Update ElastoDyn File ---
    try:
        ed_path = case_dir / model.ed_file_name
        lines = ed_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        for i, line in enumerate(lines):
            if "PtfmMass" in line: lines[i] = f"{data['mass']:<14.7E}   PtfmMass"
            elif "PtfmRIner" in line: lines[i] = f"{data['roll_inertia']:<14.7E}   PtfmRIner"
            elif "PtfmPIner" in line: lines[i] = f"{data['pitch_inertia']:<14.7E}   PtfmPIner"
            elif "PtfmYIner" in line: lines[i] = f"{data['yaw_inertia']:<14.7E}   PtfmYIner"
            elif "TowerBsHt" in line: lines[i] = f"{data['tower_base_z']:<14.7E}   TowerBsHt"
        ed_path.write_text('\n'.join(lines) + '\n', encoding='utf-8', errors='ignore')
    except Exception as e:
        model._log(f"WARNING: Could not update ElastoDyn file. Error: {e}")

    # --- Update HydroDyn File ---
    try:
        hd_path = case_dir / model.hd_file_name
        lines = hd_path.read_text(encoding='utf-8', errors='ignore').splitlines()

        # Find the line indices for key parameters and the specific joints table
        n_joints_line_idx = next(i for i, line in enumerate(lines) if "NJoints" in line)
        
        # --- DEFINITIVE FIX: Search for the header *after* the NJoints line ---
        units_header_idx = next(i for i, line in enumerate(lines) if i > n_joints_line_idx and line.strip().startswith('(-)') and 'JointID' in lines[i-1])

        vol_line_idx = next(i for i, line in enumerate(lines) if "PtfmVol0" in line)
        ref_z_line_idx = next(i for i, line in enumerate(lines) if "PtfmRefzt" in line)

        original_n_joints = int(lines[n_joints_line_idx].strip().split()[0])
        table_start_index = units_header_idx + 1
        table_end_index = table_start_index + original_n_joints

        new_joint_lines = [f"{int(row['id']):<5d} {row['x']:>11.5f} {row['y']:>11.5f} {row['z']:>11.5f} {int(row['type']):>10d} {int(row['overlap']):>12d}" for _, row in data['joints'].iterrows()]

        lines[n_joints_line_idx] = f"{len(new_joint_lines):<13d}   NJoints"
        lines[vol_line_idx] = f"{data['volume']:<13.5E}   PtfmVol0"
        lines[ref_z_line_idx] = f"{data['platform_ref_z']:<13.5f}   PtfmRefzt"
        
        # Reconstruct the file content by splicing the new table
        final_lines = lines[:table_start_index] + new_joint_lines + lines[table_end_index:]
        hd_path.write_text('\n'.join(final_lines) + '\n', encoding='utf-8', errors='ignore')

    except Exception as e:
        model._log(f"FATAL: Could not update HydroDyn file. Error: {e}\n{traceback.format_exc()}")

    # --- Update MoorDyn File ---
    try:
        md_path = case_dir / model.md_file_name
        if md_path.exists() and 'fairleads' in data and data['fairleads'].size > 0:
            lines = md_path.read_text(encoding='utf-8', errors='ignore').splitlines()
            vessel_point_indices = [i for i, line in enumerate(lines) if "VESSEL" in line.upper()]
            if len(vessel_point_indices) == data['fairleads'].shape[0]:
                for i, line_idx in enumerate(vessel_point_indices):
                    parts = lines[line_idx].strip().split()
                    if len(parts) >= 5:
                        parts[2:5] = [f"{coord:.3f}" for coord in data['fairleads'][i]]
                        lines[line_idx] = " ".join(parts)
                md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8', errors='ignore')
            else:
                model._log(f"WARNING: Mismatch in MoorDyn fairleads. Not updated.")
    except Exception as e:
        model._log(f"WARNING: Could not update MoorDyn file. Error: {e}")