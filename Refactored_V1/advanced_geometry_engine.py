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
    def __init__(self, ed_path: Path, hd_path: Path, md_path: Path, ad_path: Path): # MODIFIED: Added ad_path
        self.log: List[str] = []
        self._log(f"Initializing geometry engine with specific module files.")

        # MODIFIED: Check for AeroDyn file existence
        if not all([ed_path.exists(), hd_path.exists(), md_path.exists(), ad_path.exists()]):
            raise FileNotFoundError(f"One or more engine input files not found: ED={ed_path.exists()}, HD={hd_path.exists()}, MD={md_path.exists()}, AD={ad_path.exists()}")

        self.ed_file_name = ed_path.name
        self.hd_file_name = hd_path.name
        self.md_file_name = md_path.name
        self.ad_file_name = ad_path.name # NEW

        self.ed_data = self._parse_file_to_dict_robust(ed_path)
        self.hd_data = self._parse_file_to_dict_robust(hd_path)
        self.md_lines = md_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        # NEW: Store AeroDyn path
        self.ad_path = ad_path

        self.initial_mass = self._get_val(self.ed_data, 'PtfmMass')
        self.initial_roll_inertia = self._get_val(self.ed_data, 'PtfmRIner')
        self.initial_pitch_inertia = self._get_val(self.ed_data, 'PtfmPIner')
        self.initial_yaw_inertia = self._get_val(self.ed_data, 'PtfmYIner')
        self.initial_volume = self._get_val(self.hd_data, 'PtfmVol0')
        self.initial_tower_base_z = self._get_val(self.ed_data, 'TowerBsHt')
        self.initial_platform_ref_z = self._get_val(self.hd_data, 'PtfmRefzt')

        self.initial_joints = self._parse_table_from_lines(self.hd_data, 'NJoints', ['id', 'x', 'y', 'z', 'type', 'overlap'])
        self.initial_fairleads = self._parse_fairleads_from_md()

        self._log("Initial discovery complete.")
        self._log(f"  - Initial Mass: {self.initial_mass:.2f}, Volume: {self.initial_volume:.2f}")
        self._log(f"  - Initial Tower Base Z: {self.initial_tower_base_z:.2f}")

    def generate_variation(self, height_scale: float = 1.0, diameter_scale: float = 1.0) -> Dict[str, Any]:
        self._log(f"Generating variation: H_scale={height_scale:.3f}, D_scale={diameter_scale:.3f}")
        
        scaled_joints = self.initial_joints.copy()
        for c in ['id','x','y','z','type','overlap']:
            if c in scaled_joints.columns:
                scaled_joints[c] = pd.to_numeric(scaled_joints[c], errors='coerce')

        scaled_joints['x'] *= float(diameter_scale)
        scaled_joints['y'] *= float(diameter_scale)
        scaled_joints['z'] *= float(height_scale)

        scaled_fairleads = np.asarray(self.initial_fairleads, dtype=float).copy()
        scaled_fairleads[:, 0] *= float(diameter_scale)
        scaled_fairleads[:, 1] *= float(diameter_scale)
        scaled_fairleads[:, 2] *= float(height_scale)
        
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
        
        return {
            "mass": new_mass, 
            "roll_inertia": new_roll_inertia, 
            "pitch_inertia": new_pitch_inertia, 
            "yaw_inertia": new_yaw_inertia, 
            "volume": new_volume, 
            "joints": scaled_joints, 
            "fairleads": scaled_fairleads, 
            "tower_base_z": scaled_tower_base_z, 
            "platform_ref_z": scaled_platform_ref_z,
            # NEW: Pass scaling factors to the update function
            "height_scale": height_scale,
            "diameter_scale": diameter_scale
        }

    def _log(self, msg: str): self.log.append(msg)
    
    def _get_val(self, data: Dict[str, str], key: str) -> float:
        line = data.get(key)
        if line: 
            try:
                return float(line.strip().split()[0])
            except (ValueError, IndexError):
                self._log(f"WARNING: Could not parse float from line for key '{key}': {line}")
                return 0.0
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

    def _parse_table_from_lines(self, data_dict: Dict[str, str], count_key: str, columns: List[str]) -> pd.DataFrame:
        all_lines = list(data_dict.values())
        line_with_count = data_dict.get(count_key)
        if not line_with_count:
            self._log(f"ERROR: Count key '{count_key}' not found for table parsing.")
            return pd.DataFrame(columns=columns)
        
        try:
            num_rows = int(line_with_count.strip().split()[0])
        except (ValueError, IndexError):
            self._log(f"ERROR: Could not parse row count for key '{count_key}'.")
            return pd.DataFrame(columns=columns)

        if num_rows == 0:
            return pd.DataFrame(columns=columns)
            
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
            if len(table_lines_data) >= num_rows: break
            if line.startswith('---') or 'MEMBER CROSS-SECTION' in line.upper(): break
            
            parts = line.split()
            if line and (parts[0].isdigit() or (parts[0].startswith('-') and parts[0][1:].isdigit())):
                if len(parts) >= len(columns):
                    table_lines_data.append(parts[:len(columns)])
                else:
                    self._log(f"  -> Skipping malformed line in table '{count_key}': '{line}'")

        if len(table_lines_data) != num_rows:
            self._log(f"WARNING: Found {len(table_lines_data)} rows but expected {num_rows} for table '{count_key}'.")

        df = pd.DataFrame(table_lines_data, columns=columns)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def _parse_fairleads_from_md(self) -> np.ndarray:
        fairleads = []
        in_points_section = False
        for line in self.md_lines:
            line_upper = line.strip().upper()
            if line_upper.startswith('---'): 
                in_points_section = 'POINTS' in line_upper
                continue
            if in_points_section and "VESSEL" in line_upper:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try: fairleads.append([float(p) for p in parts[2:5]])
                    except ValueError: continue
        return np.array(fairleads)

def _find_hydrodyn_joints_region(lines: list) -> tuple[int,int,int,int]:
    n_joints_idx = next(i for i,l in enumerate(lines) if re.search(r'\bNJoints\b', l))
    joint_header_idx = next(i for i,l in enumerate(lines[n_joints_idx+1:], start=n_joints_idx+1)
                            if re.search(r'\bJointID\b', l))
    units_idx = joint_header_idx + 1
    end_idx = None
    for i in range(units_idx+1, len(lines)):
        s = lines[i].strip()
        if s.startswith('---') or re.search(r'\bCROSS-SECTION\b', s, re.IGNORECASE) or re.search(r'\bMEMBER\b', s, re.IGNORECASE):
            end_idx = i
            break
    if end_idx is None:
        raise RuntimeError("Cannot find end of HydroDyn Joints table.")
    return n_joints_idx, joint_header_idx, units_idx, end_idx

def _fmt_joint_line(jid: int, x: float, y: float, z: float, axid: int, ovrlp: int) -> str:
    return f"{jid:6d}{x:13.5f}{y:13.5f}{z:13.5f}{axid:12d}{ovrlp:12d}"

def write_hydrodyn_joints_and_scalars(hd_path: Path, joints_df, volume: float, refzt: float, logger=None):
    lines = hd_path.read_text(encoding='utf-8', errors='ignore').splitlines()

    for i, l in enumerate(lines):
        if re.search(r'\bPtfmVol0\b', l):
            lines[i] = f"{volume:>12.5E}     PtfmVol0"
        elif re.search(r'\bPtfmRefzt\b', l):
            lines[i] = f"{refzt:>12.5f}         PtfmRefzt"

    n_joints_idx, header_idx, units_idx, end_idx = _find_hydrodyn_joints_region(lines)

    required_cols = ['id','x','y','z','type','overlap']
    for c in required_cols:
        if c not in joints_df.columns:
            raise ValueError(f"Missing column '{c}' in joints_df.")
        joints_df[c] = pd.to_numeric(joints_df[c], errors='coerce')

    out_lines = []
    for _, row in joints_df.iterrows():
        out_lines.append(_fmt_joint_line(int(row['id']), float(row['x']), float(row['y']), float(row['z']),
                                         int(row['type']), int(row['overlap'])))

    nj = len(out_lines)
    lines[n_joints_idx] = re.sub(r'^\s*\d+', f"{nj:>12d}", lines[n_joints_idx])

    new_lines = lines[:units_idx+1] + out_lines + lines[end_idx:]
    hd_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8', errors='ignore')

    if logger:
        logger(f"HydroDyn updated: NJoints={nj}, PtfmVol0={volume:.5E}, PtfmRefzt={refzt:.5f}")

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
            if "PtfmMass" in line:    lines[i] = f"{data['mass']:>15.7E}   PtfmMass"
            elif "PtfmRIner" in line: lines[i] = f"{data['roll_inertia']:>15.7E}   PtfmRIner"
            elif "PtfmPIner" in line: lines[i] = f"{data['pitch_inertia']:>15.7E}   PtfmPIner"
            elif "PtfmYIner" in line: lines[i] = f"{data['yaw_inertia']:>15.7E}   PtfmYIner"
            elif "TowerBsHt" in line: lines[i] = f"{data['tower_base_z']:>15.7E}   TowerBsHt"
            # NEW: Also scale TowerHt for consistency
            elif "TowerHt" in line:
                original_tower_ht = model._get_val(model.ed_data, 'TowerHt')
                scaled_tower_ht = original_tower_ht * data['height_scale']
                lines[i] = f"{scaled_tower_ht:>15.7E}   TowerHt"
        ed_path.write_text('\n'.join(lines) + '\n', encoding='utf-8', errors='ignore')
    except Exception as e:
        model._log(f"WARNING: Could not update ElastoDyn file. Error: {e}")

    # --- Update HydroDyn File ---
    try:
        hd_path = case_dir / model.hd_file_name
        write_hydrodyn_joints_and_scalars(
            hd_path=hd_path,
            joints_df=data['joints'],
            volume=float(data['volume']),
            refzt=float(data['platform_ref_z']),
            logger=model._log
        )
    except Exception as e:
        model._log(f"FATAL: Could not update HydroDyn file. Error: {e}\n{traceback.format_exc()}")

    # --- NEW: Update AeroDyn File ---
    try:
        ad_path = case_dir / model.ad_file_name
        lines = ad_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        
        # Find the start of the tower aerodynamics table
        table_start_idx = -1
        header_idx = -1
        num_twr_nds_line = ""
        for i, line in enumerate(lines):
            if "NumTwrNds" in line:
                num_twr_nds_line = line
            if "TwrElev" in line and "TwrDiam" in line:
                header_idx = i
                table_start_idx = i + 2 # Skip header and units lines
                break
        
        if table_start_idx != -1:
            num_nodes = int(num_twr_nds_line.strip().split()[0])
            height_scale = data['height_scale']
            diameter_scale = data['diameter_scale']
            
            for i in range(table_start_idx, table_start_idx + num_nodes):
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    # Scale TwrElev and TwrDiam
                    twr_elev = float(parts[0]) * height_scale
                    twr_diam = float(parts[1]) * diameter_scale
                    # Reconstruct the line, preserving other columns
                    parts[0] = f"{twr_elev:.7E}"
                    parts[1] = f"{twr_diam:.7E}"
                    lines[i] = "  ".join(parts)
            
            ad_path.write_text('\n'.join(lines) + '\n', encoding='utf-8', errors='ignore')
            model._log(f"Successfully scaled {num_nodes} nodes in AeroDyn file.")
        else:
            model._log("WARNING: AeroDyn tower properties table not found. File not modified.")

    except Exception as e:
        model._log(f"FATAL: Could not update AeroDyn file. Error: {e}\n{traceback.format_exc()}")


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
                model._log(f"WARNING: Mismatch in MoorDyn fairleads ({len(vessel_point_indices)} found vs {data['fairleads'].shape[0]} expected). Not updated.")
    except Exception as e:
        model._log(f"WARNING: Could not update MoorDyn file. Error: {e}")