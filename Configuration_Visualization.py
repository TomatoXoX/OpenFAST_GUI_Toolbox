import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import os
import json
from datetime import datetime
from pathlib import Path

class OpenFASTVisualizer:
    """
    Generalized visualizer for OpenFAST models.
    Reads a main .fst file and all associated input files to create a detailed
    3D visualization and generate comprehensive summary reports.
    """

    def __init__(self, fst_filepath):
        self.fst_filepath = Path(fst_filepath)
        self.base_dir = self.fst_filepath.parent

        # Initialize data dictionaries for each OpenFAST module
        self.fst_data = {}
        self.elastodyn_data = {}
        self.beamdyn_data = {}
        self.inflowwind_data = {}
        self.servodyn_data = {}
        self.hydrodyn_data = {}
        self.moordyn_data = {} # Unified dict for any mooring model
        self.tower_data = {}
        self.seastate_data = {}

    def _read_value(self, line, value_type=float):
        """Helper to extract the first value from a line."""
        parts = line.strip().split()
        if parts:
            try:
                if '"' in parts[0]: 
                    return parts[0].strip('"')
                return value_type(parts[0])
            except (ValueError, IndexError): 
                return None
        return None

    def _get_file_path(self, filename):
        """Resolves the full path of an input file relative to the main FST file."""
        if not filename or filename.lower() == "unused": 
            return None
        path = self.base_dir / Path(filename)
        return path if path.exists() else None

    def read_fst_file(self):
        """Reads the main .fst file to identify other input files."""
        print(f"--- Reading FST file: {self.fst_filepath.name} ---")
        with open(self.fst_filepath, 'r') as f:
            for line in f:
                if "CompMooring" in line: 
                    self.fst_data['CompMooring'] = self._read_value(line, int)
                if "MooringFile" in line: 
                    self.fst_data['MooringFile'] = self._get_file_path(self._read_value(line, str))
                if "EDFile" in line: 
                    self.fst_data['EDFile'] = self._get_file_path(self._read_value(line, str))
                if "BDBldFile(1)" in line: 
                    self.fst_data['BDBldFile'] = self._get_file_path(self._read_value(line, str))
                if "InflowFile" in line: 
                    self.fst_data['InflowFile'] = self._get_file_path(self._read_value(line, str))
                if "ServoFile" in line: 
                    self.fst_data['ServoFile'] = self._get_file_path(self._read_value(line, str))
                if "HydroFile" in line: 
                    self.fst_data['HydroFile'] = self._get_file_path(self._read_value(line, str))
                if "SeaStFile" in line: 
                    self.fst_data['SeaStFile'] = self._get_file_path(self._read_value(line, str))
                if "WtrDpth" in line: 
                    self.fst_data['WtrDpth'] = self._read_value(line)
        
        print(f"  - CompMooring: {self.fst_data.get('CompMooring')}")
        print(f"  - MooringFile: {self.fst_data.get('MooringFile')}")

    def read_moordyn(self, filepath):
        """Robustly reads MoorDyn input files."""
        print(f"Reading MoorDyn: {filepath.name}")
        points, lines_data = [], []
        current_section = None
        
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for line in f:
                # Skip empty lines and comments
                if not line.strip() or line.strip().startswith('#'):
                    continue
                    
                # Determine section by looking for header lines
                if "POINTS" in line.upper() and "---" in line: 
                    current_section = "POINTS"
                    continue
                if "LINES" in line.upper() and "---" in line: 
                    current_section = "LINES"
                    continue
                if "SOLVER" in line.upper() and "---" in line: 
                    current_section = None
                    break
                if "END" in line.upper():
                    break

                if current_section is None: 
                    continue
                
                parts = line.strip().split()
                if not parts: 
                    continue

                try:
                    if current_section == "POINTS":
                        # Check if first element is a number (ID)
                        if parts[0].replace('.','',1).replace('-','',1).isdigit() and len(parts) >= 5:
                            points.append({
                                'id': int(parts[0]), 
                                'type': parts[1], 
                                'x': float(parts[2]), 
                                'y': float(parts[3]), 
                                'z': float(parts[4])
                            })
                    elif current_section == "LINES":
                        # Check if first element is a number (ID)
                        if parts[0].isdigit() and len(parts) >= 4:
                            lines_data.append({
                                'id': int(parts[0]), 
                                'attach_a': int(parts[2]), 
                                'attach_b': int(parts[3])
                            })
                except (ValueError, IndexError) as e:
                    print(f"    Warning: Could not parse line: {line.strip()}")
                    continue
        
        self.moordyn_data = {'points': points, 'lines': lines_data}
        print(f"  - Found {len(points)} mooring points and {len(lines_data)} lines.")
        
        # Debug: print points and lines
        if points:
            print("  - Points:")
            for p in points:
                print(f"    ID: {p['id']}, Type: {p['type']}, Pos: ({p['x']:.1f}, {p['y']:.1f}, {p['z']:.1f})")
        if lines_data:
            print("  - Lines:")
            for l in lines_data:
                print(f"    ID: {l['id']}, Connects: {l['attach_a']} to {l['attach_b']}")

    def read_elastodyn(self, filepath):
        """Reads ElastoDyn data."""
        print(f"Reading ElastoDyn: {filepath.name}")
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'TipRad' in line: 
                self.elastodyn_data['blade_length'] = self._read_value(line)
            elif 'HubRad' in line: 
                self.elastodyn_data['hub_radius'] = self._read_value(line)
            elif 'TowerHt' in line: 
                self.elastodyn_data['tower_height'] = self._read_value(line)
            elif 'Twr2Shft' in line: 
                self.elastodyn_data['tower_to_shaft'] = self._read_value(line)
            elif 'OverHang' in line: 
                self.elastodyn_data['overhang'] = self._read_value(line)
            elif 'ShftTilt' in line: 
                self.elastodyn_data['shaft_tilt'] = self._read_value(line)
            elif 'PreCone(1)' in line: 
                self.elastodyn_data['precone'] = self._read_value(line)
            elif 'NumBl' in line: 
                self.elastodyn_data['num_blades'] = self._read_value(line, int)
            elif 'PtfmRefzt' in line: 
                self.elastodyn_data['platform_ref_z'] = self._read_value(line)

    def run_parsers(self):
        """Executes all file parsers based on the main FST file."""
        self.read_fst_file()
        
        if self.fst_data.get('EDFile'): 
            self.read_elastodyn(self.fst_data['EDFile'])
        
        # Read mooring data based on CompMooring flag
        comp_mooring = self.fst_data.get('CompMooring')
        mooring_file = self.fst_data.get('MooringFile')
        
        if comp_mooring == 3 and mooring_file:
            self.read_moordyn(mooring_file)
        elif comp_mooring == 2 and mooring_file:
            self.read_feamooring(mooring_file)
        elif mooring_file:
            print(f"  - Warning: Mooring model type {comp_mooring} is not yet supported for visualization.")

    def visualize_3d(self):
        """Creates the 3D visualization."""
        print("\n--- Creating 3D Visualization ---")
        
        tower_height = self.elastodyn_data.get('tower_height', 87.6)
        blade_length = self.elastodyn_data.get('blade_length', 63.0)
        hub_radius = self.elastodyn_data.get('hub_radius', 1.5)
        water_depth = self.fst_data.get('WtrDpth', 200.0)
        platform_draft = self.hydrodyn_data.get('PtfmDraft', 47.9)
        num_blades = self.elastodyn_data.get('num_blades', 3)
        overhang = self.elastodyn_data.get('overhang', -5.0)
        shaft_tilt_deg = self.elastodyn_data.get('shaft_tilt', 5.0)
        precone_deg = self.elastodyn_data.get('precone', -2.5)
        hub_height = tower_height + self.elastodyn_data.get('tower_to_shaft', 2.0)
        wave_height = self.seastate_data.get('WaveHs', 1.0)
        wave_period = self.seastate_data.get('WaveTp', 10.0)

        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')

        # Tower
        ax.plot([0, 0], [0, 0], [0, tower_height], 'k-', linewidth=10, label='Tower', zorder=4)

        # Hub
        hub_pos_global = np.array([overhang, 0, hub_height])
        ax.scatter(*hub_pos_global, c='darkblue', s=300, marker='o', label='Hub', zorder=10)
        
        # Blades
        rot_mat_tilt = self._get_rot_mat(shaft_tilt_deg, axis='y')
        blade_angles_deg = np.linspace(0, 360, num_blades, endpoint=False)
        
        for i, azim_deg in enumerate(blade_angles_deg):
            rot_mat_azim = self._get_rot_mat(azim_deg, axis='x')
            blade_points = []
            for r in np.linspace(hub_radius, blade_length, 25):
                local_pt = np.array([-r * np.sin(np.radians(precone_deg)), 0, r * np.cos(np.radians(precone_deg))])
                final_pt = hub_pos_global + rot_mat_tilt @ rot_mat_azim @ local_pt
                blade_points.append(final_pt)
            blade_coords = np.array(blade_points)
            ax.plot(blade_coords[:,0], blade_coords[:,1], blade_coords[:,2], 'b-', linewidth=6, 
                   label='Blades' if i==0 else "", zorder=9)

        # Platform
        platform_z = -platform_draft
        platform_radius = 9 
        theta = np.linspace(0, 2 * np.pi, 50)
        x_platform, y_platform = platform_radius * np.cos(theta), platform_radius * np.sin(theta)
        ax.plot(x_platform, y_platform, [0] * len(theta), 'gray', linewidth=3, zorder=5)
        ax.plot(x_platform, y_platform, [platform_z] * len(theta), 'gray', linewidth=3, zorder=5)
        for i in range(0, len(theta), 10):
            ax.plot([x_platform[i], x_platform[i]], [y_platform[i], y_platform[i]], 
                   [0, platform_z], 'dimgray', linewidth=2, zorder=5)

        # Mooring lines
        if self.moordyn_data.get('points') and self.moordyn_data.get('lines'):
            points = self.moordyn_data['points']
            lines = self.moordyn_data['lines']
            point_map = {p['id']: p for p in points}
            fairleads = [p for p in points if p['type'].lower() == 'vessel']
            anchors = [p for p in points if p['type'].lower() == 'fixed']

            for i, line_def in enumerate(lines):
                p_a = point_map.get(line_def['attach_a'])
                p_b = point_map.get(line_def['attach_b'])
                if p_a and p_b:
                    p1 = np.array([p_a['x'], p_a['y'], p_a['z']])
                    p2 = np.array([p_b['x'], p_b['y'], p_b['z']])
                    num_segments = 20
                    s = np.linspace(0, 1, num_segments)
                    line_pts = np.outer(1-s, p1) + np.outer(s, p2)
                    # Add catenary sag
                    sag_magnitude = 50.0  # Increased for better visualization
                    line_pts[:, 2] -= sag_magnitude * 4 * s * (1 - s)  # Parabolic sag
                    ax.plot(line_pts[:,0], line_pts[:,1], line_pts[:,2], 'r-', linewidth=2, 
                           label='Mooring Lines' if i==0 else "", zorder=6)
            
            if fairleads: 
                ax.scatter([p['x'] for p in fairleads], [p['y'] for p in fairleads], 
                          [p['z'] for p in fairleads], c='red', s=150, marker='o', 
                          label='Fairleads', edgecolors='k', zorder=7)
            if anchors: 
                ax.scatter([p['x'] for p in anchors], [p['y'] for p in anchors], 
                          [p['z'] for p in anchors], c='darkred', s=150, marker='v', 
                          label='Anchors', edgecolors='k', zorder=7)

        # Water and seabed
        max_dim = max(blade_length * 1.5, water_depth * 0.7, 500)  # Ensure we can see mooring
        x_grid, y_grid = np.meshgrid([-max_dim, max_dim], [-max_dim, max_dim])
        ax.plot_surface(x_grid, y_grid, np.zeros_like(x_grid), alpha=0.15, color='c', zorder=1)
        ax.plot_surface(x_grid, y_grid, np.full_like(x_grid, -water_depth), alpha=0.2, 
                       color='saddlebrown', zorder=1)
        ax.text(0, max_dim*0.9, 0, 'Water Surface', color='blue', ha='center', zorder=2)
        ax.text(0, max_dim*0.9, -water_depth, 'Seabed', color='black', ha='center', zorder=2)

        # --- MODIFIED CODE START ---
        # Coordinate system reference axes with dashed lines
        # These lines extend from the negative to positive limits of the plot area.
        z_lim_lower = -water_depth - 20
        z_lim_upper = hub_height + blade_length + 20

        # X-axis (dashed line)
        ax.plot([-max_dim, max_dim], [0, 0], [0, 0], 'k--', linewidth=1.5, alpha=0.8, zorder=1)
        ax.text(max_dim * 0.9, 0, 0, 'X', color='r', fontweight='bold', fontsize=14)
        
        # Y-axis (dashed line)
        ax.plot([0, 0], [-max_dim, max_dim], [0, 0], 'k--', linewidth=1.5, alpha=0.8, zorder=1)
        ax.text(0, max_dim * 0.9, 0, 'Y', color='g', fontweight='bold', fontsize=14)

        # Z-axis (dashed line)
        ax.plot([0, 0], [0, 0], [z_lim_lower, z_lim_upper], 'k--', linewidth=1.5, alpha=0.8, zorder=1)
        ax.text(0, 0, z_lim_upper * 0.9, 'Z', color='b', fontweight='bold', fontsize=14)
        # --- MODIFIED CODE END ---

        # Info text
        info_text = (f"System Information:\n"
                     f"Tower Height: {tower_height:.1f} m\n"
                     f"Blade Length: {blade_length:.1f} m\n"
                     f"Hub Height: {hub_height:.1f} m\n"
                     f"Water Depth: {water_depth:.1f} m\n"
                     f"Platform Draft: {platform_draft:.1f} m\n"
                     f"Num Blades: {num_blades}\n"
                     f"Num Mooring Lines: {len(self.moordyn_data.get('lines', []))}\n"
                     f"Wave Height: {wave_height:.1f} m\n"
                     f"Wave Period: {wave_period:.1f} s")
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9)
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=props)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('OpenFAST Model: 3D Visualization\nNREL 5MW Wind Turbine with OC4-DeepCwind Semi', 
                    fontsize=16, weight='bold')
        
        # Set equal aspect ratio and limits
        ax.set_box_aspect([1,1,0.5])  # Adjust aspect ratio for better visualization
        ax.set_xlim(-max_dim, max_dim)
        ax.set_ylim(-max_dim, max_dim)
        ax.set_zlim(-water_depth - 20, hub_height + blade_length + 20)
        ax.view_init(elev=25, azim=45)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def save_summary_file(self, filename_base='openfast_model_summary'):
        """Saves comprehensive summary reports."""
        print(f"\n--- Saving Summary Files ---")
        summary = {
            'general': {'FST File': self.fst_filepath.name, 'Generated': datetime.now().isoformat()},
            'turbine': self.elastodyn_data,
            'platform': self.hydrodyn_data,
            'mooring': self.moordyn_data,
            'environment': {'water_depth': self.fst_data.get('WtrDpth'), **self.seastate_data},
        }

        json_filename = f"{filename_base}.json"
        with open(json_filename, 'w') as f: 
            json.dump(summary, f, indent=4, default=str)
        print(f"JSON summary saved to: {json_filename}")

        txt_filename = f"{filename_base}.txt"
        with open(txt_filename, 'w') as f:
            f.write("="*80 + "\n           OPENFAST MODEL SUMMARY REPORT\n")
            f.write(f"           FST File: {summary['general']['FST File']}\n")
            f.write(f"           Generated: {summary['general']['Generated']}\n" + "="*80 + "\n\n")
            for category, data in summary.items():
                if category == 'general' or not data: 
                    continue
                f.write(f"--- {category.upper()} ---\n")
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list): 
                            f.write(f"  {key}: {len(value)} items\n")
                        elif value is not None: 
                            f.write(f"  {key}: {value}\n")
                f.write("\n")
        print(f"Text summary saved to: {txt_filename}")
        
    def _get_rot_mat(self, angle_deg, axis='x'):
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        if axis == 'x': 
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == 'y': 
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        if axis == 'z': 
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

if __name__ == "__main__":
    # NOTE: You will need to have the OpenFAST model files in the same directory
    # or provide the correct path to the main .fst file.
    # The necessary files for the NREL 5MW OC4-DeepCwind model can be found in the
    # OpenFAST r-test repository: https://github.com/OpenFAST/r-test
    main_fst_file = '5MW_OC4Semi_WSt_WavesWN.fst'
    if not os.path.exists(main_fst_file):
        print(f"Error: Main FST file not found at '{main_fst_file}'")
        print("Please ensure the file exists and the script is run from the correct directory.")
    else:
        try:
            viz = OpenFASTVisualizer(main_fst_file)
            viz.run_parsers()
            viz.visualize_3d()
            viz.save_summary_file()
            print("\nVisualization and summary generation complete.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()