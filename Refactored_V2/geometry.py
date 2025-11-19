import math

# --- Global Constants ---
STEEL_DENSITY = 12180  # kg/m^3
WATER_DENSITY = 1025 # kg/m^3

# --- Default Platform Data ---
# These constants define the default properties for the bracing system and ballast.
# They are used if the user does not provide their own values.
DEFAULT_BRACES_DATA = {
    'radius': 0.8, 
    'thickness': 0.0175, 
    'upper_plane_z': 24.3,
    'lower_plane_z': -20.0, 
    'upper_juncture_radius': 22.87,
    'lower_juncture_radius': 16.87, 
    'cross_brace_main_col_z': -18.0,
}
DEFAULT_BALLAST_MASS = 9.628E+6

class SemiSubmersiblePlatform:
    """
    Calculates the structural properties of a semi-submersible platform.

    The model is composed of a main central column, three outer columns,
    and a connecting brace system based on the OC4 DeepCwind topology.
    It now also calculates the mass moments of inertia.
    """
    def __init__(self, main_column_props, upper_columns_props, base_columns_props, braces_props):
        """
        Initializes the platform with dictionaries of properties.
        """
        self.main_column_props = main_column_props
        self.upper_columns_props = upper_columns_props
        self.base_columns_props = base_columns_props
        self.braces_props = braces_props
        
        self.components = []
        self._build_platform()

    def _calculate_cylinder_properties(self, name, radius, height, thickness, z_bottom, x=0, y=0):
        """
        Calculates properties for a single vertical hollow cylinder, including
        mass, CG, and local moments of inertia about its own CG.
        """
        outer_radius = radius
        inner_radius = radius - thickness
        
        outer_volume = math.pi * outer_radius**2 * height
        inner_volume = math.pi * inner_radius**2 * height
        steel_volume = outer_volume - inner_volume
        mass = steel_volume * STEEL_DENSITY
        
        cg_z = z_bottom + height / 2.0
        cg = (x, y, cg_z)

        # Inertia of a hollow cylinder about its own CG
        # I_xx and I_yy are about transverse axes through the CG
        # I_zz is about the longitudinal (vertical) axis through the CG
        I_local_zz = 0.5 * mass * (outer_radius**2 + inner_radius**2)
        I_local_xx_yy = (1/12.0) * mass * (3 * (outer_radius**2 + inner_radius**2) + height**2)
        
        return {
            'mass': mass, 
            'cg': cg, 
            'name': name,
            # Store local inertia as (Ixx, Iyy, Izz)
            'I_local': (I_local_xx_yy, I_local_xx_yy, I_local_zz)
        }

    def _calculate_brace_properties(self, name, start_pos, end_pos, radius, thickness):
        """
        Calculates properties for a single diagonal/horizontal brace.
        NOTE: Local inertia is approximated by treating it as a cylinder, which is
        a standard and reasonable simplification for this type of analysis.
        """
        length = math.sqrt((end_pos[0] - start_pos[0])**2 + 
                           (end_pos[1] - start_pos[1])**2 + 
                           (end_pos[2] - start_pos[2])**2)
        
        outer_radius = radius
        inner_radius = radius - thickness
        
        outer_volume = math.pi * outer_radius**2 * length
        inner_volume = math.pi * inner_radius**2 * length
        steel_volume = outer_volume - inner_volume
        mass = steel_volume * STEEL_DENSITY
        
        cg = ((start_pos[0] + end_pos[0]) / 2.0, 
              (start_pos[1] + end_pos[1]) / 2.0, 
              (start_pos[2] + end_pos[2]) / 2.0)

        # Approximate local inertia using cylinder formulas with length L
        I_local_longitudinal = 0.5 * mass * (outer_radius**2 + inner_radius**2)
        I_local_transverse = (1/12.0) * mass * (3 * (outer_radius**2 + inner_radius**2) + length**2)
        
        # This is a simplification. A full inertia tensor rotation is complex and
        # often unnecessary as the parallel axis theorem term (md^2) dominates.
        # We store the transverse and longitudinal values. The get_inertia method will use them.
        return {
            'mass': mass, 
            'cg': cg, 
            'name': name,
            # For a non-axis-aligned brace, we can't assign Ixx, Iyy, Izz directly.
            # However, for the parallel axis theorem, we can approximate I_cm for all axes
            # using the transverse value, as it's larger and provides a conservative estimate.
            'I_local': (I_local_transverse, I_local_transverse, I_local_longitudinal)
        }

    def _build_platform(self):
        """Constructs the full platform by building columns and the bracing system."""
        self._build_columns()
        self._build_bracing_system()
            
    def _build_columns(self):
        """Builds the main and outer columns."""
        self.components.append(self._calculate_cylinder_properties(
            'Main Column', **self.main_column_props
        ))

        dist = self.upper_columns_props['distance']
        angles = [0, 120, 240]
        
        upper_props_for_calc = self.upper_columns_props.copy()
        upper_props_for_calc.pop('distance')
        base_props_for_calc = self.base_columns_props.copy()

        for i, angle_deg in enumerate(angles, 1):
            angle_rad = math.radians(angle_deg)
            x = dist * math.cos(angle_rad)
            y = dist * math.sin(angle_rad)
            
            self.components.append(self._calculate_cylinder_properties(
                f'Upper Column {i}', **upper_props_for_calc, x=x, y=y
            ))
            self.components.append(self._calculate_cylinder_properties(
                f'Base Column {i}', **base_props_for_calc, x=x, y=y
            ))

    def _build_bracing_system(self):
        """Builds the complex bracing system based on the OC4 topology."""
        p = self.braces_props
        mc_radius = self.main_column_props['radius']
        angles_deg = [0, 120, 240]
        angles_rad = [math.radians(deg) for deg in angles_deg]

        main_col_upper_y_pts = [(mc_radius * math.cos(rad), mc_radius * math.sin(rad), p['upper_plane_z']) for rad in angles_rad]
        main_col_lower_y_pts = [(mc_radius * math.cos(rad), mc_radius * math.sin(rad), p['lower_plane_z']) for rad in angles_rad]
        main_col_cross_brace_pts = [(mc_radius * math.cos(rad), mc_radius * math.sin(rad), p['cross_brace_main_col_z']) for rad in angles_rad]
        
        upper_juncture_pts = [(p['upper_juncture_radius'] * math.cos(rad), p['upper_juncture_radius'] * math.sin(rad), p['upper_plane_z']) for rad in angles_rad]
        lower_juncture_pts = [(p['lower_juncture_radius'] * math.cos(rad), p['lower_juncture_radius'] * math.sin(rad), p['lower_plane_z']) for rad in angles_rad]

        for i in range(3):
            j = (i + 1) % 3 
            self.components.append(self._calculate_brace_properties(f'Y Pontoon, Upper {i+1}', main_col_upper_y_pts[i], upper_juncture_pts[i], p['radius'], p['thickness']))
            self.components.append(self._calculate_brace_properties(f'Y Pontoon, Lower {i+1}', main_col_lower_y_pts[i], lower_juncture_pts[i], p['radius'], p['thickness']))
            self.components.append(self._calculate_brace_properties(f'Cross Brace {i+1}', main_col_cross_brace_pts[i], upper_juncture_pts[i], p['radius'], p['thickness']))
            self.components.append(self._calculate_brace_properties(f'Delta Pontoon, Upper {i+1}', upper_juncture_pts[i], upper_juncture_pts[j], p['radius'], p['thickness']))
            self.components.append(self._calculate_brace_properties(f'Delta Pontoon, Lower {i+1}', lower_juncture_pts[i], lower_juncture_pts[j], p['radius'], p['thickness']))

    def get_structural_properties(self):
        """Calculates total mass and combined center of gravity for the steel structure."""
        total_mass = 0
        moment_x, moment_y, moment_z = 0, 0, 0

        for comp in self.components:
            mass = comp['mass']
            cg = comp['cg']
            total_mass += mass
            moment_x += mass * cg[0]
            moment_y += mass * cg[1]
            moment_z += mass * cg[2]

        if total_mass == 0:
            return {'weight': 0, 'cg': (0, 0, 0)}

        combined_cg = (moment_x / total_mass, moment_y / total_mass, moment_z / total_mass)
        return {'weight': total_mass, 'cg': combined_cg}

    def get_total_properties_with_ballast(self, ballast_mass, ballast_cg_z):
        """Calculates total properties including ballast."""
        structural = self.get_structural_properties()
        total_mass = structural['weight'] + ballast_mass
        
        moment_x = structural['weight'] * structural['cg'][0]
        moment_y = structural['weight'] * structural['cg'][1]
        moment_z = (structural['weight'] * structural['cg'][2]) + (ballast_mass * ballast_cg_z)
        
        if total_mass == 0:
            return {'weight': 0, 'cg': (0, 0, 0)}
            
        overall_cg = (moment_x / total_mass, moment_y / total_mass, moment_z / total_mass)
        return {'weight': total_mass, 'cg': overall_cg}

    def get_structural_inertia_about_cg(self, overall_cg):
        """
        Calculates the moments of inertia for the entire steel structure about the
        platform's overall center of gravity using the Parallel Axis Theorem.
        """
        I_roll, I_pitch, I_yaw = 0.0, 0.0, 0.0
        
        for comp in self.components:
            m = comp['mass']
            comp_cg = comp['cg']
            I_local = comp['I_local'] # (I_xx, I_yy, I_zz) for the component's own CG

            # Distances for Parallel Axis Theorem
            dx = comp_cg[0] - overall_cg[0]
            dy = comp_cg[1] - overall_cg[1]
            dz = comp_cg[2] - overall_cg[2]
            
            # Apply theorem: I = I_cm + m*d^2
            # Roll is rotation about platform X-axis
            I_roll += I_local[0] + m * (dy**2 + dz**2)
            # Pitch is rotation about platform Y-axis
            I_pitch += I_local[1] + m * (dx**2 + dz**2)
            # Yaw is rotation about platform Z-axis
            I_yaw += I_local[2] + m * (dx**2 + dy**2)
            
        return {'roll': I_roll, 'pitch': I_pitch, 'yaw': I_yaw}


def calculate_semisub_properties(
    MC_radius, MC_height_above_SWL, MC_height_below_SWL, MC_thickness,
    distance,
    UC_radius, UC_height_above_SWL, UC_height_below_SWL, UC_thickness,
    BC_radius, BC_height, BC_thickness,
    braces_params=None, 
    ballast_mass=None, 
    print_results=True
):
    """
    Calculates and returns all platform properties, including mass, CG, and moments of inertia.
    
    Args:
        MC_radius (float): Radius of the main central column.
        MC_height_above_SWL (float): Height of the main column above Still Water Level (SWL).
        MC_height_below_SWL (float): Draft of the main column below SWL.
        MC_thickness (float): Wall thickness of the main column.
        distance (float): Horizontal distance from the platform center to the center of the outer columns.
        UC_radius (float): Radius of the upper part of the outer columns.
        UC_height_above_SWL (float): Height of the upper columns above SWL.
        UC_height_below_SWL (float): Draft of the entire outer column assembly below SWL.
        UC_thickness (float): Wall thickness of the upper columns.
        BC_radius (float): Radius of the base part of the outer columns.
        BC_height (float): Height of the base columns.
        BC_thickness (float): Wall thickness of the base columns.
        braces_params (dict, optional): A dictionary for the bracing system. Defaults to None, which triggers use of DEFAULT_BRACES_DATA.
        ballast_mass (float, optional): The total mass of the ballast. Defaults to None, which triggers use of DEFAULT_BALLAST_MASS.
        print_results (bool): If True, prints a formatted summary of the results.
        print_results (bool): If True, prints a formatted summary of the results.

    Returns:
        dict: A dictionary containing all calculated properties. To retrieve values:
        total_props_dict = platform_results['total_properties_with_ballast']
        total_weight = total_props_dict['weight']
        overall_cg_tuple = total_props_dict['cg']
        overall_cg_x = overall_cg_tuple[0]
        overall_cg_y = overall_cg_tuple[1]
        overall_cg_z = overall_cg_tuple[2]
        mooring_points_list = platform_results['mooring_points']
        # Example of extracting one point's data
        fairlead_1_x = mooring_points_list[0]['x']
        fairlead_1_y = mooring_points_list[0]['y']
        fairlead_1_z = mooring_points_list[0]['z']
        fairlead_2_x = mooring_points_list[1]['x']
        fairlead_2_y = mooring_points_list[1]['y']
        fairlead_2_z = mooring_points_list[1]['z']
        fairlead_3_x = mooring_points_list[2]['x']
        fairlead_3_y = mooring_points_list[2]['y']
        fairlead_3_z = mooring_points_list[2]['z']
        
        # --- NEW: Get inertia values ---
        total_inertia = platform_results['total_inertia_about_cm']
        PtfmRIner = total_inertia['roll']   # Roll inertia (I_xx)
        PtfmPIner = total_inertia['pitch']  # Pitch inertia (I_yy)
        PtfmYIner = total_inertia['yaw']    # Yaw inertia (I_zz)
    """
    # --- 0. Handle Default Parameters ---
    if braces_params is None: braces_params = DEFAULT_BRACES_DATA
    if ballast_mass is None: ballast_mass = DEFAULT_BALLAST_MASS

    # --- 1. Convert User Inputs to Internal Data Structures ---
    mc_z_top = MC_height_above_SWL
    mc_z_bottom = -MC_height_below_SWL
    main_column_params = {'radius': MC_radius, 'height': mc_z_top - mc_z_bottom, 'thickness': MC_thickness, 'z_bottom': mc_z_bottom}
    bc_z_bottom = -UC_height_below_SWL 
    bc_z_top = bc_z_bottom + BC_height
    base_columns_params = {'radius': BC_radius, 'height': BC_height, 'thickness': BC_thickness, 'z_bottom': bc_z_bottom}
    uc_z_top = UC_height_above_SWL
    uc_z_bottom = bc_z_top
    upper_columns_params = {'distance': distance, 'radius': UC_radius, 'height': uc_z_top - uc_z_bottom, 'thickness': UC_thickness, 'z_bottom': uc_z_bottom}
    
    # --- 2. Perform Mass and CG Calculations ---
    platform = SemiSubmersiblePlatform(main_column_params, upper_columns_params, base_columns_params, braces_params)
    structural_props = platform.get_structural_properties()
    ballast_cg_z = base_columns_params['z_bottom'] + (base_columns_params['height'] / 2.0)
    total_props = platform.get_total_properties_with_ballast(ballast_mass, ballast_cg_z)
    overall_cg = total_props['cg']

    # --- 3. Perform Inertia Calculations ---
    # 3a. Calculate structural inertia about the overall CG
    structural_inertia = platform.get_structural_inertia_about_cg(overall_cg)
    
    # 3b. Calculate ballast inertia about the overall CG (treating ballast as a point mass)
    # The local inertia of the ballast is considered zero. We only need the parallel axis term.
    ballast_cg_pos = (0, 0, ballast_cg_z)
    dx_b = ballast_cg_pos[0] - overall_cg[0]
    dy_b = ballast_cg_pos[1] - overall_cg[1]
    dz_b = ballast_cg_pos[2] - overall_cg[2]
    
    ballast_inertia_roll = ballast_mass * (dy_b**2 + dz_b**2)
    ballast_inertia_pitch = ballast_mass * (dx_b**2 + dz_b**2)
    ballast_inertia_yaw = ballast_mass * (dx_b**2 + dy_b**2)

    # 3c. Sum structural and ballast inertias for the total
    total_inertia = {
        'roll': structural_inertia['roll'] + ballast_inertia_roll,
        'pitch': structural_inertia['pitch'] + ballast_inertia_pitch,
        'yaw': structural_inertia['yaw'] + ballast_inertia_yaw
    }

    # --- 4. Calculate Mooring System Fairlead Coordinates ---
    dist_mooring = upper_columns_params['distance'] + upper_columns_params['radius']
    z_fairlead = upper_columns_params['z_bottom']
    angles_mooring = [60, 180, 300]
    mooring_points = []
    for i, angle_deg in enumerate(angles_mooring, 1):
        angle_rad = math.radians(angle_deg)
        x = dist_mooring * math.cos(angle_rad)
        y = dist_mooring * math.sin(angle_rad)
        mooring_points.append({'id': i, 'x': x, 'y': y, 'z': z_fairlead})

    # --- 5. Collate and Return All Results ---
    results = {
        'structural_properties': structural_props,
        'total_properties_with_ballast': total_props,
        'total_inertia_about_cm': total_inertia, # <-- NEW
        'mooring_points': mooring_points,
        'ballast_info': {'mass': ballast_mass, 'cg_z': ballast_cg_z}
    }

    # --- 6. Optionally Print Results ---
    if print_results:
        print("--- Semi-Submersible Platform Calculation Results ---")
        print("\n--- Structural Properties (Steel Only) ---")
        print(f"Structural Steel Weight: {structural_props['weight']:,.2f} kg")
        print("Structural Center of Gravity (CG) relative to SWL at (0,0,0):")
        print(f"  X: {structural_props['cg'][0]:.4f} m, Y: {structural_props['cg'][1]:.4f} m, Z: {structural_props['cg'][2]:.4f} m")
        
        print("\n--- Total Properties (Including Ballast) ---")
        print(f"Ballast Mass: {ballast_mass:,.2f} kg")
        print(f"Total Platform Weight: {total_props['weight']:,.2f} kg")
        print("Overall Center of Gravity (CG) relative to SWL at (0,0,0):")
        print(f"  X: {total_props['cg'][0]:.4f} m, Y: {total_props['cg'][1]:.4f} m, Z: {total_props['cg'][2]:.4f} m")

        print("\n--- Total Platform Inertia about CM ---")
        print(f"Roll Inertia (PtfmRIner):  {total_inertia['roll']:,.2f} kg m^2")
        print(f"Pitch Inertia (PtfmPIner): {total_inertia['pitch']:,.2f} kg m^2")
        print(f"Yaw Inertia (PtfmYIner):   {total_inertia['yaw']:,.2f} kg m^2")

        print("\n--- Mooring System ---")
        print("Mooring Point Coordinates (Fairleads):")
        for point in mooring_points:
            print(f"  Point {point['id']}: (X: {point['x']:.2f}, Y: {point['y']:.2f}, Z: {point['z']:.2f}) m")
            
    return results

def main():
    """
    Main execution block demonstrating the use of the updated calculator.
    """
    platform_geometry = {
        "MC_radius": 6.5,
        "MC_height_above_SWL": 10.0,
        "MC_height_below_SWL": 20.0,
        "MC_thickness": 0.08,
        "distance": 50,
        "UC_radius": 12,
        "UC_height_above_SWL": 12.0,
        "UC_height_below_SWL": 20.0,
        "UC_thickness": 0.06,
        "BC_radius": 24.0,
        "BC_height": 6.0,
        "BC_thickness": 0.06,
    }

    print("="*60)
    print(">>> DEMO: Calculating all properties with default values.")
    print("="*60)
    
    # Call the function with only the required geometry.
    results = calculate_semisub_properties(
        **platform_geometry,
        print_results=False # We will print programmatically
    )
    
    # --- DEMONSTRATION: Programmatic access to all results ---
    print("\n" + "="*60)
    print(">>> DEMO: Programmatic access to the results.")
    print("="*60)
    
    # Extracting mass and CG
    total_weight = results['total_properties_with_ballast']['weight']
    overall_cg = results['total_properties_with_ballast']['cg']
    
    # Extracting inertia (the new values)
    total_inertia_dict = results['total_inertia_about_cm']
    PtfmRIner = total_inertia_dict['roll']
    PtfmPIner = total_inertia_dict['pitch']
    PtfmYIner = total_inertia_dict['yaw']

    print(f"Total Platform Weight: {total_weight:,.2f} kg")
    print(f"Overall CG (X, Y, Z): ({overall_cg[0]:.3f}, {overall_cg[1]:.3f}, {overall_cg[2]:.3f}) m")
    print("\nExtracted Inertia Values:")
    print(f"  PtfmRIner (Roll):  {PtfmRIner:,.2f} kg m^2")
    print(f"  PtfmPIner (Pitch): {PtfmPIner:,.2f} kg m^2")
    print(f"  PtfmYIner (Yaw):   {PtfmYIner:,.2f} kg m^2")
    
    # Due to the platform's symmetry, Roll and Pitch inertia should be identical.
    # This serves as a good sanity check.
    print(f"\nSanity Check: Difference between Roll and Pitch Inertia = {abs(PtfmRIner - PtfmPIner):.4f}")


if __name__ == "__main__":
    main()