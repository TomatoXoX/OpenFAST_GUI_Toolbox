import geometry as ssc
import math
platform_geometry = {
    "MC_radius": 3.25,
    "MC_height_above_SWL": 12.0,
    "MC_height_below_SWL": 20.0,
    "MC_thickness": 0.03,
    "distance": 28.8675,
    "UC_radius": 6.0,
    "UC_height_above_SWL": 12.0,
    "UC_height_below_SWL": 20.0,
    "UC_thickness": 0.06,
    "BC_radius": 12.0,
    "BC_height": 6.0,
    "BC_thickness": 0.06,
}

# Step 3: Call the calculation function from the imported module.
# We will use the default brace and ballast parameters for this analysis.
# We set print_results=False because this program will handle its own output.
print("Running platform calculations...")
platform_results = ssc.calculate_semisub_properties(
    **platform_geometry,
    print_results=False  # We want to control the output in this script
)
print("Calculations complete. Extracting results...")

# Step 4: Extract each value from the results dictionary into its own variable.
# The `platform_results` dictionary has a nested structure. We navigate it to get the data.

# --- Extracting Structural Properties ---
structural_props_dict = platform_results['structural_properties']
structural_weight = structural_props_dict['weight']
structural_cg_tuple = structural_props_dict['cg']
structural_cg_x = structural_cg_tuple[0]
structural_cg_y = structural_cg_tuple[1]
structural_cg_z = structural_cg_tuple[2]

# --- Extracting Total Properties (with Ballast) ---
total_props_dict = platform_results['total_properties_with_ballast']
total_weight = total_props_dict['weight']
overall_cg_tuple = total_props_dict['cg']
overall_cg_x = overall_cg_tuple[0]
overall_cg_y = overall_cg_tuple[1]
overall_cg_z = overall_cg_tuple[2]

# --- Extracting Ballast Information ---
ballast_info_dict = platform_results['ballast_info']
ballast_mass = ballast_info_dict['mass']
ballast_cg_z = ballast_info_dict['cg_z']

# --- Extracting Mooring Point Information ---
# Mooring points are a list of dictionaries, so we can store them as a list
# or process them in a loop.
mooring_points_list = platform_results['mooring_points']
# Example of extracting one point's data
fairlead_1_x = mooring_points_list[0]['x']
fairlead_1_y = mooring_points_list[0]['y']
fairlead_1_z = mooring_points_list[0]['z']
print(fairlead_1_x, fairlead_1_y, fairlead_1_z)

# Step 5: Use the extracted variables for further analysis or reporting.
# Here, we will just print them to demonstrate that they have been successfully extracted.
print("\n--- ANALYSIS RESULTS ---")
print("\n[Platform Mass Properties]")
print(f"  - Structural Steel Weight: {structural_weight:,.2f} kg")
print(f"  - Ballast Mass:            {ballast_mass:,.2f} kg")
print(f"  - Total Displaced Mass:    {total_weight:,.2f} kg")

print("\n[Platform Center of Gravity (CG)]")
print(f"  - Structural CG (Z): {structural_cg_z:.4f} m")
print(f"  - Ballast CG (Z):    {ballast_cg_z:.4f} m")
print(f"  - Overall CG (Z):    {overall_cg_z:.4f} m")

print("\n[Mooring System Interface]")
# Here we can loop through the list we extracted
for point in mooring_points_list:
    print(f"  - Fairlead {point['id']} Coordinates: (X={point['x']:.2f}, Y={point['y']:.2f}, Z={point['z']:.2f})")

# --- Example of a simple follow-up calculation ---
# Let's calculate the Metacentric Height (a simplified version for demonstration)
# GM = KM - KG, where KG is our overall_cg_z
# Let's assume a pre-calculated KM (height of metacenter above keel) for this displacement.
# This is just a placeholder to show how you'd use the variables.
KM = -10.5  # Assumed value for demonstration
KG = overall_cg_z
metacentric_height_GM = KM - KG

print("\n[Stability Analysis (Example)]")
print(f"  - Assumed Metacenter Height (KM): {KM:.4f} m")
print(f"  - Calculated Center of Gravity (KG): {KG:.4f} m")
print(f"  - Resulting Metacentric Height (GM): {metacentric_height_GM:.4f} m")

if metacentric_height_GM > 0:
    print("  - Stability Check: Platform is stable.")
else:
    print("  - Stability Check: Platform is UNSTABLE.")