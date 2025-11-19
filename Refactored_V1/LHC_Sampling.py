import numpy as np
import itertools
import csv
import os

def generate_parameter_combinations(num_levels):
    """
    Generates a full factorial design for the given simulation parameters.

    This function defines the parameter ranges, calculates evenly spaced values
    (levels) for each, and then computes all possible combinations.

    Args:
        num_levels (int): The number of evenly spaced values to generate for
                          each parameter range (must be 2 or more).

    Returns:
        A list of lists, where each inner list represents a unique
        combination of parameter values, including a header row.
    """
    if not isinstance(num_levels, int) or num_levels < 2:
        raise ValueError("Number of levels must be an integer of 2 or greater.")

    # 1. Define parameter names and their [start, end] ranges.
    parameters = {
        'HWindSpeed': [1, 100],
        'WaveHs': [1, 18]
    }

    # 2. Generate evenly spaced values (levels) for each parameter.
    levels = {
        name: np.linspace(start, end, num_levels)
        for name, (start, end) in parameters.items()
    }
    
    # 3. Create the list of parameter value lists for the Cartesian product.
    parameter_value_lists = [levels[name] for name in parameters.keys()]

    # 4. Generate the full factorial combinations (Cartesian product).
    product = list(itertools.product(*parameter_value_lists))
    
    total_combinations = len(product)
    print(f"Calculated {total_combinations} combinations for {num_levels} levels.")

    # 5. Prepare the final output with header.
    header = list(parameters.keys())
    results = [list(row) for row in product]
    
    return [header] + results

def transpose_data(data):
    """
    Transposes a list of lists (swaps rows and columns).

    Args:
        data (list of lists): The 2D data structure to transpose.

    Returns:
        A transposed list of lists.
    """
    print("Transposing data (swapping rows and columns)...")
    # The zip(*data) idiom is a concise and efficient way to transpose.
    # We convert the resulting tuples from zip back into lists.
    transposed = list(map(list, zip(*data)))
    return transposed

def save_to_csv_file(data, filename):
    """
    Writes the provided data to a CSV file.

    Args:
        data (list of lists): The data to write.
        filename (str): The name of the output CSV file.
    """
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        
        full_path = os.path.abspath(filename)
        num_rows = len(data)
        num_cols = len(data[0]) if num_rows > 0 else 0
        print(f"\nSuccessfully exported {num_rows} rows and {num_cols} columns to:\n{full_path}")

    except IOError as e:
        print(f"Error: Could not write to file {filename}. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file export: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # --- USER SETTINGS ---
    # A higher number will result in a much wider file.
    # 3 levels = 81 columns; 5 levels = 625 columns.
    NUMBER_OF_LEVELS = 15
    
    # Define the name for the output file.
    OUTPUT_FILENAME = "parameter_combinations_transposed.csv"
    # --- END OF USER SETTINGS ---

    try:
        # 1. Generate the parameter data in standard format (rows are combinations)
        print("Generating parameter combinations...")
        parameter_data = generate_parameter_combinations(NUMBER_OF_LEVELS)
        
        # 2. Transpose the data (rows become parameters, columns become combinations)
        transposed_data = transpose_data(parameter_data)
        
        # 3. Save the transposed data to a CSV file
        print(f"Exporting transposed data to '{OUTPUT_FILENAME}'...")
        save_to_csv_file(transposed_data, OUTPUT_FILENAME)

    except ValueError as e:
        print(f"Error: {e}")