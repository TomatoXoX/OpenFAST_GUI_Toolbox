import pandas as pd
import re

def convert_openfast_to_csv(input_file, output_file):
    """
    Convert OpenFAST .out file to CSV format
    
    Parameters:
    input_file (str): Path to the .out file
    output_file (str): Path to save the .csv file
    """
    
    # Read the file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Find where the data starts
    data_start_idx = None
    column_names = []
    column_units = []
    
    for i, line in enumerate(lines):
        # Skip empty lines and header information
        if line.strip() == '':
            continue
            
        # Look for column headers (they come after description lines)
        if 'Time' in line and not line.startswith('Description'):
            # This is the column names line
            column_names = line.split()
            
            # The next line should be units
            if i + 1 < len(lines):
                column_units = lines[i + 1].strip().split()
                
            # Data starts after units line
            data_start_idx = i + 2
            break
    
    if data_start_idx is None:
        raise ValueError("Could not find data start in file")
    
    # Parse the data
    data_lines = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if line:  # Skip empty lines
            # Split by whitespace and convert to float
            values = line.split()
            if len(values) == len(column_names):
                data_lines.append([float(val) for val in values])
    
    # Create DataFrame
    df = pd.DataFrame(data_lines, columns=column_names)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Also save a metadata file with units
    metadata_file = output_file.replace('.csv', '_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write("Column Names and Units:\n")
        f.write("-" * 50 + "\n")
        for name, unit in zip(column_names, column_units):
            f.write(f"{name}: {unit}\n")
    
    print(f"Successfully converted {input_file} to {output_file}")
    print(f"Metadata saved to {metadata_file}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    return df

# Alternative version with more robust parsing
def convert_openfast_to_csv_robust(input_file, output_file):
    """
    More robust version that handles scientific notation and various formats
    """
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split into lines
    lines = content.strip().split('\n')
    
    # Find header lines
    header_found = False
    column_names = []
    column_units = []
    data_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and OpenFAST header info
        if not line or line.startswith('Predictions') or line.startswith('Description') or 'linked with' in line:
            i += 1
            continue
        
        # Check if this is the column header line
        if 'Time' in line and '\t' in line or ('Time' in line and len(line.split()) > 10):
            column_names = line.split()
            
            # Next line should be units
            if i + 1 < len(lines):
                column_units = lines[i + 1].strip().split()
            
            # Data starts after units
            i += 2
            header_found = True
            break
        
        i += 1
    
    if not header_found:
        raise ValueError("Could not find column headers in file")
    
    # Parse data lines
    while i < len(lines):
        line = lines[i].strip()
        if line:
            # Handle scientific notation
            values = line.split()
            if len(values) == len(column_names):
                try:
                    float_values = []
                    for val in values:
                        # Replace D with E for Fortran scientific notation if present
                        val = val.replace('D', 'E')
                        float_values.append(float(val))
                    data_lines.append(float_values)
                except ValueError as e:
                    print(f"Warning: Could not parse line {i}: {e}")
        i += 1
    
    # Create DataFrame
    df = pd.DataFrame(data_lines, columns=column_names)
    
    # Save to CSV with appropriate precision
    df.to_csv(output_file, index=False, float_format='%.6E')
    
    # Save metadata
    metadata_file = output_file.replace('.csv', '_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write("OpenFAST Output File Metadata\n")
        f.write("=" * 60 + "\n\n")
        
        # Extract description from original file
        for line in lines[:10]:
            if line.startswith('Description'):
                f.write(line + "\n\n")
                break
        
        f.write("Column Information:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Column':<20} {'Unit':<15} {'Description'}\n")
        f.write("-" * 60 + "\n")
        
        for name, unit in zip(column_names, column_units):
            f.write(f"{name:<20} {unit:<15}\n")
    
    print(f"\nConversion Summary:")
    print(f"{'Input file:':<20} {input_file}")
    print(f"{'Output CSV:':<20} {output_file}")
    print(f"{'Metadata file:':<20} {metadata_file}")
    print(f"{'Total rows:':<20} {len(df)}")
    print(f"{'Total columns:':<20} {len(df.columns)}")
    print(f"\nFirst few rows of data:")
    print(df.head())
    
    return df

# Main execution
if __name__ == "__main__":
    # Specify input and output files
    input_file = "5MW_OC4Semi_WSt_WavesWN.out"
    output_file = "5MW_OC4Semi_WSt_WavesWN.csv"
    
    try:
        # Use the robust version
        df = convert_openfast_to_csv_robust(input_file, output_file)
        
        # Optional: Create additional analysis files
        
        # 1. Summary statistics
        stats_file = output_file.replace('.csv', '_stats.csv')
        df.describe().to_csv(stats_file)
        print(f"\nSummary statistics saved to: {stats_file}")
        
        # 2. Create a smaller sample file (first 1000 rows)
        sample_file = output_file.replace('.csv', '_sample.csv')
        df.head(1000).to_csv(sample_file, index=False, float_format='%.6E')
        print(f"Sample file (first 1000 rows) saved to: {sample_file}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

# Example of how to use the converted data
def analyze_converted_data(csv_file):
    """
    Example function showing how to work with the converted CSV data
    """
    import matplotlib.pyplot as plt
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Basic analysis
    print("\nBasic Data Analysis:")
    print(f"Time range: {df['Time'].min():.2f} to {df['Time'].max():.2f} seconds")
    print(f"Sampling rate: {1/(df['Time'].iloc[1] - df['Time'].iloc[0]):.2f} Hz")
    
    # Example: Plot some key parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Platform surge
    axes[0, 0].plot(df['Time'], df['PtfmSurge'])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Platform Surge (m)')
    axes[0, 0].set_title('Platform Surge Motion')
    axes[0, 0].grid(True)
    
    # Platform pitch
    axes[0, 1].plot(df['Time'], df['PtfmPitch'])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Platform Pitch (deg)')
    axes[0, 1].set_title('Platform Pitch Motion')
    axes[0, 1].grid(True)
    
    # Generator power
    axes[1, 0].plot(df['Time'], df['GenPwr'])
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Generator Power (kW)')
    axes[1, 0].set_title('Generator Power Output')
    axes[1, 0].grid(True)
    
    # Wave elevation
    axes[1, 1].plot(df['Time'], df['Wave1Elev'])
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Wave Elevation (m)')
    axes[1, 1].set_title('Wave Elevation at Platform')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(csv_file.replace('.csv', '_plots.png'), dpi=300)
    print(f"\nPlots saved to: {csv_file.replace('.csv', '_plots.png')}")
    
    return df