# file_io_handler.py

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Set, List

# Import the necessary classes from the openfast-toolbox
# We use a try-except block to provide a graceful error if the library isn't installed.
try:
    from openfast_toolbox.io.fast_input_file import FASTInputFile
    from openfast_toolbox.io.fast_output_file import FASTOutFile
    # You can add more specific file types for better handling if needed
    # from openfast_toolbox.io.elastodyn_input_file import ElastoDynInputFile
    # from openfast_toolbox.io.hydrodyn_input_file import HydroDynInputFile
    TOOLBOX_AVAILABLE = True
except ImportError:
    TOOLBOX_AVAILABLE = False

def discover_model_files(initial_fst_path: Path, logger) -> Dict[str, Dict]:
    """
    Discovers all model files recursively using the openfast_io library.
    This is the robust replacement for _discover_and_parse_files_recursively.
    """
    if not TOOLBOX_AVAILABLE:
        raise ImportError("The 'openfast-toolbox' library is required. Please run 'pip install openfast-toolbox'.")

    file_structure = {}
    processed_paths = set()
    files_to_scan = [initial_fst_path]

    while files_to_scan:
        current_path = files_to_scan.pop(0)
        if not current_path or not current_path.exists() or current_path in processed_paths:
            continue

        logger.info(f"  Scanning: {current_path.name}")
        processed_paths.add(current_path)

        # Use a generic FASTInputFile reader, which works for most file types.
        try:
            fast_file = FASTInputFile(current_path)
            file_key = current_path.name
            
            # Handle potential duplicate filenames by making the key unique
            i = 1
            while file_key in file_structure:
                file_key = f"{current_path.stem}_{i}{current_path.suffix}"
                i += 1
            
            file_structure[file_key] = {'path': current_path}

            # Now, look for file paths within this file's dictionary representation
            for key, value in fast_file.to_dict().items():
                if isinstance(value, str) and value.lower() not in ['unused', 'default', 'none', 'true', 'false']:
                    # Check if the value string looks like a file path
                    if '.' in value or '/' in value or '\\' in value:
                        # Resolve the path relative to the current file
                        potential_path = (current_path.parent / value).resolve()
                        
                        if potential_path.is_file():
                            if potential_path not in processed_paths:
                                files_to_scan.append(potential_path)
                        else:
                            # Handle root names (e.g., "HydroData/marin_semi")
                            parent_dir = potential_path.parent
                            root_name = potential_path.name
                            if parent_dir.is_dir():
                                for item in parent_dir.iterdir():
                                    if item.is_file() and item.name.startswith(root_name + '.'):
                                        if item not in processed_paths:
                                            logger.info(f"    Found root name family member: {item.name}")
                                            files_to_scan.append(item)
        except Exception as e:
            logger.error(f"Could not process file {current_path.name} with openfast_io: {e}")

    return file_structure

def discover_parameters_from_files(file_structure: Dict[str, Dict], logger) -> Dict[str, Dict]:
    """
    Extracts all numerical and selectable parameters from a list of files.
    This is the robust replacement for extract_parameters_from_file.
    """
    if not TOOLBOX_AVAILABLE:
        raise ImportError("The 'openfast-toolbox' library is required.")

    discovered_parameters = {}
    for key, info in file_structure.items():
        try:
            fast_file = FASTInputFile(info['path'])
            params = {}
            # The .to_dict() method gives us all parameters and their values
            for param_name, value in fast_file.to_dict().items():
                # Filter out non-numeric/non-boolean/non-string types and known non-parameters
                if isinstance(value, (int, float, bool, str)):
                    if param_name.lower() in ['true', 'false', 'default', 'unused', 'none', 'end', 'echo'] or 'file' in param_name.lower():
                        continue
                    
                    param_type = 'unknown'
                    if isinstance(value, bool): param_type = 'bool'
                    elif isinstance(value, int): param_type = 'int'
                    elif isinstance(value, float): param_type = 'float'
                    elif isinstance(value, str): param_type = 'option' # Assume strings are selectable options

                    # The library doesn't provide descriptions/units, so we'll leave them blank for now
                    # This is a trade-off for robustness.
                    params[param_name] = {
                        'original_value': value,
                        'type': param_type,
                        'description': f"Parameter '{param_name}' from {key}",
                        'unit': ''
                    }
            if params:
                discovered_parameters[key] = params
        except Exception as e:
            logger.error(f"Could not discover parameters in {info['path'].name}: {e}")
            
    return discovered_parameters

def modify_fast_file(file_path: Path, parameter_name: str, new_value: Any):
    """
    Reads an OpenFAST input file, changes a parameter, and writes it back.
    This is the robust replacement for modify_parameter_in_file.
    """
    if not TOOLBOX_AVAILABLE:
        raise ImportError("The 'openfast-toolbox' library is required.")

    try:
        fast_file = FASTInputFile(file_path)
        # The library handles type conversion and formatting automatically
        fast_file[parameter_name] = new_value
        fast_file.write(file_path)
    except KeyError:
        raise KeyError(f"Parameter '{parameter_name}' not found in file '{file_path.name}' by the openfast_io library.")
    except Exception as e:
        raise IOError(f"Failed to write to '{file_path.name}' using openfast_io. Error: {e}")

def read_fast_out(output_file_path: str) -> pd.DataFrame:
    """
    Reads an OpenFAST .out or .outb file directly into a pandas DataFrame.
    This replaces the manual parser in the ConverterRunner class.
    """
    if not TOOLBOX_AVAILABLE:
        raise ImportError("The 'openfast-toolbox' library is required.")
        
    try:
        # The FASTOutFile class handles both text and binary output files
        fast_out = FASTOutFile(output_file_path)
        return fast_out.to_pandas()
    except Exception as e:
        raise IOError(f"Failed to read output file '{output_file_path}' using openfast_io. Error: {e}")