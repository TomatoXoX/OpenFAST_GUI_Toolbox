import traceback
from pathlib import Path
from typing import Optional

import numpy as np


class ConverterRunner:
    """Handles the conversion of OpenFAST .out files to .csv format."""

    def __init__(self, message_queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def log(self, message: str) -> None:
        """Logs a message to the GUI via the message queue."""
        self.mq.put((self.log_type, f"[{self.case_name}][CSV] {message}"))

    def convert_openfast_to_csv_robust(self, input_file: str, output_file: str) -> bool:
        """
        Converts an OpenFAST output file to CSV using a memory-efficient streaming approach.
        """
        self.log(f"Attempting to convert '{Path(input_file).name}' using streaming...")

        try:
            with open(input_file, "r", encoding="utf-8", errors="ignore") as f_in:
                header_lines: List[str] = []
                column_names: List[str] = []
                column_units: List[str] = []
                original_column_names: List[str] = []
                data_start_line_num: int = -1

                # --- Locate header/units lines near the top (no tell/seek) ---
                for line_index in range(200):
                    line = f_in.readline()
                    if not line:
                        break  # EOF before finding header (will be handled below)
                    header_lines.append(line)
                    tokens = line.split()

                    if "Time" in tokens:
                        potential_names = line.strip().split()

                        units_line = f_in.readline()
                        if units_line:
                            header_lines.append(units_line)
                            potential_units = units_line.strip().split()
                        else:
                            potential_units = []

                        if (
                            potential_units
                            and len(potential_names) == len(potential_units)
                            and len(potential_names) > 1
                        ):
                            column_names = potential_names
                            original_column_names = potential_names.copy()
                            column_units = potential_units
                            data_start_line_num = len(header_lines)
                            break
                        else:
                            # Units line invalid—remove it from header_lines and continue searching
                            if units_line:
                                header_lines.pop()

                if not column_names:
                    self.log("Error: Could not find the header and unit lines. Check .out file format.")
                    return False

                # Handle duplicate column names
                seen: Dict[str, int] = {}
                unique_columns: List[str] = []
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
                with open(output_file, "w", newline="") as f_out:
                    f_out.write(",".join(column_names) + "\n")

                    for line_num, line in enumerate(f_in, start=data_start_line_num + 1):
                        line = line.strip()
                        if not line:
                            continue

                        values = line.split()
                        if len(values) == len(column_names):
                            try:
                                formatted_values = [
                                    f"{float(val.replace('D', 'E')):.6E}" for val in values
                                ]
                                f_out.write(",".join(formatted_values) + "\n")
                                row_count += 1
                            except ValueError:
                                self.log(
                                    f"Warning: Could not parse data on line {line_num}. Skipping."
                                )
                        else:
                            self.log(
                                f"Warning: Mismatch in column count on line {line_num}. "
                                f"Expected {len(column_names)}, found {len(values)}. Skipping."
                            )

                if row_count == 0:
                    self.log("Error: No data was successfully parsed from the file.")
                    try:
                        Path(output_file).unlink()
                    except OSError:
                        pass
                    return False

        except FileNotFoundError:
            self.log(f"Error: The input file was not found at '{input_file}'")
            return False
        except Exception as exc:
            self.log(f"Error during streaming conversion: {exc}\n{traceback.format_exc()}")
            return False

        # --- Write Metadata ---
        metadata_file = output_file.rsplit(".", 1)[0] + "_metadata.txt"
        with open(metadata_file, "w") as f_out:
            f_out.write("OpenFAST Output File Metadata\n" + "=" * 60 + "\n\n")
            f_out.write(f"Source File: {Path(input_file).name}\n\n")

            desc_lines = [line for line in header_lines if "Description:" in line]
            if desc_lines:
                f_out.writelines(desc_lines)
            else:
                f_out.write("No 'Description:' line found in the original file header.\n")

            f_out.write("\nColumn Information:\n" + "-" * 60 + "\n")
            f_out.write(f"{'Column Name':<25} {'Units'}\n" + "-" * 60 + "\n")
            for name, unit in zip(original_column_names, column_units):
                f_out.write(f"{name:<25} {unit}\n")

        self.log("--- Conversion Summary ---")
        self.log(f"{'Input file:':<20} {Path(input_file).name}")
        self.log(f"{'Output CSV:':<20} {Path(output_file).name}")
        self.log(f"{'Rows/Cols:':<20} {row_count} / {len(column_names)}")

        return True