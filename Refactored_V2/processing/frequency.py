import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .plotting import MATPLOTLIB_AVAILABLE  # reuse available flag

try:  # pragma: no cover - optional dependency
    from scipy.signal import find_peaks

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SCIPY_AVAILABLE = False
    find_peaks = None  # type: ignore

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
else:  # pragma: no cover - optional dependency
    plt = None  # type: ignore


class FrequencyAnalysisRunner:
    """Encapsulates the logic for calculating natural frequencies from free decay tests."""

    _CANONICAL_COLUMN_ALIASES: Dict[str, str] = {
        "fairten1": "FAIRTEN1",
        "fair1ten": "FAIRTEN1",
        "fairten2": "FAIRTEN2",
        "fair2ten": "FAIRTEN2",
        "fairten3": "FAIRTEN3",
        "fair3ten": "FAIRTEN3",
        "anchten1": "ANCHTEN1",
        "anch1ten": "ANCHTEN1",
        "anchten2": "ANCHTEN2",
        "anch2ten": "ANCHTEN2",
        "anchten3": "ANCHTEN3",
        "anch3ten": "ANCHTEN3",
    }

    def __init__(self, message_queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def log(self, message: str) -> None:
        self.mq.put((self.log_type, f"[{self.case_name}][Freq] {message}"))

    def _canonicalize_column_names(self, df: pd.DataFrame, units_map: Dict[str, str]) -> None:
        rename_map: Dict[str, str] = {}
        for col in list(df.columns):
            key = str(col).strip().lower()
            canonical = self._CANONICAL_COLUMN_ALIASES.get(key)
            if not canonical or canonical == col:
                continue

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
        try:
            df_raw = pd.read_csv(csv_file, comment="#", skip_blank_lines=True, engine="python")
            if df_raw.empty:
                raise ValueError(f"CSV '{csv_file}' appears to be empty.")

            df_raw.columns = [str(col).strip() for col in df_raw.columns]
            first_row = df_raw.iloc[0]
            units_map: Dict[str, str] = {}

            def looks_like_units(value) -> bool:
                return isinstance(value, str) and any(char.isalpha() for char in value)

            if any(looks_like_units(val) for val in first_row):
                for col in df_raw.columns:
                    unit_token = str(first_row[col])
                    units_map[col] = unit_token.strip(" []()")
                df = df_raw.iloc[1:].reset_index(drop=True)
            else:
                df = df_raw.copy()
                units_map = {col: "" for col in df.columns}

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            self._canonicalize_column_names(df, units_map)
            return df, units_map
        except Exception as exc:
            self.log(f"Error reading CSV for frequency analysis: {exc}")
            return None, None

    def _calculate_frequencies_from_decay(self, time: np.ndarray, data: np.ndarray) -> Dict:
        if find_peaks is None:  # pragma: no cover - optional dependency
            raise RuntimeError("SciPy is required for frequency analysis but is not installed.")

        prominence_threshold = (np.max(data) - np.min(data)) * 0.1
        peak_indices, _ = find_peaks(data, prominence=prominence_threshold)

        if len(peak_indices) < 2:
            raise ValueError("Could not find at least 2 significant peaks. Check signal or start time.")

        peak_times = time[peak_indices]
        peak_values = data[peak_indices]
        damped_periods = np.diff(peak_times)
        mean_damped_period = float(np.mean(damped_periods))
        damped_frequency_hz = 1.0 / mean_damped_period
        damped_frequency_rad = 2.0 * np.pi * damped_frequency_hz

        log_decrements = []
        for i in range(len(peak_values) - 1):
            if peak_values[i] > 0 and peak_values[i + 1] > 0:
                log_decrements.append(np.log(peak_values[i] / peak_values[i + 1]))

        if not log_decrements:
            raise ValueError("Could not calculate logarithmic decrement. Are peak values valid?")

        mean_log_decrement = float(np.mean(log_decrements))
        damping_ratio = mean_log_decrement / np.sqrt((2 * np.pi) ** 2 + mean_log_decrement ** 2)

        if damping_ratio >= 1.0:
            natural_frequency_rad = float("nan")
            natural_period = float("nan")
        else:
            natural_frequency_rad = damped_frequency_rad / np.sqrt(1.0 - damping_ratio**2)
            natural_period = 2.0 * np.pi / natural_frequency_rad

        return {
            "damped_period_s": mean_damped_period,
            "damped_frequency_hz": damped_frequency_hz,
            "damped_frequency_rad_s": damped_frequency_rad,
            "logarithmic_decrement": mean_log_decrement,
            "damping_ratio_zeta": damping_ratio,
            "natural_period_s": natural_period,
            "natural_frequency_rad_s": natural_frequency_rad,
            "peak_indices": peak_indices.tolist(),
            "peak_times": peak_times.tolist(),
            "peak_values": peak_values.tolist(),
        }

    def _plot_decay_analysis(self, time, data, results, column_name, units, filename) -> None:
        fig, ax = plt.subplots(figsize=(12, 7))  # type: ignore
        try:
            ax.plot(time, data, label=f'"{column_name}" Signal (Mean Subtracted)', color="cornflowerblue", zorder=2)
            ax.plot(
                results["peak_times"],
                results["peak_values"],
                "o",
                color="crimson",
                markersize=8,
                label=f"Detected Peaks ({len(results['peak_times'])} found)",
                zorder=3,
            )

            A0 = results["peak_values"][0]
            zeta = results["damping_ratio_zeta"]
            wn = results["natural_frequency_rad_s"]

            envelope_time = np.linspace(results["peak_times"][0], time[-1], 500)
            decay_envelope = A0 * np.exp(-zeta * wn * (envelope_time - envelope_time[0]))
            ax.plot(envelope_time, decay_envelope, "--", color="black", label="Fitted Exponential Decay Envelope", zorder=4)
            ax.plot(envelope_time, -decay_envelope, "--", color="black", zorder=4)

            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel(f"Amplitude ({units})", fontsize=12)
            ax.set_title(f'Free Decay Analysis for "{column_name}"', fontsize=14, weight="bold")
            ax.legend(loc="upper right")

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
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)

            plt.tight_layout()  # type: ignore
            plt.savefig(filename, dpi=150)  # type: ignore
            self.log(f"Saved frequency analysis plot: {Path(filename).name}")
        finally:
            plt.close(fig)  # type: ignore

    def run(self, csv_file: str, column_name: str, output_dir: str, start_time: float) -> None:
        if not SCIPY_AVAILABLE:
            self.log("SciPy not found, skipping frequency analysis.")
            return
        if not MATPLOTLIB_AVAILABLE:
            self.log("Matplotlib not found, skipping frequency analysis plots.")
            return

        self.log(f"Starting frequency analysis for column '{column_name}'")
        df, units_map = self._read_fast_csv(csv_file)
        if df is None:
            return

        analysis_col = column_name
        if analysis_col not in df.columns:
            for alias, canonical in self._CANONICAL_COLUMN_ALIASES.items():
                if alias == column_name.strip().lower() and canonical in df.columns:
                    analysis_col = canonical
                    break
            else:
                raise KeyError(f"Column '{column_name}' not found. Available: {', '.join(df.columns)}")

        self.log(f"Analyzing canonical column: '{analysis_col}'")
        df_filtered = df[df["Time"] >= start_time].copy()
        if df_filtered.empty:
            raise ValueError(f"No data available after start_time={start_time}s.")

        time_series = df_filtered["Time"].to_numpy()
        data_series_raw = df_filtered[analysis_col].to_numpy()
        mean_value = float(np.mean(data_series_raw))
        data_series_zero_meaned = data_series_raw - mean_value
        self.log(f"Subtracted signal mean ({mean_value:.4f}) for peak analysis.")

        results = self._calculate_frequencies_from_decay(time_series, data_series_zero_meaned)

        results_path = Path(output_dir) / f"frequency_results_{analysis_col}.json"
        with open(results_path, "w") as f_out:
            json.dump(results, f_out, indent=4)
        self.log(f"Saved numerical results to: {results_path.name}")

        plot_path = Path(output_dir) / f"frequency_plot_{analysis_col}.png"
        self._plot_decay_analysis(
            time_series,
            data_series_zero_meaned,
            results,
            analysis_col,
            units_map.get(analysis_col, "-") if units_map else "-",
            filename=str(plot_path),
        )