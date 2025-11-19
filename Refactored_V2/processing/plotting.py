import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore
    blended_transform_factory = None  # type: ignore


class PlottingRunner:
    """Generates plots from a given CSV data file."""

    _VECTOR_BASES = ["TwrBsF", "TwrBsM", "HydroF", "HydroM"]
    _SCALAR_CHANNELS = [
        "PtfmRoll",
        "PtfmPitch",
        "PtfmYaw",
        "PtfmSurge",
        "PtfmSway",
        "PtfmHeave",
        "FairTen1",
        "FairTen2",
        "FairTen3",
    ]

    def __init__(self, message_queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def log(self, message: str) -> None:
        self.mq.put((self.log_type, f"[{self.case_name}][Plot] {message}"))

    @staticmethod
    def _simplify_header(name: str) -> str:
        return re.sub(r"\s+", "", re.sub(r"\s*\(.*?\)\s*$", "", str(name))).lower()

    @staticmethod
    def _strip_units(name: str) -> str:
        return re.sub(r"\s*\(.*?\)\s*$", "", str(name)).strip()

    @staticmethod
    def _extract_units_from_header(name: str) -> str:
        match = re.search(r"\((.*?)\)", str(name))
        return match.group(1) if match else ""

    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        for column in df.columns:
            if re.match(r"^\s*Time\b", str(column), re.IGNORECASE):
                return column
        return None

    def _build_units_map_from_csv(self, columns: List[str]) -> Dict[str, str]:
        return {col: self._extract_units_from_header(col) for col in columns}

    def _find_vector_columns(
        self, df: pd.DataFrame, base_name: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        simplified_map = {col: self._simplify_header(col) for col in df.columns}
        base_simple = self._simplify_header(base_name)
        cols = {"x": None, "y": None, "z": None}

        for component in ["x", "y", "z"]:
            pattern = re.compile(rf"^{re.escape(base_simple)}{component}[a-z0-9]*$", re.IGNORECASE)
            for original_col, simplified_col in simplified_map.items():
                if pattern.match(simplified_col):
                    cols[component] = original_col
                    break
            else:
                self.log(f"Component search failed for base '{base_name}', component '{component}'")

        return cols["x"], cols["y"], cols["z"]

    def _get_unit_with_fallback(self, column_name: str, units_map_csv: Dict[str, str]) -> str:
        unit = units_map_csv.get(column_name, "")
        if unit:
            return unit

        metadata_units = {
            "time": "s",
            "twrbsfxt": "kN",
            "twrbsmxt": "kN-m",
            "ptfmroll": "deg",
            "ptfmsurge": "m",
            "hydrofxi": "N",
            "hydromxi": "N-m",
            "fairten1x": "N",
        }

        key = self._simplify_header(column_name)
        for meta_key, meta_unit in metadata_units.items():
            if key.startswith(meta_key[:-1]) and key[-1] in "xyz":
                return meta_unit
            if key == meta_key:
                return meta_unit
        return ""

    def _compute_stats_after_threshold(
        self, t: pd.Series, y: pd.Series, threshold_time: float
    ) -> Tuple[float, float, float, float, float, bool]:
        t_numeric = pd.to_numeric(t, errors="coerce")
        y_numeric = pd.to_numeric(y, errors="coerce")

        mask = (t_numeric >= threshold_time) & y_numeric.notna()
        if mask.any():
            series = y_numeric[mask]
            t_series = t_numeric[mask]
            used_tail = True
        else:
            series = y_numeric[y_numeric.notna()]
            t_series = t_numeric[y_numeric.notna()]
            used_tail = False

        if series.empty:
            return (np.nan,) * 5 + (used_tail,)

        mean_val = series.mean()
        idx_min = series.idxmin()
        idx_max = series.idxmax()

        return (
            mean_val,
            series.loc[idx_min],
            series.loc[idx_max],
            float(t_series.loc[idx_min]),
            float(t_series.loc[idx_max]),
            used_tail,
        )

    def _draw_stats_for_series(
        self,
        ax,
        color: str,
        time_col: str,
        df: pd.DataFrame,
        y_col: str,
        mean_start: float,
        time_unit: str,
        y_unit: str,
        label_prefix: str,
        always_minmax: bool,
        minmax_range_frac: float,
        minmax_abs: float,
    ) -> None:
        if blended_transform_factory is None:  # pragma: no cover - optional dependency
            return

        def format_engineering(val: float) -> str:
            if pd.isna(val):
                return "N/A"
            if abs(val) >= 1e4 or (0 < abs(val) < 1e-2):
                return f"{val:.3e}"
            return f"{val:.4f}"

        def annotate_at_y_axis(ax_obj, y_value: float, text: str) -> None:
            trans = blended_transform_factory(ax_obj.transAxes, ax_obj.transData)
            ax_obj.annotate(
                text,
                xy=(0.0, y_value),
                xycoords=trans,
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray"),
            )

        (
            mean_val,
            min_val,
            max_val,
            t_min,
            t_max,
            used_tail,
        ) = self._compute_stats_after_threshold(df[time_col], df[y_col], mean_start)

        desc_base = f"≥ {mean_start:g}{' ' + time_unit if time_unit else ''}" if used_tail else "all data"
        mean_text = f"{format_engineering(mean_val)}{' ' + y_unit if y_unit else ''}"

        ax.axhline(
            y=mean_val,
            color=color,
            linestyle="--",
            linewidth=1.3,
            label=f"{label_prefix} -- Mean ({desc_base}): {mean_text}",
        )
        annotate_at_y_axis(ax, mean_val, mean_text)

        rng = float(max_val - min_val) if pd.notna(max_val) and pd.notna(min_val) else np.nan
        show_minmax = always_minmax or (
            pd.notna(rng) and rng >= max(minmax_range_frac * max(abs(mean_val), 1e-12), minmax_abs)
        )

        if show_minmax and pd.notna(min_val) and pd.notna(max_val):
            ax.axhline(
                y=min_val,
                color=color,
                linestyle=":",
                linewidth=1.2,
                label=f"{label_prefix} -- Min at t={t_min:.2f}: {format_engineering(min_val)}",
            )
            annotate_at_y_axis(ax, min_val, format_engineering(min_val))

            ax.axhline(
                y=max_val,
                color=color,
                linestyle="-.",
                linewidth=1.2,
                label=f"{label_prefix} -- Max at t={t_max:.2f}: {format_engineering(max_val)}",
            )
            annotate_at_y_axis(ax, max_val, format_engineering(max_val))

    def _plot_group(
        self,
        time_col: str,
        df: pd.DataFrame,
        series_cols: List[str],
        series_labels: Dict[str, str],
        series_units: Dict[str, str],
        group_title: str,
        x_label: str,
        y_unit_hint: Optional[str],
        mean_start: float,
        time_unit: str,
        case_suffix: str,
        output_dir: str,
        file_stub: str,
        **kwargs,
    ) -> None:
        units = [series_units.get(c, "") for c in series_cols]
        chosen_unit = y_unit_hint or next((u for u in units if u), "")
        if any((u and chosen_unit and u != chosen_unit) for u in units):
            self.log(f"Warning: Mixed units in group '{group_title}'. Using '{chosen_unit}'.")

        fig, ax = plt.subplots(figsize=(12, 6))  # type: ignore
        for col in series_cols:
            if col not in df.columns:
                self.log(f"Warning: Column '{col}' missing for group '{group_title}'.")
                continue

            label = series_labels.get(col, self._strip_units(col))
            (line_handle,) = ax.plot(df[time_col], df[col], label=label, linewidth=1.4)

            self._draw_stats_for_series(
                ax,
                line_handle.get_color(),
                time_col,
                df,
                col,
                mean_start,
                time_unit,
                series_units.get(col, ""),
                label,
                **kwargs,
            )

        ax.set_title(f"{group_title} vs. {self._strip_units(time_col)}{case_suffix}", fontsize=16)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f"{group_title}{f' [{chosen_unit}]' if chosen_unit else ''}", fontsize=12)
        ax.grid(True)
        legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, framealpha=0.9)
        plt.tight_layout()  # type: ignore

        save_path = os.path.join(
            output_dir,
            re.sub(r'[\\/*?:"<>|()\s]', "", str(file_stub)) + ".png",
        )
        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", bbox_extra_artists=(legend,))  # type: ignore
            self.log(f"Saved group plot: '{Path(save_path).name}'")
        except Exception as exc:
            self.log(f"Error saving group plot '{group_title}': {exc}")
        finally:
            plt.close(fig)  # type: ignore

    def run(
        self,
        csv_file: str,
        output_dir: str,
        case_name: Optional[str] = None,
        mean_start: float = 300.0,
        **kwargs,
    ) -> None:
        if not MATPLOTLIB_AVAILABLE:
            self.log("Matplotlib not found, skipping plotting.")
            return

        self.log(f"Reading data from '{Path(csv_file).name}'...")
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
        except Exception as exc:
            self.log(f"Error reading CSV file: {exc}")
            return

        time_col = self._find_time_column(df)
        if not time_col:
            self.log("Error: No 'Time' column found.")
            return

        csv_units_map = self._build_units_map_from_csv(list(df.columns))
        series_label: Dict[str, str] = {}
        series_unit: Dict[str, str] = {}

        time_unit = self._get_unit_with_fallback(time_col, csv_units_map)
        x_label = f"{self._strip_units(time_col)}{f' [{time_unit}]' if time_unit else ''}"

        channels_to_plot: List[str] = []
        found_scalars: Dict[str, str] = {}

        for base in self._VECTOR_BASES:
            self.log(f"Searching for vector components for base: '{base}'")
            x_col, y_col, z_col = self._find_vector_columns(df, base)
            self.log(f"  -> Found components: X='{x_col}', Y='{y_col}', Z='{z_col}'")

            if all([x_col, y_col, z_col]):
                try:
                    mag_vals = np.sqrt(
                        pd.to_numeric(df[x_col], errors="coerce") ** 2
                        + pd.to_numeric(df[y_col], errors="coerce") ** 2
                        + pd.to_numeric(df[z_col], errors="coerce") ** 2
                    )
                    mag_col = f"{base}_Magnitude"
                    df[mag_col] = mag_vals
                    channels_to_plot.append(mag_col)
                    series_label[mag_col] = mag_col
                    series_unit[mag_col] = self._get_unit_with_fallback(x_col, csv_units_map)
                except Exception as exc:
                    self.log(f"Warning: Failed to compute magnitude for '{base}': {exc}")
            else:
                self.log(f"  -> FAILED to find all three components for '{base}'. Skipping magnitude calculation.")

        for channel in self._SCALAR_CHANNELS:
            matches = [c for c in df.columns if str(c).strip().lower().startswith(channel.lower())]
            if matches:
                col = matches[0]
                channels_to_plot.append(col)
                series_label[col] = self._strip_units(col)
                series_unit[col] = self._get_unit_with_fallback(col, csv_units_map)
                found_scalars[channel] = col

        if not channels_to_plot:
            self.log("No channels were found to plot.")
            return

        os.makedirs(output_dir, exist_ok=True)
        self.log("Generating plots...")
        plt.style.use("ggplot")  # type: ignore
        case_suffix = f" -- {case_name}" if case_name else ""

        # Individual plots
        for channel in channels_to_plot:
            fig, ax = plt.subplots(figsize=(12, 6))  # type: ignore
            try:
                label = series_label.get(channel, self._strip_units(channel))
                (line_handle,) = ax.plot(df[time_col], df[channel], label=label)  # type: ignore
                y_unit = series_unit.get(channel, "")
                ax.set_title(f"{label} vs. {self._strip_units(time_col)}{case_suffix}", fontsize=16)
                ax.set_xlabel(x_label, fontsize=12)
                ax.set_ylabel(f"{label}{f' [{y_unit}]' if y_unit else ''}", fontsize=12)

                self._draw_stats_for_series(
                    ax,
                    line_handle.get_color(),
                    time_col,
                    df,
                    channel,
                    mean_start,
                    time_unit,
                    y_unit,
                    label,
                    **kwargs,
                )
                ax.legend()
                ax.grid(True)
                plt.tight_layout()  # type: ignore

                save_path = os.path.join(
                    output_dir,
                    re.sub(r'[\\/*?:"<>|()\s]', "", label) + ".png",
                )
                plt.savefig(save_path, dpi=150)  # type: ignore
            except Exception as exc:
                self.log(f"Error plotting channel '{channel}': {exc}")
            finally:
                plt.close(fig)  # type: ignore

        # Group plots
        rpy_cols = [
            c for c in [found_scalars.get(key) for key in ["PtfmRoll", "PtfmPitch", "PtfmYaw"]] if c
        ]
        if rpy_cols:
            self._plot_group(
                time_col,
                df,
                rpy_cols,
                series_label,
                series_unit,
                "Platform Roll/Pitch/Yaw",
                x_label,
                "deg",
                mean_start,
                time_unit,
                case_suffix,
                output_dir,
                "Ptfm_RollPitchYaw",
                **kwargs,
            )

        ssh_cols = [
            c for c in [found_scalars.get(key) for key in ["PtfmSurge", "PtfmSway", "PtfmHeave"]] if c
        ]
        if ssh_cols:
            self._plot_group(
                time_col,
                df,
                ssh_cols,
                series_label,
                series_unit,
                "Platform Surge/Sway/Heave",
                x_label,
                "m",
                mean_start,
                time_unit,
                case_suffix,
                output_dir,
                "Ptfm_SurgeSwayHeave",
                **kwargs,
            )

        fair_cols = [
            c for c in [found_scalars.get(key) for key in ["FairTen1", "FairTen2", "FairTen3"]] if c
        ]
        if fair_cols:
            self._plot_group(
                time_col,
                df,
                fair_cols,
                series_label,
                series_unit,
                "Fairlead Tensions",
                x_label,
                "N",
                mean_start,
                time_unit,
                case_suffix,
                output_dir,
                "Fairlead_Tensions",
                **kwargs,
            )

        self.log("Plotting complete.")