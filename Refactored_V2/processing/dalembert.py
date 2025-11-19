import gc
import json
import logging
import math
import os
import re
import traceback
from datetime import datetime
from math import cos, sin, radians
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.file_utils import _find_fst_refs, _read_lines, _strip_quotes


class DalembertLogHandler(logging.Handler):
    """Custom logging handler to route logs to the GUI's message queue."""

    def __init__(self, message_queue, case_name, log_type):
        super().__init__()
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - GUI integration
        self.mq.put((self.log_type, f"[{self.case_name}][Dalembert] {self.format(record)}"))


class DalembertRunner:
    """
    Performs d'Alembert staticization to extract quasi-static loads from dynamic simulation results.
    """

    def __init__(self, message_queue, case_name: str, log_type: str):
        self.mq = message_queue
        self.case_name = case_name
        self.log_type = log_type
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"dalembert_{self.case_name}_{id(self)}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            handler = DalembertLogHandler(self.mq, self.case_name, self.log_type)
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def log(self, message: str) -> None:
        self.logger.info(message)

    def run(
        self,
        fst: str,
        glue_out: str,
        outdir: str,
        analysis_start_time: float,
        **kwargs,
    ) -> None:
        self.logger.info("========== d'Alembert staticization: START ==========")
        try:
            class Args:
                def __init__(self, data: Dict[str, Any]) -> None:
                    self.__dict__.update(data)

            args = Args(
                {
                    "fst": fst,
                    "glue_out": glue_out,
                    "outdir": outdir,
                    "outb": kwargs.get("outb", False),
                    "moordyn_out": kwargs.get("moordyn_out"),
                    "rotate_ed": kwargs.get("rotate_ed", True),
                    "override_mass": kwargs.get("override_mass"),
                    "override_com": kwargs.get("override_com"),
                    "override_inertia": kwargs.get("override_inertia"),
                    "verbose": True,
                    "log_step": kwargs.get("log_step", 100),
                }
            )
            self.logger.info("Arguments: " + json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
            os.makedirs(args.outdir, exist_ok=True)

            builder = self.MassPropertyBuilder(args.fst, self.logger)
            auto_mass, auto_com, auto_Icom = builder.compute()

            mass = args.override_mass if args.override_mass is not None else auto_mass
            r_com = np.array(args.override_com, float) if args.override_com is not None else auto_com
            inertia_tensor = (
                np.array(
                    [
                        [args.override_inertia[0], args.override_inertia[3], args.override_inertia[4]],
                        [args.override_inertia[3], args.override_inertia[1], args.override_inertia[5]],
                        [args.override_inertia[4], args.override_inertia[5], args.override_inertia[2]],
                    ],
                    float,
                )
                if args.override_inertia is not None
                else auto_Icom
            )

            self.logger.info(f"Mass properties in use: m={mass:.6e} kg, CoM={r_com.tolist()}")

            refs = _find_fst_refs(args.fst, self.logger)
            ed_path = refs["EDFile"]
            md_path = refs["MooringFile"]

            geo = self._parse_elastodyn_geometry(ed_path)
            fairleads, anchors = self._parse_moordyn_points(md_path)
            PRP = np.zeros(3)
            yaw_xyz = geo["YawBearing"]
            twrbase_xyz = geo["TowerBase"]

            df = self._parse_glue_text(args.glue_out)
            df = self._collapse_duplicates(df)

            df_md = self._parse_glue_text(args.moordyn_out) if args.moordyn_out else None
            if df_md is not None and "time" in df_md.columns:
                df_md = df_md.set_index("time")

            self._perform_dalembert_calculations(
                df=df,
                df_md=df_md,
                args=args,
                mass=mass,
                r_com=r_com,
                inertia_tensor=inertia_tensor,
                PRP=PRP,
                yaw_xyz=yaw_xyz,
                twrbase_xyz=twrbase_xyz,
                fairleads=fairleads,
                anchors=anchors,
                analysis_start_time=analysis_start_time,
                geo=geo,
            )

        except Exception as exc:  # pragma: no cover - logging driven
            self.logger.error(f"FATAL ERROR in d'Alembert analysis: {exc}\n{traceback.format_exc()}")
        finally:
            self.logger.info("========== d'Alembert staticization: END ==========")

    def _perform_dalembert_calculations(
        self,
        df: pd.DataFrame,
        df_md: Optional[pd.DataFrame],
        args: Any,
        mass: float,
        r_com: np.ndarray,
        inertia_tensor: np.ndarray,
        PRP: np.ndarray,
        yaw_xyz: np.ndarray,
        twrbase_xyz: np.ndarray,
        fairleads: Dict[int, np.ndarray],
        anchors: Dict[int, np.ndarray],
        analysis_start_time: float,
        geo: Dict[str, Any],
    ) -> None:
        hydro_cols = ["hydrofxi", "hydrofyi", "hydrofzi", "hydromxi", "hydromyi", "hydromzi"]

        if all(col in df.columns for col in ["twrbsfxt", "twrbsfyt", "twrbsfzt", "twrbsmxt", "twrbsmyt", "twrbsmzt"]):
            edF_cols = ["twrbsfxt", "twrbsfyt", "twrbsfzt"]
            edM_cols = ["twrbsmxt", "twrbsmyt", "twrbsmzt"]
            ed_point = twrbase_xyz
            ed_name = "ed_towerbase_interface"
            self.logger.info("Using ED interface at Tower Base (TwrBs*) in platform axes")
        elif all(col in df.columns for col in ["yawbrfxp", "yawbrfyp", "yawbrfzp", "yawbrmxp", "yawbrmyp", "yawbrmzp"]):
            edF_cols = ["yawbrfxp", "yawbrfyp", "yawbrfzp"]
            edM_cols = ["yawbrmxp", "yawbrmyp", "yawbrmzp"]
            ed_point = yaw_xyz
            ed_name = "ed_yawbr_interface"
            self.logger.info("Using ED interface at Yaw Bearing (YawBr*) in platform axes")
        else:
            raise RuntimeError("Missing ED interface loads (TwrBs* or YawBr*).")

        rows: List[Dict[str, float]] = []
        total_rows = len(df)
        self.logger.info(f"Beginning time loop over {total_rows} rows")

        force_methods = self._detect_mooring_force_methods(df, df_md, fairleads)

        for idx, row in df.iterrows():
            time_value = row["time"]

            hydro_force = row[hydro_cols[:3]].values
            hydro_moment = row[hydro_cols[3:]].values

            ed_force_local = row[edF_cols].values
            ed_moment_local = row[edM_cols].values

            R_plat = self._rotmat_from_rpy_deg(row["ptfmroll"], row["ptfmpitch"], row["ptfmyaw"])
            if args.rotate_ed:
                ed_force = R_plat @ ed_force_local
                ed_moment = R_plat @ ed_moment_local
            else:
                ed_force = ed_force_local
                ed_moment = ed_moment_local

            ed_moment_at_prp = ed_moment + np.cross((PRP - ed_point), ed_force)

            moor_force, moor_moment, fair_entries = self._calculate_mooring_loads(
                row=row,
                df_md=df_md,
                R_plat=R_plat,
                fairleads=fairleads,
                anchors=anchors,
                force_methods=force_methods,
                timestep_index=idx,
            )

            external_force = hydro_force + moor_force + ed_force
            external_moment_prp = hydro_moment + moor_moment + ed_moment_at_prp
            inertia_force = -external_force
            inertia_moment = -external_moment_prp + np.cross((r_com - PRP), external_force)

            def add_entry(name: str, F: np.ndarray, P: np.ndarray, Mv: Optional[np.ndarray] = None) -> None:
                rows.append(
                    {
                        "Time": time_value,
                        "LoadName": name,
                        "Px": P[0],
                        "Py": P[1],
                        "Pz": P[2],
                        "Fx": F[0],
                        "Fy": F[1],
                        "Fz": F[2],
                        "Mx": (Mv[0] if Mv is not None else 0.0),
                        "My": (Mv[1] if Mv is not None else 0.0),
                        "Mz": (Mv[2] if Mv is not None else 0.0),
                        "F_norm": float(np.linalg.norm(F)),
                        "M_norm": float(np.linalg.norm(Mv)) if Mv is not None else 0.0,
                    }
                )

            add_entry("HydroDyn_Total_at_PRP", hydro_force, PRP, hydro_moment)
            add_entry(ed_name, ed_force, ed_point, ed_moment)

            for line_id, rk_local, F_line, method in fair_entries:
                method_label = {
                    "moordyn_main": "MoorDyn_HiFi",
                    "moordyn_file": "MoorDyn_HiFi_File",
                    "geometric": "MoorDyn_Approx",
                }[method]
                add_entry(f"{method_label}_Fairlead{line_id}", F_line, rk_local, None)

            if idx % 100 == 0:
                if not hasattr(self, "moor_force_samples"):
                    self.moor_force_samples = {line_id: [] for line_id in sorted(fairleads.keys())}
                for line_id, _, F_line, method in fair_entries:
                    self.moor_force_samples[line_id].append(
                        {
                            "time": time_value,
                            "Fx": F_line[0],
                            "Fy": F_line[1],
                            "Fz": F_line[2],
                            "magnitude": np.linalg.norm(F_line),
                            "method": method,
                        }
                    )

            add_entry("Inertia_Trans_CoM", inertia_force, r_com, None)
            add_entry("Inertia_Rot_CoM", np.zeros(3), r_com, inertia_moment)

            total_force = external_force + inertia_force
            total_moment = external_moment_prp + inertia_moment + np.cross((r_com - PRP), inertia_force)
            add_entry("TOTAL_with_Inertia_at_PRP", total_force, PRP, total_moment)

            if idx > 0 and idx % (getattr(args, "log_step", 100) * 10) == 0:
                self.logger.debug(f"Processing: {idx / total_rows:.1%} complete (t={time_value:.2f}s)")

        loads_df = pd.DataFrame(rows)
        loads_csv = os.path.join(args.outdir, "loads_timeseries_staticized.csv")
        loads_df.to_csv(loads_csv, index=False)
        self.logger.info(f"Wrote timeseries loads: {Path(loads_csv).name}")
        self._write_reports(
            loads_df=loads_df,
            args=args,
            geo=geo,
            fairleads=fairleads,
            mass=mass,
            r_com=r_com,
            inertia_tensor=inertia_tensor,
            analysis_start_time=analysis_start_time,
            force_methods=force_methods,
        )

    def _detect_mooring_force_methods(
        self,
        df: pd.DataFrame,
        df_md: Optional[pd.DataFrame],
        fairleads: Dict[int, np.ndarray],
    ) -> Dict[int, str]:
        force_methods: Dict[int, str] = {}
        self.logger.info("Mooring force calculation method detection:")

        for line_id in sorted(fairleads.keys()):
            connection_point = line_id + 3

            def _has_column(source_df: pd.DataFrame, prefix: str, comp: str) -> bool:
                target = f"{prefix}{comp}"
                return any(col.lower() == target.lower() for col in source_df.columns)

            if any(
                _has_column(df, f"Con{connection_point}F", comp) for comp in ["x", "y", "z"]
            ) or any(_has_column(df, f"Line{line_id}F", comp) for comp in ["x", "y", "z"]):
                force_methods[line_id] = "moordyn_main"
                self.logger.info(
                    f"  Line {line_id}: Using MoorDyn force components from main output (HIGH FIDELITY)"
                )
            elif df_md is not None and any(
                _has_column(df_md, f"Con{connection_point}F", comp) for comp in ["x", "y", "z"]
            ):
                force_methods[line_id] = "moordyn_file"
                self.logger.info(
                    f"  Line {line_id}: Using MoorDyn force components from separate file (HIGH FIDELITY)"
                )
            else:
                force_methods[line_id] = "geometric"
                self.logger.warning(
                    f"  Line {line_id}: Using geometric approximation (REDUCED ACCURACY)"
                )

        if any(method == "geometric" for method in force_methods.values()):
            self.logger.warning("╔════════════════════════════════════════════════════════════╗")
            self.logger.warning("║ NOTICE: Using straight-line approximation for some lines.  ║")
            self.logger.warning("║ For high fidelity, add Con<N>Fx/Fy/Fz to MoorDyn OUTPUTS.  ║")
            self.logger.warning("╚════════════════════════════════════════════════════════════╝")
        return force_methods

    def _calculate_mooring_loads(
        self,
        row: pd.Series,
        df_md: Optional[pd.DataFrame],
        R_plat: np.ndarray,
        fairleads: Dict[int, np.ndarray],
        anchors: Dict[int, np.ndarray],
        force_methods: Dict[int, str],
        timestep_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray, np.ndarray, str]]]:
        moor_force = np.zeros(3)
        moor_moment = np.zeros(3)
        fair_entries: List[Tuple[int, np.ndarray, np.ndarray, str]] = []
        platform_pos = row[["ptfmsurge", "ptfmsway", "ptfmheave"]].values

        for line_id, rk_local in sorted(fairleads.items()):
            F_line = np.zeros(3)
            method_used = force_methods[line_id]
            rk_global = platform_pos + R_plat @ rk_local
            connection_point = line_id + 3

            def _fetch_force_components(series: pd.Series, prefix: str) -> Optional[np.ndarray]:
                fx_col = next(
                    (c for c in series.index if c.lower() == f"{prefix}fx"), None
                )
                fy_col = next(
                    (c for c in series.index if c.lower() == f"{prefix}fy"), None
                )
                fz_col = next(
                    (c for c in series.index if c.lower() == f"{prefix}fz"), None
                )
                if fx_col and fy_col and fz_col:
                    return series[[fx_col, fy_col, fz_col]].values.astype(float)
                return None

            if method_used == "moordyn_main":
                F_line_local = _fetch_force_components(row, f"con{connection_point}")
                if F_line_local is None:
                    F_line_local = _fetch_force_components(row, f"line{line_id}")
                if F_line_local is not None:
                    F_line = F_line_local
                else:
                    method_used = "geometric"  # Fallback

            if method_used == "moordyn_file" and df_md is not None:
                idx_closest = (df_md.index - row["time"]).abs().argmin()
                F_line_local = _fetch_force_components(df_md.iloc[idx_closest], f"con{connection_point}")
                if F_line_local is not None:
                    F_line = F_line_local
                else:
                    method_used = "geometric"

            if method_used == "geometric":
                tension_col = next((c for c in row.index if c.lower() == f"fairten{line_id}"), None)
                if tension_col and anchors.get(line_id) is not None:
                    tension_mag = row[tension_col]
                    direction_vec = anchors[line_id] - rk_global
                    norm = np.linalg.norm(direction_vec)
                    if norm > 1e-6:
                        F_line = tension_mag * (direction_vec / norm)

            tension_col = next((c for c in row.index if c.lower() == f"fairten{line_id}"), None)
            if tension_col and not np.allclose(F_line, 0.0):
                reported_tension = row[tension_col]
                computed_magnitude = np.linalg.norm(F_line)
                relative_error = abs(computed_magnitude - reported_tension) / max(reported_tension, 1e-6)
                if relative_error > 0.10 and timestep_index % 500 == 0:
                    self.logger.warning(
                        f"t={row['time']:.2f}, Line {line_id}: Force magnitude mismatch! "
                        f"Computed={computed_magnitude:.2e}, Reported={reported_tension:.2e}, Err={relative_error:.1%}"
                    )

            moor_moment += np.cross(rk_global, F_line)
            moor_force += F_line
            fair_entries.append((line_id, rk_local, F_line, method_used))

        return moor_force, moor_moment, fair_entries

    def _write_reports(
        self,
        loads_df: pd.DataFrame,
        args: Any,
        geo: Dict[str, Any],
        fairleads: Dict[int, np.ndarray],
        mass: float,
        r_com: np.ndarray,
        inertia_tensor: np.ndarray,
        analysis_start_time: float,
        force_methods: Dict[int, str],
    ) -> None:
        extrema_lines: List[str] = []
        total = loads_df[
            (loads_df["LoadName"] == "TOTAL_with_Inertia_at_PRP") & (loads_df["Time"] >= analysis_start_time)
        ].copy()

        if total.empty:
            extrema_lines.append(f"No TOTAL_with_Inertia_at_PRP samples after {analysis_start_time:.2f}s.")
        else:
            force_mag = np.sqrt(total["Fx"] ** 2 + total["Fy"] ** 2 + total["Fz"] ** 2)
            moment_mag = np.sqrt(total["Mx"] ** 2 + total["My"] ** 2 + total["Mz"] ** 2)

            cases = {"F_max": force_mag.idxmax(), "M_max": moment_mag.idxmax()}
            extrema_data = [
                {"Case": name, **total.loc[idx][["Time", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]]}
                for name, idx in cases.items()
            ]
            extrema_df = pd.DataFrame(extrema_data)
            extrema_csv = os.path.join(args.outdir, f"loads_extrema_after{int(analysis_start_time)}s.csv")
            extrema_df.to_csv(extrema_csv, index=False)
            self.logger.info(f"Wrote extrema CSV: {Path(extrema_csv).name}")
            extrema_lines = extrema_df.to_string(index=False).split("\n")

        mooring_stats_lines = self._generate_mooring_stats_report(fairleads)

        report_lines = [
            "=" * 80,
            f"d'ALEMBERT STATICIZATION REPORT: {self.case_name}",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SIMULATION PARAMETERS:",
            "-" * 80,
        ]
        try:
            fst_content = Path(args.fst).read_text(encoding="utf-8", errors="ignore")
            tmax_match = re.search(r"^\s*([^\s]+)\s+TMax\b", fst_content, re.MULTILINE | re.IGNORECASE)
            tmax = float(tmax_match.group(1)) if tmax_match else math.nan
            report_lines.append(f"  FST File: {Path(args.fst).name}")
            report_lines.append(f"  Simulation Duration (TMax) : {tmax:.2f} s")
            report_lines.append(f"  Analysis Start Time        : {analysis_start_time:.2f} s")
            if not total.empty:
                report_lines.append(
                    f"  Analysis Duration          : {total['Time'].max() - analysis_start_time:.2f} s"
                )
            else:
                report_lines.append("  Analysis Duration          : N/A")
        except Exception as exc:
            self.logger.warning(f"Could not extract simulation parameters: {exc}")

        report_lines.extend(
            [
                "",
                "MASS PROPERTIES:",
                "-" * 80,
                f"  Total Mass: {mass:.6e} kg",
                f"  Center of Mass (CoM): {r_com.tolist()}",
                "",
            ]
        )
        report_lines.extend(mooring_stats_lines)
        report_lines.extend(
            [
                "",
                f"LOAD EXTREMA SUMMARY (t >= {analysis_start_time:.2f} s):",
                "=" * 80,
                *extrema_lines,
                "",
            ]
        )

        report_path = os.path.join(args.outdir, "staticized_report.txt")
        with open(report_path, "w") as f_out:
            f_out.write("\n".join(report_lines))
        self.logger.info(f"Wrote comprehensive report: {Path(report_path).name}")

    def _generate_mooring_stats_report(self, fairleads: Dict[int, np.ndarray]) -> List[str]:
        lines = ["\nMooring Force Statistics:", "=" * 80]
        if not hasattr(self, "moor_force_samples") or not self.moor_force_samples:
            lines.append("  No mooring force samples collected.")
            return lines

        for line_id in sorted(fairleads.keys()):
            samples = self.moor_force_samples.get(line_id, [])
            if not samples:
                continue
            df_samples = pd.DataFrame(samples)
            mag_stats = df_samples["magnitude"].describe()
            method = df_samples["method"].iloc[0]

            lines.append(f"\nLine {line_id} - Method: {method}:")
            lines.append(
                "  Force Mag [N]: "
                f"mean={mag_stats['mean']:.3e}, std={mag_stats['std']:.3e}, "
                f"min={mag_stats['min']:.3e}, max={mag_stats['max']:.3e}"
            )
            lines.append(
                "  Mean F [N]:    "
                f"Fx={df_samples['Fx'].mean():.3e}, "
                f"Fy={df_samples['Fy'].mean():.3e}, "
                f"Fz={df_samples['Fz'].mean():.3e}"
            )
        return lines

    def _parse_glue_text(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        if path is None:  # pragma: no cover - guard
            return None
        self.logger.info(f"Parsing glue (text) output with streaming: {path}")

        header_line_num = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f_in:
                for i, line in enumerate(f_in):
                    if line.strip().startswith("Time"):
                        header_line_num = i
                        break
        except Exception as exc:
            raise RuntimeError(f"Could not read or find header in {path}: {exc}")

        if header_line_num is None:
            raise RuntimeError(f"Header 'Time' not found in {path}")

        try:
            df = pd.read_csv(path, sep=r"\s+", header=header_line_num, encoding="utf-8", low_memory=True)
            if not pd.api.types.is_numeric_dtype(df.iloc[0, 0]):
                df = df.iloc[1:].reset_index(drop=True)

            df = df.apply(pd.to_numeric, errors="coerce")
            df.columns = [c.lower() for c in df.columns]
            self.logger.debug(f"Glue columns: {list(df.columns)}; rows={len(df)}")
            return df
        except Exception as exc:
            self.logger.error(f"Pandas failed to parse {path}. Error: {exc}\n{traceback.format_exc()}")
            raise RuntimeError(f"Pandas parsing error in {path}")

    def _collapse_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df.columns) == len(set(df.columns)):
            return df
        collapsed = {}
        for col in dict.fromkeys(df.columns):
            same = df.loc[:, df.columns == col]
            if same.shape[1] > 1:
                self.logger.debug(f"Collapsing duplicate column '{col}' by averaging {same.shape[1]} copies.")
                collapsed[col] = same.apply(pd.to_numeric, errors="coerce").mean(axis=1)
            else:
                collapsed[col] = pd.to_numeric(same.iloc[:, 0], errors="coerce")
        return pd.DataFrame(collapsed)

    def _parse_elastodyn_geometry(self, ed_path: str) -> Dict[str, Any]:
        lines = _read_lines(ed_path, self.logger)

        def fetch_value(key: str) -> Optional[float]:
            for line in lines:
                if key in line and not line.strip().startswith(("!", "#")):
                    return float(line.strip().split()[0].strip('"\''))
            return None

        tower_ht = fetch_value("TowerHt") or 90.0
        tower_bs_ht = fetch_value("TowerBsHt") or 0.0

        geo_data = {
            "TowerHt": tower_ht,
            "TowerBsHt": tower_bs_ht,
            "YawBearing": np.array([0.0, 0.0, tower_ht]),
            "TowerBase": np.array([0.0, 0.0, tower_bs_ht]),
            "OverHang": fetch_value("OverHang") or 0.0,
            "ShftTilt": fetch_value("ShftTilt") or 0.0,
            "Twr2Shft": fetch_value("Twr2Shft") or 0.0,
            "TipRad": fetch_value("TipRad") or 0.0,
            "HubRad": fetch_value("HubRad") or 0.0,
        }
        self.logger.debug(
            f"Extracted geometry: TowerHt={tower_ht:.1f}m, TowerBsHt={tower_bs_ht:.1f}m"
        )
        return geo_data

    def _parse_moordyn_points(self, md_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        self.logger.info(f"Parsing MoorDyn fairlead and anchor points: {md_path}")
        lines = _read_lines(md_path, self.logger)

        points_data = []
        lines_data = []
        in_points = False
        in_lines = False

        for line in lines:
            upper = line.strip().upper()
            if upper.startswith("---"):
                in_points = "POINTS" in upper
                in_lines = "LINES" in upper
                continue
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            if in_points and len(parts) >= 5:
                points_data.append(
                    (int(parts[0]), parts[1].upper(), float(parts[2]), float(parts[3]), float(parts[4]))
                )
            elif in_lines and len(parts) >= 4:
                lines_data.append((int(parts[0]), int(parts[2]), int(parts[3])))

        all_points_map = {
            pid: {"att": att, "pos": np.array([x, y, z])} for pid, att, x, y, z in points_data
        }
        fairleads: Dict[int, np.ndarray] = {}
        anchors: Dict[int, np.ndarray] = {}

        for line_id, pida, pidb in lines_data:
            point_a = all_points_map.get(pida)
            point_b = all_points_map.get(pidb)
            if not point_a or not point_b:
                continue
            if point_a["att"] == "VESSEL" and point_b["att"] == "FIXED":
                fairleads[line_id] = point_a["pos"]
                anchors[line_id] = point_b["pos"]
            elif point_b["att"] == "VESSEL" and point_a["att"] == "FIXED":
                fairleads[line_id] = point_b["pos"]
                anchors[line_id] = point_a["pos"]

        self.logger.info(f"Found {len(fairleads)} fairleads and {len(anchors)} anchors.")
        return fairleads, anchors

    @staticmethod
    def _rotmat_from_rpy_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
        rz = radians(yaw)
        ry = radians(pitch)
        rx = radians(roll)

        cz, sz = cos(rz), sin(rz)
        cy, sy = cos(ry), sin(ry)
        cx, sx = cos(rx), sin(rx)

        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        return Rz @ Ry @ Rx

    class MassPropertyBuilder:
        def __init__(self, fst_path: str, logger: logging.Logger):
            self.logger = logger
            self.fst_path = fst_path
            self.refs = _find_fst_refs(fst_path, self.logger)
            self.ed_path = self.refs.get("EDFile")
            if not self.ed_path or not os.path.isfile(self.ed_path):
                raise RuntimeError("ElastoDyn file not found from FST.")
            self.ed = self._parse_elastodyn(self.ed_path)
            self.twr_path = self._find_tower_file_from_ed()
            if not self.twr_path or not os.path.isfile(self.twr_path):
                raise RuntimeError("ElastoDyn tower properties file not found.")
            self.tower_dist = self._parse_dist_prop(self.twr_path, "HtFract", "TMassDen")
            self.blade_mass_path = self._find_ed_blade_mass_file()
            if not self.blade_mass_path or not os.path.isfile(self.blade_mass_path):
                raise RuntimeError("ElastoDyn blade mass file not found.")
            self.bl_mass_dist = self._parse_dist_prop(self.blade_mass_path, "BlFract", "BMassDen", 3)

        def _read_lines(self, path: str) -> List[str]:
            return _read_lines(path, self.logger)

        def _parse_elastodyn(self, ed_path: str) -> Dict[str, Any]:
            lines = self._read_lines(ed_path)

            def fetch_value(key: str) -> Optional[float]:
                for line in lines:
                    if key in line:
                        try:
                            return float(line.strip().split()[0])
                        except ValueError:
                            return None
                return None

            num_blades = int(fetch_value("NumBl") or 3)
            return {
                "TowerHt": fetch_value("TowerHt") or 0.0,
                "TowerBsHt": fetch_value("TowerBsHt") or 0.0,
                "PtfmMass": fetch_value("PtfmMass") or 0.0,
                "PtfmI": (
                    fetch_value("PtfmRIner") or 0,
                    fetch_value("PtfmPIner") or 0,
                    fetch_value("PtfmYIner") or 0,
                    fetch_value("PtfmXYIner") or 0,
                    fetch_value("PtfmXZIner") or 0,
                    fetch_value("PtfmYZIner") or 0,
                ),
                "PtfmCM": np.array(
                    [
                        fetch_value("PtfmCMxt") or 0,
                        fetch_value("PtfmCMyt") or 0,
                        fetch_value("PtfmCMzt") or 0,
                    ]
                ),
                "NacMass": fetch_value("NacMass") or 0.0,
                "NacYIner": fetch_value("NacYIner") or 0.0,
                "NacCMn": np.array(
                    [
                        fetch_value("NacCMxn") or 0,
                        fetch_value("NacCMyn") or 0,
                        fetch_value("NacCMzn") or 0,
                    ]
                ),
                "HubMass": fetch_value("HubMass") or 0.0,
                "HubIner": fetch_value("HubIner") or 0.0,
                "NumBl": num_blades,
                "TipRad": fetch_value("TipRad") or 0.0,
                "HubRad": fetch_value("HubRad") or 0.0,
                "PreCone": [fetch_value(f"PreCone({i})") or 0.0 for i in range(1, num_blades + 1)],
                "OverHang": fetch_value("OverHang") or 0.0,
                "ShftTilt": fetch_value("ShftTilt") or 0.0,
                "Twr2Shft": fetch_value("Twr2Shft") or 0.0,
            }

        def _find_tower_file_from_ed(self) -> Optional[str]:
            base = os.path.dirname(os.path.abspath(self.ed_path))
            for line in self._read_lines(self.ed_path):
                if "TwrFile" in line:
                    return os.path.normpath(os.path.join(base, _strip_quotes(line.split()[0])))
            return self.refs.get("TwrFile")

        def _find_ed_blade_mass_file(self) -> Optional[str]:
            base = os.path.dirname(os.path.abspath(self.ed_path))
            for line in self._read_lines(self.ed_path):
                if "BldFile(1)" in line or ("BldFile" in line and "ADBlFile" not in line):
                    return os.path.normpath(os.path.join(base, _strip_quotes(line.split()[0])))
            return None

        def _parse_dist_prop(
            self,
            path: Optional[str],
            key1: str,
            key2: str,
            val_idx: int = 1,
        ) -> List[Tuple[float, float]]:
            if not path or not os.path.isfile(path):
                self.logger.warning(f"Dist prop file not found: {path}")
                return []

            lines = self._read_lines(path)
            data: List[Tuple[float, float]] = []
            started = False
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith(("!", "#")):
                    continue
                if key1 in stripped and key2 in stripped:
                    started = True
                    continue
                if started:
                    parts = stripped.split()
                    if len(parts) > max(0, val_idx):
                        try:
                            data.append((float(parts[0]), float(parts[val_idx])))
                        except ValueError:
                            pass
            return sorted(data, key=lambda tup: tup[0])

        @staticmethod
        def parallel_axis(Ic: np.ndarray, mass: float, r: np.ndarray) -> np.ndarray:
            r_vec = np.asarray(r).reshape(3)
            return Ic + mass * ((r_vec @ r_vec) * np.eye(3) - np.outer(r_vec, r_vec))

        def compute(self) -> Tuple[float, np.ndarray, np.ndarray]:
            self.logger.info("Computing mass properties...")
            masses: List[float] = []
            centers: List[np.ndarray] = []
            inertias: List[np.ndarray] = []

            ptfm_mass = self.ed["PtfmMass"]
            ptfm_com = self.ed["PtfmCM"]
            ptfm_inertia = np.array(
                [
                    [self.ed["PtfmI"][0], self.ed["PtfmI"][3], self.ed["PtfmI"][4]],
                    [self.ed["PtfmI"][3], self.ed["PtfmI"][1], self.ed["PtfmI"][5]],
                    [self.ed["PtfmI"][4], self.ed["PtfmI"][5], self.ed["PtfmI"][2]],
                ]
            )
            masses.append(ptfm_mass)
            centers.append(ptfm_com)
            inertias.append(ptfm_inertia)

            tower_masses, tower_centers, tower_inertias = self._tower_mass_properties()
            masses.extend(tower_masses)
            centers.extend(tower_centers)
            inertias.extend(tower_inertias)

            nac_mass = self.ed["NacMass"]
            nac_com = np.array([0, 0, self.ed["TowerHt"]]) + self.ed["NacCMn"]
            nac_inertia = np.diag([0.0, self.ed["NacYIner"], 0.0])
            masses.append(nac_mass)
            centers.append(nac_com)
            inertias.append(nac_inertia)

            hub_mass = self.ed["HubMass"]
            hub_inertia_value = self.ed["HubIner"]
            rotor_origin = np.array([0, 0, self.ed["TowerHt"]])
            r_hub = rotor_origin + np.array([self.ed["OverHang"], 0, self.ed["Twr2Shft"]])
            R_rotor = DalembertRunner._rotmat_from_rpy_deg(0, self.ed["ShftTilt"], 0)
            hub_inertia_tensor = R_rotor @ np.diag([0, hub_inertia_value, 0]) @ R_rotor.T
            masses.append(hub_mass)
            centers.append(r_hub)
            inertias.append(hub_inertia_tensor)

            blade_masses, blade_centers, blade_inertias = self._blades_mass_properties(r_hub, R_rotor)
            masses.extend(blade_masses)
            centers.extend(blade_centers)
            inertias.extend(blade_inertias)

            total_mass = float(np.sum(masses))
            r_com = np.sum([m * np.asarray(r) for m, r in zip(masses, centers)], axis=0) / max(total_mass, 1e-16)
            I_origin = np.sum(
                [self.parallel_axis(Ic, m, r) for m, r, Ic in zip(masses, centers, inertias)],
                axis=0,
            )
            I_com = I_origin - self.parallel_axis(np.zeros((3, 3)), total_mass, r_com)
            return total_mass, r_com, I_com

        def _tower_mass_properties(self) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
            z0 = self.ed["TowerBsHt"]
            z_top = self.ed["TowerHt"]
            height = z_top - z0
            if height <= 0:
                return [], [], []

            z_list = [z0 + hf * height for hf, _ in self.tower_dist]
            md_list = [md for _, md in self.tower_dist]

            masses: List[float] = []
            centers: List[np.ndarray] = []
            inertias: List[np.ndarray] = []

            for i in range(len(z_list) - 1):
                segment_length = z_list[i + 1] - z_list[i]
                if segment_length <= 0:
                    continue
                m_seg = 0.5 * (md_list[i] + md_list[i + 1]) * segment_length
                r_center = np.array([0, 0, 0.5 * (z_list[i] + z_list[i + 1])])
                Ic_local = np.diag([m_seg * segment_length**2 / 12.0] * 2 + [0.0])
                masses.append(m_seg)
                centers.append(r_center)
                inertias.append(Ic_local)
            return masses, centers, inertias

        def _blades_mass_properties(
            self,
            r_hub: np.ndarray,
            R_rotor: np.ndarray,
        ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
            num_blades = self.ed["NumBl"]
            r_root = self.ed["HubRad"]
            blade_length = self.ed["TipRad"] - self.ed["HubRad"]

            fracs = [frac for frac, _ in self.bl_mass_dist]
            mdens = [md for _, md in self.bl_mass_dist]

            def mass_density_at_r(radius: float) -> float:
                if not fracs:
                    return 0.0
                frac = max(0.0, min(1.0, (radius - r_root) / max(blade_length, 1e-9)))
                return float(np.interp(frac, fracs, mdens))

            masses: List[float] = []
            centers: List[np.ndarray] = []
            inertias: List[np.ndarray] = []

            for blade_index, azimuth in enumerate([i * 360.0 / num_blades for i in range(num_blades)]):
                R_azimuth = np.array(
                    [
                        [cos(radians(azimuth)), 0, sin(radians(azimuth))],
                        [0, 1, 0],
                        [-sin(radians(azimuth)), 0, cos(radians(azimuth))],
                    ]
                )
                R_cone = DalembertRunner._rotmat_from_rpy_deg(0, self.ed["PreCone"][blade_index], 0)
                R_blade = R_rotor @ R_azimuth @ R_cone

                spans = sorted(np.linspace(r_root, self.ed["TipRad"], 21))
                for i in range(len(spans) - 1):
                    span_start, span_end = spans[i], spans[i + 1]
                    segment_length = span_end - span_start
                    m_seg = 0.5 * (
                        mass_density_at_r(span_start) + mass_density_at_r(span_end)
                    ) * segment_length
                    r_global = r_hub + (0.5 * (span_start + span_end)) * (
                        R_blade @ np.array([1, 0, 0])
                    )
                    Ic_local = np.diag([0.0, m_seg * segment_length**2 / 12.0, m_seg * segment_length**2 / 12.0])
                    masses.append(m_seg)
                    centers.append(r_global)
                    inertias.append(R_blade @ Ic_local @ R_blade.T)
            return masses, centers, inertias