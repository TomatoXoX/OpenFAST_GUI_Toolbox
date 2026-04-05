"""
views/tutorial_tab.py
======================
Tutorial tab — pure presentation, no business logic.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk


class TutorialTab(ttk.Frame):
    """Renders the tutorial/help content as a read-only styled text widget."""

    def __init__(self, parent: ttk.Widget) -> None:
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        text = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            relief="flat",
            padx=10,
            pady=10,
        )
        text.pack(fill="both", expand=True)

        text.tag_configure("h1", font=("TkDefaultFont", 16, "bold"), spacing3=10)
        text.tag_configure("h2", font=("TkDefaultFont", 12, "bold"), spacing1=15, spacing3=5)
        text.tag_configure("bold", font=("TkDefaultFont", 9, "bold"))
        text.tag_configure("code", font=("Consolas", 9), background="#f0f0f0")

        content = [
            ("Welcome to the OpenFAST Workflow Manager!\n", "h1"),
            (
                "This tool streamlines running large batches of OpenFAST simulations "
                "and analyzing their results.  The workflow is organised into three main tabs.\n\n",
                "",
            ),

            ("Tab 1: Setup Cases\n", "h2"),
            ("The goal of this tab is to create a set of test case directories, each containing "
             "a modified version of a base OpenFAST model.\n\n", ""),

            ("1. File Selection:", "bold"),
            (" Select your main OpenFAST input file (", ""),
            (".fst", "code"),
            (") and specify an Output Directory.\n", ""),

            ("2. Geometry Import (Optional):", "bold"),
            (" Browse & Import a CSV file with platform geometries. Each row is a separate geometry case.\n", ""),

            ("3. Parameter Discovery:", "bold"),
            (" Click ", ""),
            ("Discover Parameters", "code"),
            (" to scan all referenced input files and find tunable parameters.\n", ""),

            ("4. Parameter Configuration:", "bold"),
            (" Click ", ""),
            ("Add from Discovery", "code"),
            (" to select parameters to vary.  These are applied to every geometry case.\n", ""),

            ("5. Generate Cases:", "bold"),
            (" Click ", ""),
            ("Generate Test Cases", "code"),
            (". Each combination of geometry × parameter set becomes one case directory.\n\n", ""),

            ("IMPORTANT: The 5MW baseline folder must be copied into the output "
             "directory if running the included example model.\n", "h2"),

            ("\nTab 2: Run Simulations\n", "h2"),
            ("1. Configuration:", "bold"),
            (" Browse for the ", ""),
            ("OpenFAST executable", "code"),
            (" and set the number of parallel workers.\n", ""),

            ("2. Load Cases:", "bold"),
            (" Click ", ""),
            ("Load Cases", "code"),
            (" to read the summary JSON from the output directory.\n", ""),

            ("3. Run:", "bold"),
            (" Select cases and click ", ""),
            ("Run Selected Simulations", "code"),
            (". Progress is shown in the table and log.\n\n", ""),

            ("\nTab 3: Post-Process Results\n", "h2"),
            ("1. Configuration:", "bold"),
            (" Confirm the results directory and choose which analysis tasks to run.\n", ""),

            ("2. Load & Run:", "bold"),
            (" Load cases and click ", ""),
            ("Run Post-Processing", "code"),
            (".\n", ""),

            ("3. Review:", "bold"),
            (" Right-click any row → ", ""),
            ("Open Folder", "code"),
            (" to view CSVs, reports, and plots.\n\n", ""),

            ("Final Notes\n", "h2"),
            (
                "Author: Trang Vinh Nghi\n"
                "Department of Aerospace Engineering — Ho Chi Minh City University of Technology\n"
                "Email: trangvinhnghi2212@gmail.com\n"
                "GitHub: https://github.com/TomatoXoX/OpenFAST_GUI_Toolbox\n",
                "",
            ),
        ]

        for chunk, tag in content:
            text.insert(tk.END, chunk, tag)

        text.config(state="disabled")
