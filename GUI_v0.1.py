import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import re
import subprocess
import threading
import queue
import itertools

class OpenFASTTestCaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenFAST Test Case Generator")
        self.root.geometry("1200x800")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Accent.TButton", foreground="white", background="#0078D7")
        style.configure("Disabled.TLabel", foreground="gray")
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.setup_tab = ttk.Frame(self.notebook)
        self.run_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_tab, text="Setup Test Cases")
        self.notebook.add(self.run_tab, text="Run Tests")
        
        self.base_fst_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="test_cases")
        self.num_cases = tk.IntVar(value=10)
        self.parameter_entries = []
        self.discovered_parameters = {}
        self.file_structure = {}
        self.openfast_exe = tk.StringVar()
        self.num_threads = tk.IntVar(value=max(1, os.cpu_count() // 2))
        self.run_button = None
        self.job_queue = queue.Queue()
        self.progress_lock = threading.Lock()
        self.completed_cases = 0
        self.total_cases_to_run = 0
        self.message_queue = queue.Queue()
        
        self.create_setup_tab()
        self.create_run_tab()
        self.process_queue()
        
    def create_setup_tab(self):
        canvas = tk.Canvas(self.setup_tab)
        scrollbar = ttk.Scrollbar(self.setup_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.create_file_selection_section(scrollable_frame)
        self.create_test_config_section(scrollable_frame)
        self.create_parameter_discovery_section(scrollable_frame)
        self.create_parameter_section(scrollable_frame)
        self.create_action_section(scrollable_frame)
        self.create_log_section(scrollable_frame)
        
    def create_run_tab(self):
        main_frame = ttk.Frame(self.run_tab, padding="10")
        main_frame.pack(fill='both', expand=True)
        config_frame = ttk.LabelFrame(main_frame, text="Run Configuration", padding="10")
        config_frame.pack(fill='x', pady=5)
        ttk.Label(config_frame, text="OpenFAST Path:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(config_frame, textvariable=self.openfast_exe, width=50).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_frame, text="Browse", command=self.browse_openfast_exe).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(config_frame, text="Number of parallel runs:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Spinbox(config_frame, from_=1, to=os.cpu_count() or 8, textvariable=self.num_threads, width=8).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        config_frame.columnconfigure(1, weight=1)
        case_frame = ttk.LabelFrame(main_frame, text="Test Cases", padding="10")
        case_frame.pack(fill='both', expand=True, pady=5)
        btn_frame = ttk.Frame(case_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Load Test Cases", command=self.load_test_cases).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Select All", command=self.select_all_cases).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Deselect All", command=self.deselect_all_cases).pack(side='left', padx=5)
        self.run_button = ttk.Button(btn_frame, text="Run Selected", command=self.run_selected_cases, style="Accent.TButton")
        self.run_button.pack(side='left', padx=20)
        list_frame = ttk.Frame(case_frame)
        list_frame.pack(fill='both', expand=True)
        columns = ('Status', 'Parameters', 'Runtime', 'Result')
        self.case_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10, selectmode='extended')
        self.case_tree.heading('#0', text='Test Case'); self.case_tree.column('#0', width=150, anchor='w')
        for col in columns: self.case_tree.heading(col, text=col)
        self.case_tree.column('Status', width=100); self.case_tree.column('Parameters', width=300)
        self.case_tree.column('Runtime', width=100, anchor='center'); self.case_tree.column('Result', width=200)
        tree_scroll_y = ttk.Scrollbar(list_frame, orient="vertical", command=self.case_tree.yview)
        tree_scroll_x = ttk.Scrollbar(list_frame, orient="horizontal", command=self.case_tree.xview)
        self.case_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.case_tree.grid(row=0, column=0, sticky='nsew'); tree_scroll_y.grid(row=0, column=1, sticky='ns'); tree_scroll_x.grid(row=1, column=0, sticky='ew')
        list_frame.grid_rowconfigure(0, weight=1); list_frame.grid_columnconfigure(0, weight=1)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        log_frame = ttk.LabelFrame(main_frame, text="Execution Log", padding="10")
        log_frame.pack(fill='both', expand=True, pady=5)
        self.run_log = scrolledtext.ScrolledText(log_frame, height=10, width=80, wrap=tk.WORD)
        self.run_log.pack(fill='both', expand=True)
        self.test_cases = {}
        
    def create_file_selection_section(self, parent):
        frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        frame.pack(fill='x', pady=5, padx=5)
        ttk.Label(frame, text="Base FST File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame, textvariable=self.base_fst_path, width=60).grid(row=0, column=1, padx=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse", command=self.browse_fst_file).grid(row=0, column=2, padx=5)
        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.output_dir, width=60).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        
    def create_test_config_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Test Configuration", padding="10")
        frame.pack(fill='x', pady=5, padx=5)
        ttk.Label(frame, text="Number of Test Cases:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.num_cases_spinbox = ttk.Spinbox(frame, from_=2, to=10000, textvariable=self.num_cases, width=10)
        self.num_cases_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(frame, text="Distribution Type:").grid(row=0, column=2, sticky=tk.W, padx=20)
        self.distribution_var = tk.StringVar(value="grid_search")
        dist_combo = ttk.Combobox(frame, textvariable=self.distribution_var, values=["grid_search", "csv_columnwise", "latin_hypercube", "uniform", "normal"], width=15)
        dist_combo.grid(row=0, column=3, sticky=tk.W, padx=5)
        dist_combo.bind("<<ComboboxSelected>>", self.on_distribution_change)
        
    def create_parameter_discovery_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameter Discovery", padding="10")
        frame.pack(fill='x', pady=5, padx=5)
        ttk.Button(frame, text="Discover Parameters", command=self.discover_parameters, style="Accent.TButton").pack(side='left', padx=5)
        self.discovery_status = ttk.Label(frame, text="Select a .fst file and click 'Discover Parameters'")
        self.discovery_status.pack(side='left', padx=20)
        
    def create_parameter_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Parameter Configuration", padding="10")
        frame.pack(fill='both', expand=True, pady=5, padx=5)
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', pady=5)
        ttk.Button(control_frame, text="Add from Discovery", command=self.show_parameter_selector).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_parameters).pack(side='left', padx=5)
        canvas = tk.Canvas(frame, height=250)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.param_list_frame = ttk.Frame(canvas)
        self.param_list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.param_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_action_section(self, parent):
        frame = ttk.Frame(parent, padding="5")
        frame.pack(fill='x', pady=10)
        ttk.Button(frame, text="Generate Test Cases", command=self.generate_test_cases, style="Accent.TButton").pack(side='left', padx=5)
        ttk.Button(frame, text="Load Configuration", command=self.load_config).pack(side='left', padx=5)
        ttk.Button(frame, text="Save Configuration", command=self.save_config).pack(side='left', padx=5)
        ttk.Button(frame, text="View File Structure", command=self.show_file_structure).pack(side='left', padx=5)
        
    def create_log_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Output Log", padding="10")
        frame.pack(fill='both', expand=True, pady=5, padx=5)
        self.log_text = scrolledtext.ScrolledText(frame, height=8, width=80, wrap=tk.WORD)
        self.log_text.pack(fill='both', expand=True)
        
    def resolve_file_path(self, base_dir, filename):
        if not filename or filename.lower() in ['unused', 'none', '']: return None
        filename = filename.strip('"').strip("'")
        paths_to_check = [ base_dir / Path(filename), Path(filename), base_dir / Path(filename).name, Path(self.base_fst_path.get()).parent / Path(filename) ]
        for path in paths_to_check:
            try:
                if path.exists(): return path.resolve()
            except: continue
        self.log(f"Warning: Could not find file: {filename}"); return None
        
    def _find_referenced_files(self, file_path, base_dir):
        found_files = {}
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
        except Exception as e: self.log(f"Could not read file {file_path}: {e}"); return {}
        pattern = re.compile(r'^\s*["\']([^"\']+\.(?:dat|txt|csv|af|ipt|bld|twr|in))["\']\s+([a-zA-Z_][a-zA-Z0-9_\(\)]*)\s*', re.MULTILINE | re.IGNORECASE)
        for match in pattern.finditer(content):
            filename, keyword = match.group(1).strip(), match.group(2).strip()
            if keyword.lower() in ['true', 'false', 'default', 'none', 'unused', 'echo']: continue
            resolved_path = self.resolve_file_path(base_dir, filename)
            if resolved_path: found_files[keyword] = resolved_path
        num_af_match = re.search(r'^\s*(\d+)\s+NumAFfiles', content, re.MULTILINE | re.IGNORECASE)
        if num_af_match:
            num_af_files = int(num_af_match.group(1))
            af_block_pattern = re.compile(r'^\s*["\']([^"\']+)["\']', re.MULTILINE)
            start_pos = num_af_match.end()
            af_matches = af_block_pattern.finditer(content, pos=start_pos)
            for i, match in enumerate(af_matches):
                if i >= num_af_files: break
                filename = match.group(1).strip()
                resolved_path = self.resolve_file_path(base_dir, filename)
                if resolved_path: found_files[f'AirfoilFile_{i+1}'] = resolved_path
        return found_files
        
    def discover_parameters(self):
        if not self.base_fst_path.get(): messagebox.showerror("Error", "Please select a base FST file first"); return
        self.log("Starting parameter discovery..."); self.discovery_status.config(text="Scanning files..."); self.root.update()
        self.discovered_parameters, self.file_structure = {}, {}
        try:
            initial_fst_path = Path(self.base_fst_path.get())
            files_to_scan, processed_paths = [('Main_FST', initial_fst_path)], set()
            while files_to_scan:
                file_key, file_path = files_to_scan.pop(0)
                if not file_path or file_path in processed_paths or not file_path.exists(): continue
                self.log(f"Processing {file_key}: {file_path.name}"); processed_paths.add(file_path)
                self.file_structure[file_key] = {'path': file_path, 'params': {}}
                newly_found_files = self._find_referenced_files(file_path, file_path.parent)
                for new_key, new_path in newly_found_files.items():
                    if new_path not in processed_paths:
                        unique_key = new_key; i = 2
                        while f"{new_key}_{i}" in self.file_structure: i += 1
                        unique_key = f"{new_key}_{i}" if unique_key in self.file_structure else new_key
                        files_to_scan.append((unique_key, new_path))
            total_params = 0
            for file_key, file_info in self.file_structure.items():
                path = file_info.get('path')
                if path and path.exists():
                    params = self.extract_parameters_from_file(path, file_key)
                    if params: self.discovered_parameters[file_key] = params; total_params += len(params)
            self.discovery_status.config(text=f"Discovered {total_params} parameters across {len(self.file_structure)} files.")
            self.log(f"Parameter discovery complete: {total_params} parameters found.")
        except Exception as e: self.log(f"Error during parameter discovery: {str(e)}"); messagebox.showerror("Error", f"Failed to discover parameters: {str(e)}")
        
    def extract_parameters_from_file(self, file_path, file_type):
        parameters = {}
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: lines = f.readlines()
        param_pattern = re.compile(r'^\s*([^\s!#"]+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[-!]')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith(('!', '#', '-', '=')): continue
            match = param_pattern.match(line_stripped)
            if match:
                value_str, param_name = match.groups()
                if param_name.lower() in ['true', 'false', 'default', 'unused', 'none', 'end', 'echo'] or any(ext in value_str for ext in ['.dat', '.txt', '.csv']): continue
                param_info = self.parse_parameter_value(value_str, param_name, line)
                if param_info:
                    parameters[param_name] = {'line_number': i, 'original_value': param_info['value'], 'type': param_info['type'], 'description': line.split('!', 1)[-1].split('-', 1)[-1].strip(), 'unit': self.extract_unit(line)}
        return parameters
        
    def parse_parameter_value(self, value_str, param_name, description):
        value_str = value_str.strip().strip('"\'')
        if value_str.upper() in ['DEFAULT']: return None
        try:
            value = float(value_str)
            if value == int(value) and '.' not in value_str and 'e' not in value_str.lower(): return {'value': int(value), 'type': 'int'}
            else: return {'value': value, 'type': 'float'}
        except ValueError: pass
        if value_str.lower() in ['true', 'false']: return {'value': value_str.lower() == 'true', 'type': 'bool'}
        if any(keyword in description.lower() for keyword in ['option', 'method', 'model', 'type', 'switch', 'code', 'name', 'file']): return {'value': value_str, 'type': 'option'}
        return None
        
    def extract_unit(self, description):
        match = re.search(r'\(([^)]+)\)', description)
        if match:
            unit = match.group(1)
            if not any(word in unit.lower() for word in ['flag', 'switch', 'quoted', 'string', 'option']): return unit
        return ''
        
    def show_parameter_selector(self):
        if not self.discovered_parameters: messagebox.showinfo("Info", "Run 'Discover Parameters' first."); return
        dialog = tk.Toplevel(self.root); dialog.title("Select Parameters to Vary"); dialog.geometry("900x700")
        search_frame = ttk.Frame(dialog); search_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side='left', padx=5)
        search_var = tk.StringVar(); search_entry = ttk.Entry(search_frame, textvariable=search_var, width=30); search_entry.pack(side='left', padx=5)
        tree_frame = ttk.Frame(dialog); tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        tree = ttk.Treeview(tree_frame, columns=('Type', 'Value', 'Unit', 'Description'), show='tree headings')
        tree.heading('#0', text='Parameter'); tree.heading('Type', text='Type'); tree.heading('Value', text='Current Value'); tree.heading('Unit', text='Unit'); tree.heading('Description', text='Description')
        tree.column('#0', width=200); tree.column('Type', width=80); tree.column('Value', width=100, anchor='e'); tree.column('Unit', width=80); tree.column('Description', width=350)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview); hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set); tree.grid(row=0, column=0, sticky='nsew'); vsb.grid(row=0, column=1, sticky='ns'); hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1); tree_frame.grid_columnconfigure(0, weight=1)
        all_items = []
        for file_type, params in sorted(self.discovered_parameters.items()):
            file_node = tree.insert('', 'end', text=file_type, open=False, tags=('file_node',))
            for param_name, param_info in sorted(params.items()):
                val_str = f"{param_info['original_value']:.4g}" if isinstance(param_info['original_value'], float) else str(param_info['original_value'])
                item = tree.insert(file_node, 'end', text=param_name, values=(param_info['type'], val_str, param_info.get('unit', ''), param_info['description'][:100]))
                all_items.append((item, file_type.lower(), param_name.lower(), param_info['description'].lower()))
        tree.tag_configure('file_node', font=('TkDefaultFont', 10, 'bold'))
        def search_params(*args):
            search_term = search_var.get().lower()
            for child in tree.get_children(): tree.item(child, open=False); tree.reattach(child, '', 'end')
            if not search_term: return
            for child in tree.get_children(): tree.detach(child)
            for item, file_type, param_name, desc in all_items:
                if search_term in param_name or search_term in desc or search_term in file_type:
                    parent = tree.parent(item); tree.reattach(parent, '', 'end'); tree.item(parent, open=True)
        search_var.trace('w', search_params)
        btn_frame = ttk.Frame(dialog); btn_frame.pack(fill='x', pady=10, padx=10)
        def add_selected():
            added_count = 0
            for item in tree.selection():
                parent = tree.parent(item)
                if parent:
                    file_type = tree.item(parent)['text']; param_name = tree.item(item)['text']
                    self.add_parameter_with_info(file_type, param_name, self.discovered_parameters[file_type][param_name])
                    added_count += 1
            dialog.destroy()
            if added_count > 0: self.log(f"Added {added_count} parameters for variation.")
        ttk.Button(btn_frame, text="Add Selected", command=add_selected, style="Accent.TButton").pack(side='right')
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='right', padx=5)

    def add_parameter_with_info(self, file_type, param_name, param_info):
        for entry in self.parameter_entries:
            if entry['file_type'] == file_type and entry['param_name'] == param_name:
                self.log(f"Parameter {file_type} - {param_name} is already added."); return

        row_frame = ttk.Frame(self.param_list_frame)
        row_frame.pack(fill='x', pady=4, padx=2)
        
        param_label = ttk.Label(row_frame, text=f"{file_type} - {param_name}", width=35, anchor='w', wraplength=220)
        param_label.grid(row=0, column=0, rowspan=2, padx=5, sticky='w')
        
        param_type = param_info['type']
        current_val = param_info['original_value']
        
        entry_data = {'frame': row_frame, 'file_type': file_type, 'param_name': param_name, 'param_info': param_info, 'widgets': {}}

        # --- CSV Mode Widget (add for all types) ---
        csv_var = tk.StringVar(value=str(current_val))
        entry_data['widgets']['csv_lbl'] = ttk.Label(row_frame, text="CSV Values:")
        entry_data['widgets']['csv_ent'] = ttk.Entry(row_frame, textvariable=csv_var, width=40)
        entry_data.update({'csv_var': csv_var})
        csv_var.trace_add("write", self.update_total_cases)

        if param_type == 'float':
            start_default, end_default = (current_val * 0.8, current_val * 1.2) if isinstance(current_val, (int, float)) and abs(current_val) > 1e-9 else (-1.0, 1.0)
            start_var, end_var, steps_var = tk.DoubleVar(value=start_default), tk.DoubleVar(value=end_default), tk.IntVar(value=5)
            
            entry_data['widgets']['range_lbl_s'] = ttk.Label(row_frame, text="Start:")
            entry_data['widgets']['range_ent_s'] = ttk.Entry(row_frame, textvariable=start_var, width=10)
            entry_data['widgets']['range_lbl_e'] = ttk.Label(row_frame, text="End:")
            entry_data['widgets']['range_ent_e'] = ttk.Entry(row_frame, textvariable=end_var, width=10)
            entry_data['widgets']['range_lbl_st'] = ttk.Label(row_frame, text="Steps:")
            entry_data['widgets']['range_spn_st'] = ttk.Spinbox(row_frame, from_=1, to=100, textvariable=steps_var, width=5)
            
            entry_data.update({'start_var': start_var, 'end_var': end_var, 'steps_var': steps_var})
            steps_var.trace_add("write", self.update_total_cases)

        elif param_type == 'int':
            int_mode_var = tk.StringVar(value="Range")
            start_var, end_var, steps_var = tk.DoubleVar(value=current_val), tk.DoubleVar(value=current_val+4), tk.IntVar(value=5)
            list_var = tk.StringVar(value=str(current_val))

            def update_int_widgets():
                mode = int_mode_var.get()
                is_range = mode == "Range"
                for name, w in entry_data['widgets'].items():
                    if name.startswith('range_'): w.grid() if is_range else w.grid_remove()
                    if name.startswith('list_'): w.grid() if not is_range else w.grid_remove()
                self.update_total_cases()
            
            entry_data['update_func'] = update_int_widgets

            entry_data['widgets']['rad_range'] = ttk.Radiobutton(row_frame, text="Range", variable=int_mode_var, value="Range", command=update_int_widgets)
            entry_data['widgets']['rad_list'] = ttk.Radiobutton(row_frame, text="List", variable=int_mode_var, value="List", command=update_int_widgets)
            
            entry_data['widgets']['range_lbl_s'] = ttk.Label(row_frame, text="Start:"); entry_data['widgets']['range_ent_s'] = ttk.Entry(row_frame, textvariable=start_var, width=8)
            entry_data['widgets']['range_lbl_e'] = ttk.Label(row_frame, text="End:"); entry_data['widgets']['range_ent_e'] = ttk.Entry(row_frame, textvariable=end_var, width=8)
            entry_data['widgets']['range_lbl_st'] = ttk.Label(row_frame, text="Steps:"); entry_data['widgets']['range_spn_st'] = ttk.Spinbox(row_frame, from_=1, to=100, textvariable=steps_var, width=5)
            entry_data['widgets']['list_lbl'] = ttk.Label(row_frame, text="List (CSV):"); entry_data['widgets']['list_ent'] = ttk.Entry(row_frame, textvariable=list_var, width=25)

            entry_data.update({'int_mode_var': int_mode_var, 'start_var': start_var, 'end_var': end_var, 'steps_var': steps_var, 'list_var': list_var})
            steps_var.trace_add("write", self.update_total_cases); list_var.trace_add("write", self.update_total_cases)

        elif param_type == 'bool':
            bool_var = tk.StringVar(value="Vary (True & False)")
            entry_data['widgets']['bool_lbl'] = ttk.Label(row_frame, text="Value:")
            entry_data['widgets']['bool_combo'] = ttk.Combobox(row_frame, textvariable=bool_var, values=["Vary (True & False)", "True", "False"], width=20)
            entry_data.update({'bool_var': bool_var}); bool_var.trace_add("write", self.update_total_cases)

        elif param_type == 'option':
            options_var = tk.StringVar(value=f'"{current_val}"')
            entry_data['widgets']['opt_lbl'] = ttk.Label(row_frame, text="Options (CSV):")
            entry_data['widgets']['opt_ent'] = ttk.Entry(row_frame, textvariable=options_var, width=30)
            entry_data.update({'options_var': options_var}); options_var.trace_add("write", self.update_total_cases)

        info_text = f"[{param_info.get('unit', '')}] (Type: {param_type}, Current: {current_val})"
        info_label = ttk.Label(row_frame, text=info_text, foreground='gray')
        entry_data['widgets']['info_lbl'] = info_label
        
        remove_btn = ttk.Button(row_frame, text="Remove", command=lambda e=entry_data: self.remove_parameter(e))
        entry_data['widgets']['remove_btn'] = remove_btn
        
        row_frame.columnconfigure(8, weight=1)
        self.parameter_entries.append(entry_data)
        self.on_distribution_change()

    def remove_parameter(self, entry_to_remove):
        entry_to_remove['frame'].destroy()
        self.parameter_entries.remove(entry_to_remove)
        self.update_total_cases()
        
    def generate_test_cases(self):
        if not self.base_fst_path.get() or not self.parameter_entries:
            messagebox.showerror("Error", "Please select a base FST file and add at least one parameter.")
            return
            
        self.log_text.delete(1.0, tk.END)
        self.log("Starting test case generation...")
        
        try:
            output_path = Path(self.output_dir.get())
            if output_path.exists() and any(output_path.iterdir()):
                if not messagebox.askyesno("Warning", f"Output directory '{output_path}' is not empty. Overwrite?"):
                    return
            output_path.mkdir(parents=True, exist_ok=True)
            
            num_cases = self.num_cases.get()
            parameter_values = self.generate_parameter_values(num_cases)
            if parameter_values.size == 0:
                self.log("No parameter values generated. Aborting.")
                return
            
            num_cases = parameter_values.shape[1]
            test_summary = []
            
            for i in range(num_cases):
                case_name = f"case_{i+1:04d}"
                case_dir = output_path / case_name
                case_dir.mkdir(exist_ok=True)
                self.log(f"Creating test case {i+1}/{num_cases}: {case_name}")
                
                for file_info in self.file_structure.values():
                    if file_info['path'].exists():
                        shutil.copy2(file_info['path'], case_dir / file_info['path'].name)
                
                case_params = {}
                for j, param_entry in enumerate(self.parameter_entries):
                    file_type, param_name, param_info = param_entry['file_type'], param_entry['param_name'], param_entry['param_info']
                    
                    value = parameter_values[j][i]
                    
                    if isinstance(value, np.integer):
                        value = int(value)
                    elif isinstance(value, np.floating):
                        value = float(value)

                    case_params[f"{file_type}/{param_name}"] = value
                    self.modify_parameter_in_file(case_dir, file_type, param_name, value, param_info)
                
                case_info = {'case_name': case_name, 'fst_file': Path(self.base_fst_path.get()).name, 'parameters': case_params, 'created': datetime.now().isoformat()}
                test_summary.append(case_info)
                with open(case_dir / 'case_info.json', 'w') as f:
                    json.dump(case_info, f, indent=2)
            
            summary_file = output_path / "test_cases_summary.json"
            with open(summary_file, 'w') as f:
                json.dump({'generation_date': datetime.now().isoformat(), 'base_fst_file': self.base_fst_path.get(), 'num_cases': num_cases, 'distribution': self.distribution_var.get(), 'test_cases': test_summary, 'file_structure': {k: str(v.get('path')) for k, v in self.file_structure.items()}}, f, indent=4)
            
            self.log(f"Successfully generated {num_cases} test cases in '{output_path}'")
            if messagebox.askyesno("Success", f"Generated {num_cases} test cases.\nSwitch to 'Run Tests' tab?"):
                self.notebook.select(self.run_tab)
                self.output_dir.set(str(output_path))
                self.load_test_cases()
        except Exception as e:
            self.log(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate test cases: {str(e)}")
    def modify_parameter_in_file(self, case_dir, file_type, param_name, value, param_info):
        original_path = self.file_structure[file_type]['path']
        file_path = case_dir / original_path.name
        if not file_path.exists(): self.log(f"Warning: File {file_path} not found for parameter {param_name}"); return
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: lines = f.readlines()
        modified, line_num = False, param_info.get('line_number', -1)
        if 0 <= line_num < len(lines) and param_name in lines[line_num]:
            lines[line_num] = self.format_parameter_line(lines[line_num], value, param_info); modified = True
        if not modified:
            for i, line in enumerate(lines):
                if re.search(r'\b' + re.escape(param_name) + r'\b', line) and not line.strip().startswith(('!', '#')):
                    lines[i] = self.format_parameter_line(line, value, param_info); modified = True; break
        if modified:
            with open(file_path, 'w', encoding='utf-8', errors='ignore') as f: f.writelines(lines)
        else: self.log(f"Warning: Parameter '{param_name}' not found in {file_path.name}")
            
    def format_parameter_line(self, line, new_value, param_info):
        param_type = param_info.get('type')
        if param_type == 'int': value_str = str(int(new_value))
        elif param_type == 'float': value_str = f"{new_value:.7f}".rstrip('0').rstrip('.') if 0.001 < abs(new_value) < 10000 else f"{new_value:.6e}"
        elif param_type == 'bool': value_str = str(bool(new_value))
        elif param_type == 'option': value_str = f'"{new_value}"' if ' ' in str(new_value) else str(new_value)
        else: value_str = str(new_value)
        parts = line.split()
        if not parts: return line
        old_value = parts[0]
        old_value_pos = line.find(old_value)
        new_line = line[:old_value_pos] + value_str + line[old_value_pos + len(old_value):]
        return new_line if new_line.endswith('\n') else new_line + '\n'
        
    def generate_parameter_values(self, num_cases):
        dist = self.distribution_var.get()
        if dist == "grid_search":
            self.log("Using Grid Search for all parameter combinations.")
            if not self.parameter_entries: return np.array([])
            individual_param_steps = []
            for entry in self.parameter_entries:
                param_type = entry['param_info']['type']
                values = []
                if param_type == 'float':
                    start, end, steps = entry['start_var'].get(), entry['end_var'].get(), entry['steps_var'].get()
                    values = np.array([start]) if steps == 1 else np.linspace(start, end, steps)
                elif param_type == 'int':
                    if entry['int_mode_var'].get() == 'Range':
                        start, end, steps = entry['start_var'].get(), entry['end_var'].get(), entry['steps_var'].get()
                        values = np.array([start]) if steps == 1 else np.linspace(start, end, steps)
                        values = np.round(values).astype(int)
                    else: # List mode
                        list_str = entry['list_var'].get()
                        for item in list_str.split(','):
                            item = item.strip()
                            if item:
                                try: values.append(int(item))
                                except ValueError: self.log(f"Warning: Invalid item '{item}' in integer list for {entry['param_name']}. Ignoring.")
                elif param_type == 'bool':
                    choice = entry['bool_var'].get()
                    values = [True, False] if "Vary" in choice else [choice == "True"]
                elif param_type == 'option':
                    values = [opt.strip().strip('"\'') for opt in entry['options_var'].get().split(',') if opt.strip()]
                individual_param_steps.append(values if len(values) > 0 else [entry['param_info']['original_value']])
            combinations = list(itertools.product(*individual_param_steps))
            return np.array(combinations, dtype=object).T
        elif dist == "csv_columnwise":
            self.log("Using CSV Column-wise generation.")
            if not self.parameter_entries: return np.array([])
            
            all_value_lists, param_names_for_error = [], []
            for entry in self.parameter_entries:
                param_info, param_name = entry['param_info'], entry['param_name']
                param_type = param_info['type']
                param_names_for_error.append(param_name)
                csv_string = entry['csv_var'].get()
                str_values = [item.strip() for item in csv_string.split(',') if item.strip()]
                
                try:
                    if param_type == 'float': typed_values = [float(v) for v in str_values]
                    elif param_type == 'int': typed_values = [int(v) for v in str_values]
                    elif param_type == 'bool': typed_values = [v.lower() in ['true', '1', 't', 'y', 'yes'] for v in str_values]
                    elif param_type == 'option': typed_values = [v.strip('"\'') for v in str_values]
                    else: typed_values = str_values
                    all_value_lists.append(typed_values)
                except ValueError as e:
                    msg = f"Invalid value in CSV for parameter '{param_name}'. Please ensure all values match the type '{param_type}'. Details: {e}"
                    self.log(f"Error: {msg}"); messagebox.showerror("Input Error", msg); return np.array([])

            if not all_value_lists: self.log("No values provided in CSV inputs."); return np.array([])
            first_len = len(all_value_lists[0])
            if first_len == 0: self.log("CSV inputs are empty."); return np.array([])
            for i, lst in enumerate(all_value_lists):
                if len(lst) != first_len:
                    msg = f"All CSV inputs must have the same number of values. Expected {first_len}, but parameter '{param_names_for_error[i]}' has {len(lst)}."
                    self.log(f"Error: {msg}"); messagebox.showerror("Input Error", msg); return np.array([])
            
            self.log(f"Validated {len(all_value_lists)} parameters with {first_len} cases each.")
            return np.array(all_value_lists, dtype=object)
        else: # Sampling distributions
            self.log(f"Using {dist} sampling for {num_cases} cases.")
            parameter_values, numeric_params = [], [p for p in self.parameter_entries if p['param_info']['type'] in ['float', 'int']]
            if not numeric_params: self.log("Warning: Sampling distributions require at least one numeric parameter."); return np.array([])
            num_numeric_params = len(numeric_params)
            try: from scipy.stats import qmc; scipy_available = True
            except ImportError: scipy_available = False
            if dist == "latin_hypercube" and not scipy_available: self.log("Warning: 'scipy' not installed. Falling back to uniform."); dist = "uniform"
            sample = qmc.LatinHypercube(d=num_numeric_params).sample(n=num_cases) if dist == "latin_hypercube" and scipy_available else np.random.rand(num_cases, num_numeric_params)
            param_idx = 0
            for entry in numeric_params:
                param_type = entry['param_info']['type']
                min_val, max_val = entry['start_var'].get(), entry['end_var'].get()
                scaled_sample = min_val + (max_val - min_val) * sample[:, param_idx]
                values = np.round(scaled_sample).astype(int) if param_type == 'int' else scaled_sample
                parameter_values.append(values); param_idx += 1
            return np.array(parameter_values)

    def on_distribution_change(self, event=None):
        dist_mode = self.distribution_var.get()
        is_grid = dist_mode == "grid_search"
        is_csv = dist_mode == "csv_columnwise"
        is_sampling = not is_grid and not is_csv

        self.num_cases_spinbox.config(state='disabled' if is_grid or is_csv else 'normal')

        for entry in self.parameter_entries:
            # Hide all controllable widgets first
            for w in entry['widgets'].values():
                if hasattr(w, 'grid_remove'): w.grid_remove()

            param_type = entry['param_info']['type']
            if is_csv:
                entry['widgets']['csv_lbl'].grid(row=0, column=1, padx=(10, 2))
                entry['widgets']['csv_ent'].grid(row=0, column=2, columnspan=5, sticky='ew')
            else:  # Grid or Sampling
                # Restore widget layout from add_parameter_with_info
                if param_type == 'float':
                    entry['widgets']['range_lbl_s'].grid(row=0, column=1, padx=(10, 2)); entry['widgets']['range_ent_s'].grid(row=0, column=2)
                    entry['widgets']['range_lbl_e'].grid(row=0, column=3, padx=5); entry['widgets']['range_ent_e'].grid(row=0, column=4)
                    entry['widgets']['range_lbl_st'].grid(row=0, column=5, padx=5); entry['widgets']['range_spn_st'].grid(row=0, column=6)
                elif param_type == 'int':
                    entry['widgets']['rad_range'].grid(row=0, column=1, sticky='w', padx=5)
                    entry['widgets']['rad_list'].grid(row=1, column=1, sticky='w', padx=5)
                    if 'update_func' in entry: entry['update_func']()
                elif param_type == 'bool':
                    entry['widgets']['bool_lbl'].grid(row=0, column=1, padx=(10,2))
                    entry['widgets']['bool_combo'].grid(row=0, column=2, columnspan=3)
                elif param_type == 'option':
                    entry['widgets']['opt_lbl'].grid(row=0, column=1, padx=(10,2))
                    entry['widgets']['opt_ent'].grid(row=0, column=2, columnspan=5, sticky='ew')

                # Set widget state based on mode (Grid vs Sampling)
                if is_sampling:
                    is_numeric = param_type in ['float', 'int']
                    for name, widget in entry['widgets'].items():
                        if hasattr(widget, 'config') and name not in ['info_lbl', 'remove_btn']:
                            widget.config(state='disabled')
                    if is_numeric:
                        entry['widgets']['range_ent_s'].config(state='normal')
                        entry['widgets']['range_ent_e'].config(state='normal')

            # Always show the non-controllable widgets
            entry['widgets']['info_lbl'].grid(row=0, column=8, padx=5, sticky='w')
            entry['widgets']['remove_btn'].grid(row=0, column=9, rowspan=2, padx=10)
        
        self.update_total_cases()

    def update_total_cases(self, *args):
        dist_mode = self.distribution_var.get()
        if dist_mode == "grid_search":
            total_cases = 1 if self.parameter_entries else 0
            for entry in self.parameter_entries:
                param_type = entry['param_info']['type']
                try:
                    if param_type == 'float': total_cases *= entry['steps_var'].get()
                    elif param_type == 'int':
                        if entry['int_mode_var'].get() == 'Range': total_cases *= entry['steps_var'].get()
                        else: total_cases *= max(1, len([i for i in entry['list_var'].get().split(',') if i.strip()]))
                    elif param_type == 'bool':
                        if "Vary" in entry['bool_var'].get(): total_cases *= 2
                    elif param_type == 'option':
                        total_cases *= max(1, len([o for o in entry['options_var'].get().split(',') if o.strip()]))
                except (tk.TclError, ValueError): pass
            self.num_cases.set(total_cases)
        elif dist_mode == "csv_columnwise":
            total_cases = 0
            if self.parameter_entries:
                try:
                    csv_str = self.parameter_entries[0]['csv_var'].get()
                    total_cases = len([i for i in csv_str.split(',') if i.strip()])
                except (tk.TclError, IndexError): pass
            self.num_cases.set(total_cases)
            
    def browse_fst_file(self):
        filename = filedialog.askopenfilename(title="Select base FST file", filetypes=[("FST files", "*.fst"), ("All files", "*.*")])
        if filename: self.base_fst_path.set(filename); self.log("Selected FST file: " + filename)
        if filename and messagebox.askyesno("Discover Parameters", "Discover parameters for this file now?"): self.discover_parameters()
        
    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname: self.output_dir.set(dirname); self.log("Selected output directory: " + dirname)
        
    def browse_openfast_exe(self):
        filename = filedialog.askopenfilename(title="Select OpenFAST executable", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if filename: self.openfast_exe.set(filename); self.run_log_message(f"Selected OpenFAST executable: {filename}")
        
    def load_test_cases(self):
        test_dir = self.output_dir.get() or filedialog.askdirectory(title="Select Test Case Directory")
        if not test_dir: return
        self.output_dir.set(test_dir); self.case_tree.delete(*self.case_tree.get_children()); self.test_cases = {}
        summary_file = Path(test_dir) / "test_cases_summary.json"
        if not summary_file.exists(): messagebox.showerror("Error", f"Could not find 'test_cases_summary.json' in {test_dir}"); return
        with open(summary_file, 'r') as f: summary = json.load(f)
        for case_info in summary.get('test_cases', []):
            params_str = ', '.join([f"{k.split('/')[-1]}={v:.3g}" if isinstance(v, (int,float)) else f"{k.split('/')[-1]}={v}" for k, v in case_info['parameters'].items()])
            item_id = self.case_tree.insert('', 'end', text=case_info['case_name'], values=('Ready', params_str, '-', '-'))
            self.test_cases[item_id] = {'path': Path(test_dir) / case_info['case_name'], 'fst_file': case_info['fst_file'], 'name': case_info['case_name']}
        self.run_log_message(f"Loaded {len(self.test_cases)} test cases from {test_dir}"); self.select_all_cases()
        
    def select_all_cases(self): self.case_tree.selection_set(self.case_tree.get_children())
    def deselect_all_cases(self): self.case_tree.selection_set([])
    
    def run_selected_cases(self):
        if not self.openfast_exe.get() or not Path(self.openfast_exe.get()).exists(): messagebox.showerror("Error", "Please select a valid OpenFAST executable."); return
        selected_items = self.case_tree.selection()
        if not selected_items: messagebox.showwarning("Warning", "No test cases selected to run."); return
        if not messagebox.askyesno("Confirm", f"This will run {len(selected_items)} OpenFAST simulations. Continue?"): return
        self.progress_var.set(0); self.completed_cases = 0; self.total_cases_to_run = len(selected_items)
        while not self.job_queue.empty(): self.job_queue.get()
        for item_id in selected_items: self.job_queue.put(item_id)
        self.run_button.config(state='disabled'); threading.Thread(target=self.run_manager_thread, daemon=True).start()
        
    def run_manager_thread(self):
        num_workers = self.num_threads.get()
        self.message_queue.put(('log', f"Starting {self.total_cases_to_run} simulations with {num_workers} parallel workers..."))
        threads = [threading.Thread(target=self.run_worker, daemon=True) for _ in range(num_workers)]
        for t in threads: t.start()
        self.job_queue.join()
        self.message_queue.put(('log', "\n--- All selected tests completed. ---")); self.message_queue.put(('enable_run_button', None))
        
    def run_worker(self):
        while True:
            try: item_id = self.job_queue.get_nowait()
            except queue.Empty: return
            case_data = self.test_cases[item_id]
            self.message_queue.put(('tree_update', (item_id, 'Status', 'Running')))
            self.message_queue.put(('log', f"--- Running {case_data['name']} ---"))
            start_time = datetime.now()
            try:
                cmd = [self.openfast_exe.get(), case_data['fst_file']]
                process = subprocess.Popen(cmd, cwd=str(case_data['path']), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
                for line in iter(process.stdout.readline, ''): self.message_queue.put(('log', f"[{case_data['name']}] {line.strip()}"))
                process.wait()
                runtime, result, status = (datetime.now() - start_time).total_seconds(), "Success", "Completed"
                if process.returncode != 0: result, status = f"Error (code {process.returncode})", "Failed"
            except Exception as e:
                runtime, result, status = (datetime.now() - start_time).total_seconds(), f"Exception: {str(e)}", "Failed"
                self.message_queue.put(('log', f"FATAL ERROR in {case_data['name']}: {str(e)}"))
            self.message_queue.put(('tree_update', (item_id, 'Status', status)))
            self.message_queue.put(('tree_update', (item_id, 'Runtime', f"{runtime:.1f}s")))
            self.message_queue.put(('tree_update', (item_id, 'Result', result)))
            with self.progress_lock:
                self.completed_cases += 1
                self.message_queue.put(('progress', (self.completed_cases / self.total_cases_to_run) * 100))
            self.job_queue.task_done()
            
    def show_file_structure(self):
        if not self.file_structure: messagebox.showinfo("Info", "Run 'Discover Parameters' first."); return
        dialog = tk.Toplevel(self.root); dialog.title("Discovered File Structure"); dialog.geometry("800x600")
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Consolas', 10)); text.pack(fill='both', expand=True, padx=10, pady=10)
        text.insert('end', "OpenFAST File Structure:\n" + "="*60 + "\n\n")
        for file_type, file_info in sorted(self.file_structure.items()):
            path = file_info.get('path')
            if path:
                text.insert('end', f"{file_type}:\n", 'heading'); text.insert('end', f"  Path: {path}\n")
                text.insert('end', f"  Parameters Found: {len(self.discovered_parameters.get(file_type, {}))}\n\n")
        text.tag_config('heading', font=('Consolas', 11, 'bold'), foreground='darkblue'); text.config(state='disabled')
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
        
    def save_config(self):
        if not self.parameter_entries: messagebox.showinfo("Info", "No parameters to save."); return
        config = {'base_fst_path': self.base_fst_path.get(), 'output_dir': self.output_dir.get(), 'num_cases': self.num_cases.get(), 'distribution': self.distribution_var.get(), 'parameters': []}
        for p in self.parameter_entries:
            p_data = {'file_type': p['file_type'], 'param_name': p['param_name']}
            param_type = p['param_info']['type']
            p_data['csv_list'] = p['csv_var'].get()
            if param_type == 'float': p_data.update({'start': p['start_var'].get(), 'end': p['end_var'].get(), 'steps': p['steps_var'].get()})
            elif param_type == 'int': p_data.update({'int_mode': p['int_mode_var'].get(), 'start': p['start_var'].get(), 'end': p['end_var'].get(), 'steps': p['steps_var'].get(), 'int_list': p['list_var'].get()})
            elif param_type == 'bool': p_data.update({'bool_choice': p['bool_var'].get()})
            elif param_type == 'option': p_data.update({'options_list': p['options_var'].get()})
            config['parameters'].append(p_data)
        filename = filedialog.asksaveasfilename(title="Save Configuration", defaultextension=".json", filetypes=[("JSON config", "*.json")])
        if filename:
            with open(filename, 'w') as f: json.dump(config, f, indent=4)
            self.log(f"Configuration saved to: {filename}")
            
    def load_config(self):
        filename = filedialog.askopenfilename(title="Load Configuration", filetypes=[("JSON config", "*.json")])
        if not filename: return
        try:
            with open(filename, 'r') as f: config = json.load(f)
            self.base_fst_path.set(config.get('base_fst_path', '')); self.output_dir.set(config.get('output_dir', 'test_cases'))
            self.num_cases.set(config.get('num_cases', 10)); self.distribution_var.set(config.get('distribution', 'grid_search'))
            self.clear_parameters()
            if self.base_fst_path.get() and not self.discovered_parameters: self.log("Base FST found, running discovery..."); self.discover_parameters()
            if not self.discovered_parameters: messagebox.showwarning("Warning", "Run parameter discovery before loading parameters."); return
            for param_config in config.get('parameters', []):
                file_type, param_name = param_config.get('file_type'), param_config.get('param_name')
                if file_type and param_name and file_type in self.discovered_parameters and param_name in self.discovered_parameters[file_type]:
                    param_info = self.discovered_parameters[file_type][param_name]
                    self.add_parameter_with_info(file_type, param_name, param_info)
                    entry = self.parameter_entries[-1]
                    param_type = entry['param_info']['type']
                    
                    if 'csv_list' in param_config: entry['csv_var'].set(param_config.get('csv_list', ''))

                    if param_type == 'float':
                        entry['start_var'].set(param_config.get('start', 0)); entry['end_var'].set(param_config.get('end', 1)); entry['steps_var'].set(param_config.get('steps', 5))
                    elif param_type == 'int':
                        mode = param_config.get('int_mode', 'Range')
                        entry['int_mode_var'].set(mode)
                        entry['start_var'].set(param_config.get('start', 0)); entry['end_var'].set(param_config.get('end', 1)); entry['steps_var'].set(param_config.get('steps', 5))
                        entry['list_var'].set(param_config.get('int_list', '1,2,3'))
                    elif param_type == 'bool': entry['bool_var'].set(param_config.get('bool_choice', 'Vary (True & False)'))
                    elif param_type == 'option': entry['options_var'].set(param_config.get('options_list', ''))
                else: self.log(f"Warning: Could not find '{param_name}' in '{file_type}' from config.")
            self.log(f"Configuration loaded from: {filename}"); self.on_distribution_change()
        except Exception as e: messagebox.showerror("Error", f"Failed to load configuration: {str(e)}"); self.log(f"Error loading config: {e}")

    def clear_parameters(self):
        for entry in self.parameter_entries: entry['frame'].destroy()
        self.parameter_entries.clear(); self.update_total_cases()
        
    def log(self, message):
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"); self.log_text.see(tk.END); self.root.update_idletasks()
        
    def run_log_message(self, message):
        self.run_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n"); self.run_log.see(tk.END)
        
    def process_queue(self):
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                if msg_type == 'log': self.run_log.insert(tk.END, msg_data + '\n'); self.run_log.see(tk.END)
                elif msg_type == 'tree_update':
                    item_id, column, value = msg_data
                    if self.case_tree.exists(item_id): self.case_tree.set(item_id, column, value)
                elif msg_type == 'progress': self.progress_bar['value'] = msg_data
                elif msg_type == 'enable_run_button': self.run_button.config(state='normal')
        except queue.Empty: pass
        finally: self.root.after(100, self.process_queue)

def main():
    try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError): pass
    root = tk.Tk()
    app = OpenFASTTestCaseGUI(root)
    app.log("Welcome! Now with enhanced data type support."); app.log("=" * 60)
    app.log("1. Discover parameters. The GUI will detect their types (int, float, bool, option).")
    app.log("2. For 'grid_search', configure each parameter using its specific controls.")
    app.log("3. For 'csv_columnwise', provide comma-separated values for each parameter.")
    app.log("4. For sampling distributions, non-numeric parameters will be disabled.")
    app.log("5. Generate and run your test cases."); app.log("=" * 60)
    root.mainloop()

if __name__ == "__main__":
    main()