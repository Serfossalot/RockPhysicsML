"""Milestone 2 desktop interface for the existing Rock Physics ML pipeline."""
from __future__ import annotations

import queue
import os
import sys
import shutil
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from real_wells import load_real_well, evaluate_real_well, real_well_diagnostics
from rockphysics import (DataPreparationController, EvaluationController,
                         ModelTrainingController, PredictionController,
                         VisualizationController, WellData)

APP_VERSION = '1.0.2'

# Frozen builds: the executable directory is the installed application directory.
# Development: the directory containing gui.py is the application directory.
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
# PyInstaller --onefile extracts bundled resources to _MEIPASS.
BUNDLE_DIR = Path(getattr(sys, '_MEIPASS', APP_DIR))
# Installed sample data is shared and read-only; never write into Program Files.
DATA_DIR = APP_DIR / 'data'
RESULTS_DIR = Path.home() / 'Documents'
if not RESULTS_DIR.is_dir():
    RESULTS_DIR = Path.home()
DEFAULT_DATA = DATA_DIR / 'synthetic_wells_combined.csv'


class RockPhysicsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        default_data = DEFAULT_DATA
        self.title(f'Rock Physics ML v{APP_VERSION} — Rock Physics Modeling')
        icon_file = BUNDLE_DIR / 'oil-pumpjack-svgrepo-com.ico'
        if icon_file.exists():
            try:
                self.iconbitmap(default=str(icon_file))
            except tk.TclError:
                pass
        self.geometry('1120x760')
        self.minsize(900, 620)
        self.data = None
        self.prepared = None
        self.models = None
        self.predictions = {}
        self.metrics = {}
        self.real_frame = None
        self.real_predictions = None
        self.real_metrics = None
        self.real_path = tk.StringVar(value=str(DATA_DIR / 'real' / 'BT2.npy'))
        self.events = queue.Queue()
        self.busy = False
        self.data_path = tk.StringVar(value=str(default_data))
        self.train_pct = tk.StringVar(value='70')
        self.test_pct = tk.StringVar(value='20')
        self.blind_pct = tk.StringVar(value='10')
        self.status = tk.StringVar(value='Ready. Load a well-data CSV to begin.')
        self._build()
        self.after(100, self._poll)
        self.after(150, self._show_splash)

    def _show_splash(self):
        """Show branding briefly without blocking the Tk event loop."""
        self.withdraw()
        splash = tk.Toplevel(self)
        splash.overrideredirect(True)
        splash.configure(bg='#1264AD')
        width, height = 480, 290
        x = (splash.winfo_screenwidth() - width) // 2
        y = (splash.winfo_screenheight() - height) // 2
        splash.geometry(f'{width}x{height}+{x}+{y}')
        icon_file = BUNDLE_DIR / 'oil-pumpjack-svgrepo-com.ico'
        if icon_file.exists():
            try:
                from PIL import Image, ImageTk
                with Image.open(icon_file) as img:
                    img = img.convert('RGBA')
                    img.thumbnail((100, 100))
                    splash.pumpjack_image = ImageTk.PhotoImage(img)
                tk.Label(splash, image=splash.pumpjack_image, bg='white', padx=14, pady=10).pack(pady=(20, 8))
            except (ImportError, OSError, tk.TclError):
                pass
        tk.Label(splash, text='Rock Physics ML', font=('Segoe UI', 24, 'bold'),
                 fg='white', bg='#1264AD').pack(pady=(5, 12))
        tk.Label(splash, text=f'Version {APP_VERSION}', font=('Segoe UI', 11),
                 fg='white', bg='#1264AD').pack(pady=(0, 6))
        tk.Label(splash, text='Written by: Joel R. Serfoss', font=('Segoe UI', 12),
                 fg='white', bg='#1264AD').pack()
        splash.lift()
        splash.after(1700, lambda: (splash.destroy(), self.deiconify()))

    def _save_plot_as(self, predictions, default_name, label=None):
        path = filedialog.asksaveasfilename(
            parent=self, title='Save plot as', initialdir=str(RESULTS_DIR),
            initialfile=default_name, defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('PDF document', '*.pdf'),
                       ('JPEG image', '*.jpg'), ('All files', '*.*')])
        if not path:
            return
        try:
            if label is None:
                VisualizationController.plot(predictions, Path(path))
            else:
                VisualizationController.plot(predictions, Path(path), label=label)
            self._log(f'Saved plot: {path}')
            self.status.set(f'Plot saved: {path}')
        except Exception as exc:
            messagebox.showerror('Plot export failed', str(exc))

    def _build(self):
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        shell = ttk.Frame(self, padding=14)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='Rock Physics Modeling Using Machine Learning',
                  font=('Segoe UI', 17, 'bold')).pack(anchor='w')
        ttk.Label(shell, text='Synthetic training • Independent real-well evaluation • SVR / Random Forest / MLP',
                  foreground='#555').pack(anchor='w', pady=(3, 12))

        data_box = ttk.LabelFrame(shell, text='1. Load and inspect well data', padding=10)
        data_box.pack(fill='x', pady=(0, 10))
        path_row = ttk.Frame(data_box)
        path_row.pack(fill='x')
        ttk.Entry(path_row, textvariable=self.data_path).pack(side='left', fill='x', expand=True)
        ttk.Button(path_row, text='Browse CSV…', command=self.browse).pack(side='left', padx=6)
        self.load_button = ttk.Button(path_row, text='Load and validate', command=self.load)
        self.load_button.pack(side='left')
        self.summary = ttk.Label(data_box, text='No data loaded.')
        self.summary.pack(anchor='w', pady=(8, 0))

        prep = ttk.LabelFrame(shell, text='2. Configure well-level split and train', padding=10)
        prep.pack(fill='x', pady=(0, 10))
        split_row = ttk.Frame(prep)
        split_row.pack(fill='x')
        for title, var in [('Training %', self.train_pct), ('Testing %', self.test_pct),
                           ('Blind validation %', self.blind_pct)]:
            ttk.Label(split_row, text=title).pack(side='left', padx=(0, 5))
            ttk.Entry(split_row, textvariable=var, width=7).pack(side='left', padx=(0, 18))
        self.run_button = ttk.Button(split_row, text='Train and evaluate', command=self.run_training)
        self.run_button.pack(side='right')
        ttk.Label(prep, text='Entire wells are held out; samples from the same well never cross splits.',
                  foreground='#555').pack(anchor='w', pady=(8, 0))

        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill='both', expand=True)
        preview_tab = ttk.Frame(self.tabs, padding=8)
        results_tab = ttk.Frame(self.tabs, padding=8)
        log_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(preview_tab, text='Data preview')
        self.tabs.add(results_tab, text='Model performance')
        self.tabs.add(log_tab, text='Activity log')
        real_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(real_tab, text='Real well evaluation')
        self._build_real_tab(real_tab)

        self.preview = self._table(preview_tab)
        actions = ttk.Frame(results_tab)
        actions.pack(fill='x', pady=(0, 8))
        ttk.Label(actions, text='Dataset:').pack(side='left')
        self.split_select = ttk.Combobox(actions, values=['Blind validation', 'Test'], state='readonly', width=18)
        self.split_select.set('Blind validation')
        self.split_select.pack(side='left', padx=6)
        self.split_select.bind('<<ComboboxSelected>>', lambda _event: self.show_metrics())
        self.graph_button = ttk.Button(actions, text='View comparison plot', command=self.view_plot)
        self.graph_button.pack(side='right', padx=(6, 0))
        ttk.Button(actions, text='Save plot as…', command=self.save_plot).pack(side='right', padx=6)
        self.export_button = ttk.Button(actions, text='Export CSV and PNG…', command=self.export)
        self.export_button.pack(side='right')
        self.metrics_table = self._table(results_tab)
        ttk.Label(results_tab, text='R², RMSE, MAE, and MAPE are computed on unseen wells. '
                  'MAPE is mean absolute percentage error, not classification accuracy.',
                  foreground='#555', wraplength=950).pack(anchor='w', pady=(8, 0))
        self.log_text = tk.Text(log_tab, height=14, state='disabled', wrap='word', font=('Consolas', 10))
        self.log_text.pack(fill='both', expand=True)
        ttk.Label(shell, textvariable=self.status, anchor='w').pack(fill='x', pady=(8, 0))
        self._set_busy(False)

    @staticmethod
    def _table(parent):
        container = ttk.Frame(parent)
        container.pack(fill='both', expand=True)
        tree = ttk.Treeview(container, show='headings')
        ys = ttk.Scrollbar(container, orient='vertical', command=tree.yview)
        xs = ttk.Scrollbar(container, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        tree.grid(row=0, column=0, sticky='nsew')
        ys.grid(row=0, column=1, sticky='ns')
        xs.grid(row=1, column=0, sticky='ew')
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        return tree

    @staticmethod
    def _fill_table(tree, frame, limit=100):
        tree.delete(*tree.get_children())
        columns = [str(c) for c in frame.columns]
        tree['columns'] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=max(100, min(180, len(col)*12)), anchor='center', stretch=False)
        for row in frame.head(limit).itertuples(index=False, name=None):
            vals = [f'{x:.5f}' if isinstance(x, float) else str(x) for x in row]
            tree.insert('', 'end', values=vals)

    def _log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def _set_busy(self, busy):
        self.busy = busy
        state = 'disabled' if busy else 'normal'
        for button in (self.load_button, self.run_button, self.graph_button, self.export_button):
            button.configure(state=state)

    def browse(self):
        path = filedialog.askopenfilename(title='Select synthetic well data', initialdir=str(DATA_DIR),
                                          filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if path:
            self.data_path.set(path)
            self.load()

    def load(self):
        if self.busy:
            return
        try:
            data = WellData.load(self.data_path.get().strip())
            data.validate()
            self.data = data
            self.prepared = None
            self.models = None
            self.predictions.clear()
            self.metrics.clear()
            self.real_predictions = None
            self.real_metrics = None
            self._fill_table(self.preview, data.frame)
            self._fill_table(self.metrics_table, pd.DataFrame())
            self.summary.configure(text=f'{len(data.frame):,} records across '
                                        f'{data.frame.WELL.nunique()} wells; '
                                        f'{len(data.frame.columns)} columns. Preview shows first 100 records.')
            self.status.set('Data validated. Configure splits and click Train and evaluate.')
            self._log(f'Loaded and validated: {self.data_path.get()}')
        except Exception as exc:
            self.status.set('Could not load dataset.')
            messagebox.showerror('Invalid well data', str(exc))

    def run_training(self):
        if self.busy:
            return
        if self.data is None:
            messagebox.showinfo('Load data first', 'Load and validate a well-data CSV first.')
            return
        try:
            percentages = [float(v.get()) for v in (self.train_pct, self.test_pct, self.blind_pct)]
            if any(not 0 < v < 100 for v in percentages) or abs(sum(percentages)-100) > 0.00001:
                raise ValueError('Training, testing, and blind percentages must be positive and total 100%.')
            train, test = percentages[0]/100, percentages[1]/100
            prepared = DataPreparationController.prepare(self.data, train, test)
        except Exception as exc:
            messagebox.showerror('Invalid split', str(exc))
            return
        self._set_busy(True)
        self.status.set('Training SVR, Random Forest, and MLP… Please wait.')
        self._log('Training started. The interface remains responsive.')
        threading.Thread(target=self._train_worker, args=(prepared,), daemon=True).start()

    def _train_worker(self, prepared):
        try:
            models = ModelTrainingController().train(prepared)
            predictions = {}
            metrics = {}
            for name, frame in [('test', prepared.test), ('blind', prepared.blind)]:
                predictions[name] = PredictionController.predict(models, frame)
                metrics[name] = EvaluationController.evaluate(predictions[name])
            self.events.put(('success', (prepared, models, predictions, metrics)))
        except Exception:
            self.events.put(('error', traceback.format_exc()))

    def _poll(self):
        try:
            while True:
                event, payload = self.events.get_nowait()
                self._set_busy(False)
                if event == 'error':
                    self._log(payload)
                    self.status.set('Training failed; see activity log.')
                    messagebox.showerror('Training failed', payload.splitlines()[-1])
                else:
                    self.prepared, self.models, self.predictions, self.metrics = payload
                    self.real_predictions = None
                    self.real_metrics = None
                    self.show_metrics()
                    counts = [self.prepared.train.WELL.nunique(), self.prepared.test.WELL.nunique(),
                              self.prepared.blind.WELL.nunique()]
                    self._log(f'Training completed. Well counts: train={counts[0]}, test={counts[1]}, blind={counts[2]}')
                    self.status.set('Training complete. Review metrics, view plots, or export results.')
                    self.tabs.select(1)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def show_metrics(self):
        split = 'blind' if self.split_select.get() == 'Blind validation' else 'test'
        if split in self.metrics:
            self._fill_table(self.metrics_table, self.metrics[split])

    def view_plot(self):
        if not self.predictions:
            messagebox.showinfo('No predictions', 'Train the models first.')
            return
        import tempfile
        import os
        split = 'blind' if self.split_select.get() == 'Blind validation' else 'test'
        path = Path(tempfile.gettempdir()) / f'rockphysics_{split}_plot.png'
        VisualizationController.plot(self.predictions[split], path)
        try:
            os.startfile(str(path))  # Windows: open in the default image viewer
        except AttributeError:
            messagebox.showinfo('Plot saved', f'Open the plot at:\n{path}')
        self._log(f'Generated {split} comparison plot: {path}')

    def save_plot(self):
        if not self.predictions:
            messagebox.showinfo('No predictions', 'Train the models first.')
            return
        split = 'blind' if self.split_select.get() == 'Blind validation' else 'test'
        self._save_plot_as(self.predictions[split], f'{split}_plot.png')

    def export(self):
        if not self.predictions:
            messagebox.showinfo('No results', 'Train the models first.')
            return
        directory = filedialog.askdirectory(title='Select results export folder', initialdir=str(RESULTS_DIR))
        if not directory:
            return
        try:
            out = Path(directory)
            for split in ('test', 'blind'):
                self.predictions[split].to_csv(out / f'{split}_predictions.csv', index=False)
                self.metrics[split].to_csv(out / f'{split}_metrics.csv', index=False)
                VisualizationController.plot(self.predictions[split], out / f'{split}_plot.png')
            self._log(f'Exported six result files to {out}')
            self.status.set(f'Export complete: {out}')
            messagebox.showinfo('Export complete', f'Saved six files to:\n{out}')
        except Exception as exc:
            messagebox.showerror('Export failed', str(exc))

    def _build_real_tab(self, parent):
        ttk.Label(parent, text='Evaluate an external real well using models trained only on synthetic data.',
                  wraplength=900).pack(anchor='w', pady=(0, 8))
        bar = ttk.Frame(parent)
        bar.pack(fill='x', pady=(0, 8))
        ttk.Entry(bar, textvariable=self.real_path).pack(side='left', fill='x', expand=True)
        ttk.Button(bar, text='Browse .npy…', command=self.browse_real).pack(side='left', padx=5)
        ttk.Button(bar, text='Load real well', command=self.load_real).pack(side='left')
        self.real_summary = ttk.Label(parent, text='No real well loaded. Train synthetic models first.')
        self.real_summary.pack(anchor='w', pady=(0, 8))
        real_actions = ttk.Frame(parent)
        real_actions.pack(fill='x', pady=(0, 8))
        ttk.Button(real_actions, text='Predict and evaluate', command=self.run_real).pack(side='left')
        ttk.Button(real_actions, text='View real-well plot', command=self.view_real_plot).pack(side='left', padx=6)
        ttk.Button(real_actions, text='Save real-well plot as…', command=self.save_real_plot).pack(side='left', padx=6)
        ttk.Button(real_actions, text='Export real-well results…', command=self.export_real).pack(side='left')
        self.real_table = self._table(parent)
        ttk.Label(parent, text='Real-well metrics are out-of-domain evaluation, not synthetic blind validation. '
                  'Units and input distributions must be checked before scientific interpretation.',
                  wraplength=950).pack(anchor='w', pady=(8, 0))

    def browse_real(self):
        path = filedialog.askopenfilename(title='Select real well', initialdir=str(DATA_DIR / 'real'), filetypes=[('NumPy arrays', '*.npy')])
        if path:
            self.real_path.set(path)
            self.load_real()

    def load_real(self):
        try:
            frame, removed = load_real_well(self.real_path.get())
            self.real_frame = frame
            self.real_predictions = None
            self.real_metrics = None
            self._fill_table(self.real_table, frame)
            issues = real_well_diagnostics(frame)
            self.real_summary.configure(text=f'{frame.WELL.iloc[0]}: {len(frame):,} valid records; '
                                        f'{removed} invalid rows excluded. ' +
                                        ('Warnings: ' + '; '.join(issues) if issues else 'Basic range checks passed.'))
            self._log(f'Loaded real well {self.real_path.get()}; valid={len(frame)}, excluded={removed}')
            for issue in issues: self._log('Real well warning: ' + issue)
        except Exception as exc:
            messagebox.showerror('Real well load failed', str(exc))

    def run_real(self):
        if self.models is None:
            messagebox.showinfo('Train first', 'Train models using the synthetic CSV first.')
            return
        if self.real_frame is None:
            messagebox.showinfo('Load real well', 'Load a real-well .npy file first.')
            return
        try:
            self.real_predictions, self.real_metrics = evaluate_real_well(self.models, self.real_frame)
            self._fill_table(self.real_table, self.real_metrics)
            self.status.set(f'Real-well evaluation complete: {self.real_frame.WELL.iloc[0]}')
            self._log(f'Evaluated real well {self.real_frame.WELL.iloc[0]} without fitting on real data.')
        except Exception as exc:
            messagebox.showerror('Real-well evaluation failed', str(exc))

    def view_real_plot(self):
        if self.real_predictions is None:
            messagebox.showinfo('No predictions', 'Run real-well evaluation first.')
            return
        import tempfile, os
        path = Path(tempfile.gettempdir()) / 'rockphysics_real_well_plot.png'
        VisualizationController.plot(self.real_predictions, path, label="real well")
        try: os.startfile(str(path))
        except AttributeError: messagebox.showinfo('Plot saved', str(path))

    def save_real_plot(self):
        if self.real_predictions is None:
            messagebox.showinfo('No predictions', 'Run real-well evaluation first.')
            return
        name = str(self.real_frame.WELL.iloc[0])
        self._save_plot_as(self.real_predictions, f'{name}_real_plot.png', label='real well')

    def export_real(self):
        if self.real_predictions is None:
            messagebox.showinfo('No predictions', 'Run real-well evaluation first.')
            return
        directory = filedialog.askdirectory(title='Select real-well export folder', initialdir=str(RESULTS_DIR))
        if not directory: return
        try:
            out = Path(directory)
            name = self.real_frame.WELL.iloc[0]
            self.real_predictions.to_csv(out / f'{name}_real_predictions.csv', index=False)
            self.real_metrics.to_csv(out / f'{name}_real_metrics.csv', index=False)
            VisualizationController.plot(self.real_predictions, out / f'{name}_real_plot.png', label='real well')
            self._log(f'Exported real well {name} to {out}')
            messagebox.showinfo('Export complete', f'Saved three files for {name} to:\n{out}')
        except Exception as exc:
            messagebox.showerror('Real-well export failed', str(exc))


if __name__ == '__main__':
    RockPhysicsApp().mainloop()
