import tkinter as tk
from tkinter import ttk, messagebox
from .shared_state import update_system_status, get_system_status
from app.db.crud import export_to_csv
from app.db.database import get_db
from typing import Callable

class SystemControls(ttk.Frame):
    def __init__(self, parent, start_callback: Callable, stop_callback: Callable):
        super().__init__(parent)
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.setup_ui()

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        
        # Control buttons frame
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Start/Stop buttons
        self.start_btn = ttk.Button(
            btn_frame, 
            text="Start System",
            command=self.start_system
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            btn_frame,
            text="Stop System",
            command=self.stop_system,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Export button
        self.export_btn = ttk.Button(
            btn_frame,
            text="Export Logs",
            command=self.export_logs
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="System: Ready")
        self.status_label = ttk.Label(
            btn_frame,
            textvariable=self.status_var,
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Update status
        self.update_status()

    def start_system(self):
        try:
            self.start_callback()
            update_system_status(running=True)
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("System: Running")
            self.status_label.config(style='Status.Running.TLabel')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system: {str(e)}")
            update_system_status(running=False, error=str(e))
            self.status_var.set(f"Error: {str(e)}")
            self.status_label.config(style='Status.Error.TLabel')
            # Re-enable start button if failed
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def stop_system(self):
        try:
            self.stop_callback()
            update_system_status(running=False)
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.status_var.set("System: Stopped")
            self.status_label.config(style='Status.Stopped.TLabel')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop system: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self.status_label.config(style='Status.Error.TLabel')
            # Ensure buttons are in correct state even if error occurs
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def export_logs(self):
        try:
            db = next(get_db())
            file_path = export_to_csv(db)
            messagebox.showinfo(
                "Export Successful", 
                f"Logs exported to:\n{file_path}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Failed", 
                f"Error exporting logs:\n{str(e)}"
            )

    def update_status(self):
        status = get_system_status()
        if status['running']:
            self.status_var.set("System: Running")
            self.status_label.config(style='Status.Running.TLabel')
        else:
            self.status_var.set("System: Stopped")
            self.status_label.config(style='Status.Stopped.TLabel')
        
        self.after(1000, self.update_status)