import tkinter as tk
from tkinter import ttk, messagebox
from app.db.crud import get_attendance_logs
from app.db.models import AttendanceLog
from typing import List, Optional
from datetime import datetime
from .shared_state import get_system_status

class LogsView(ttk.Frame):
    def __init__(self, parent, refresh_interval: int = 5000):
        super().__init__(parent)
        self.refresh_interval = refresh_interval
        self.setup_ui()
        self.load_data()
        self.auto_refresh()

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Treeview with scrollbar
        self.tree_frame = ttk.Frame(self)
        self.tree_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.tree_frame.columnconfigure(0, weight=1)
        self.tree_frame.rowconfigure(0, weight=1)
        
        self.tree = ttk.Treeview(
            self.tree_frame,
            columns=("id", "name", "camera", "confidence", "timestamp"),
            show="headings",
            selectmode="browse"
        )
        
        # Configure columns
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="Person Name")
        self.tree.heading("camera", text="Camera ID")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("timestamp", text="Timestamp")
        
        self.tree.column("id", width=50, anchor=tk.CENTER)
        self.tree.column("name", width=200, anchor=tk.W)
        self.tree.column("camera", width=100, anchor=tk.CENTER)
        self.tree.column("confidence", width=100, anchor=tk.CENTER)
        self.tree.column("timestamp", width=200, anchor=tk.CENTER)
        
        # Add scrollbar
        # Add horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(
            self.tree_frame,
            orient=tk.HORIZONTAL,
            command=self.tree.xview
        )
        self.tree.configure(xscrollcommand=h_scrollbar.set)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.tree.grid(row=0, column=0, sticky="nsew")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Context menu
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(
            label="Refresh",
            command=self.load_data
        )
        self.tree.bind("<Button-3>", self.show_context_menu)

    def load_data(self):
        try:
            # Clear existing data
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Fetch and display logs
            logs = get_attendance_logs(limit=100)
            for log in logs:
                self.tree.insert("", tk.END, values=(
                    log.id,
                    log.person_name,
                    log.camera_id,
                    f"{log.confidence_score:.2f}" if log.confidence_score else "N/A",
                    log.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                ))
            
            self.status_var.set(f"Loaded {len(logs)} records")
        except Exception as e:
            self.status_var.set("Error loading data")
            messagebox.showerror("Error", f"Failed to load logs: {str(e)}")

    def auto_refresh(self):
        if get_system_status()['running']:
            self.load_data()
        self.after(self.refresh_interval, self.auto_refresh)

    def show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()