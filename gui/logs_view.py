# gui/logs_view.py
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import sys
import os
import logging
from typing import List

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.db.crud import get_attendance_logs
from app.db.models import AttendanceLog

class LogsView:
    def __init__(self, parent_frame, main_app):
        self.parent = parent_frame
        self.main_app = main_app
        self.auto_refresh_enabled = False
        
        # Create main frame for logs view
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(
            self.frame,
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
        self.tree.column("name", width=150, anchor=tk.W)
        self.tree.column("camera", width=100, anchor=tk.CENTER)
        self.tree.column("confidence", width=80, anchor=tk.CENTER)
        self.tree.column("timestamp", width=150, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            self.frame,
            orient=tk.VERTICAL,
            command=self.tree.yview
        )
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Context menu
        self.context_menu = tk.Menu(self.frame, tearoff=0)
        self.context_menu.add_command(
            label="Refresh",
            command=self.load_data
        )
        self.tree.bind("<Button-3>", self.show_context_menu)
        
        # Status label
        self.status_var = tk.StringVar(value="Logs: Ready")
        self.status_label = ttk.Label(
            self.frame,
            textvariable=self.status_var,
            foreground="green"
        )
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Load initial data
        self.load_data()

    def load_data(self):
        """Load attendance logs from database"""
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
            self.status_var.set(f"Logs: Loaded {len(logs)} records")
            print(f"[DEBUG] Loaded {len(logs)} log records")
        except Exception as e:
            self.status_var.set("Logs: Error loading data")
            print(f"[ERROR] Failed to load logs: {str(e)}")
            messagebox.showerror("Database Error", f"Failed to load logs: {str(e)}")

    def auto_refresh(self, interval=5000):
        """Auto-refresh logs at regular intervals"""
        if not self.auto_refresh_enabled:
            return
            
        self.load_data()
        if (hasattr(self.main_app, 'recognition_system') and 
            self.main_app.recognition_system and
            self.auto_refresh_enabled):
            self.frame.after(interval, lambda: self.auto_refresh(interval))

    def start_auto_refresh(self):
        """Start auto-refresh"""
        self.auto_refresh_enabled = True
        self.auto_refresh()

    def stop_auto_refresh(self):
        """Stop auto-refresh"""
        self.auto_refresh_enabled = False

    def show_context_menu(self, event):
        """Display right-click context menu"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()