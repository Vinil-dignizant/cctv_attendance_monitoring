# gui/logs_view.py
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from app.db.crud import get_attendance_logs

class LogsView:
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.tree = ttk.Treeview(
            self.parent,
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
            self.parent,
            orient=tk.VERTICAL,
            command=self.tree.yview
        )
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Context menu
        self.context_menu = tk.Menu(self.parent, tearoff=0)
        self.context_menu.add_command(
            label="Refresh",
            command=self.load_data
        )
        self.tree.bind("<Button-3>", self.show_context_menu)
        
        # Load initial data
        self.load_data()

    def load_data(self):
        """Load attendance logs from database"""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Fetch and display logs
        logs = get_attendance_logs(limit=100)  # Implement this in crud.py
        for log in logs:
            self.tree.insert("", tk.END, values=(
                log.id,
                log.person_name,
                log.camera_id,
                f"{log.confidence_score:.2f}" if log.confidence_score else "N/A",
                log.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            ))

    def show_context_menu(self, event):
        """Display right-click context menu"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()