import tkinter as tk
from tkinter import ttk, messagebox
from app.db.crud import get_attendance_logs
from app.db.models import DailySummary
from typing import List, Optional
from datetime import datetime, timedelta
from .shared_state import get_system_status
from app.db.database import get_db
from sqlalchemy.orm import Session

class SummaryView(ttk.Frame):
    def __init__(self, parent, refresh_interval: int = 10000):
        super().__init__(parent)
        self.refresh_interval = refresh_interval
        self.setup_ui()  
        self.load_data()
        self.auto_refresh()

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)  # Treeview gets most space
        
        # Filter controls
        filter_frame = ttk.Frame(self)
        filter_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Filter by Name:").pack(side=tk.LEFT, padx=5)
        
        self.name_filter = tk.StringVar()
        self.name_entry = ttk.Entry(filter_frame, textvariable=self.name_filter)
        self.name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.filter_btn = ttk.Button(
            filter_frame,
            text="Apply Filter",
            command=self.load_data
        )
        self.filter_btn.pack(side=tk.LEFT, padx=5)
        
        # Treeview with scrollbar
        self.tree_frame = ttk.Frame(self)
        self.tree_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.tree_frame.columnconfigure(0, weight=1)
        self.tree_frame.rowconfigure(0, weight=1)
        
        self.tree = ttk.Treeview(
            self.tree_frame,
            columns=("name", "date", "camera", "first_login", "last_logout", "hours", "logins", "logouts"),
            show="headings"
        )
        
        # Configure columns
        self.tree.heading("name", text="Person Name")
        self.tree.heading("date", text="Date")
        self.tree.heading("camera", text="Camera")
        self.tree.heading("first_login", text="First Login")
        self.tree.heading("last_logout", text="Last Logout")
        self.tree.heading("hours", text="Working Hours")
        self.tree.heading("logins", text="Total Logins")
        self.tree.heading("logouts", text="Total Logouts")
        
        # Configure columns with more width
        self.tree.column("name", width=200)  # Increased width
        self.tree.column("date", width=120)  # Increased width
        self.tree.column("camera", width=150)  # Increased width
        self.tree.column("first_login", width=150)  # Increased width
        self.tree.column("last_logout", width=150)  # Increased width
        self.tree.column("hours", width=120)  # Increased width
        self.tree.column("logins", width=100, anchor=tk.CENTER)  # Increased width
        self.tree.column("logouts", width=100, anchor=tk.CENTER)  # Increased width
        
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
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

    def load_data(self):
        try:
            # Clear existing data
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Get filter value
            name_filter = self.name_filter.get().strip()
            
            # Fetch data from database
            db = next(get_db())
            query = db.query(DailySummary).order_by(
                DailySummary.date.desc(),
                DailySummary.person_name
            )
            
            if name_filter:
                query = query.filter(DailySummary.person_name.ilike(f"%{name_filter}%"))
            
            summaries = query.limit(100).all()
            
            # Add data to treeview
            for summary in summaries:
                # Format working hours
                hours = ""
                if summary.working_hours:
                    total_seconds = summary.working_hours.total_seconds()
                    hours = str(timedelta(seconds=total_seconds)).split('.')[0]
                
                self.tree.insert("", tk.END, values=(
                    summary.person_name,
                    summary.date.strftime("%Y-%m-%d"),
                    summary.camera_id,
                    summary.first_login.strftime("%H:%M:%S") if summary.first_login else "",
                    summary.last_logout.strftime("%H:%M:%S") if summary.last_logout else "",
                    hours,
                    summary.total_logins,
                    summary.total_logouts
                ))
            
            self.status_var.set(f"Loaded {len(summaries)} records")
        except Exception as e:
            self.status_var.set("Error loading data")
            messagebox.showerror("Error", f"Failed to load summaries: {str(e)}")

    def auto_refresh(self):
        if get_system_status()['running']:
            self.load_data()
        self.after(self.refresh_interval, self.auto_refresh)