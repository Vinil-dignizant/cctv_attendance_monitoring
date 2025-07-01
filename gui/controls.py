# gui/controls.py
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.multi_camera_face_recognition import MultiCameraFaceRecognition

class SystemControls:
    def __init__(self, parent_frame, main_app):
        self.parent = parent_frame
        self.main_app = main_app
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.recognition_system = None
        self.system_thread = None
        
        # Control buttons
        self.start_btn = ttk.Button(
            self.frame,
            text="Start Recognition",
            command=self.start_system
        )
        self.stop_btn = ttk.Button(
            self.frame,
            text="Stop Recognition",
            command=self.stop_system,
            state=tk.DISABLED
        )
        self.export_btn = ttk.Button(
            self.frame,
            text="Export Logs",
            command=self.export_logs
        )
        
        # Layout
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.status_var = tk.StringVar(value="System: Ready")
        self.status_label = ttk.Label(
            self.frame,
            textvariable=self.status_var,
            foreground="green"
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def start_system(self):
        """Start the face recognition system"""
        try:
            # Check if config file exists
            config_path = "config/camera_config.yaml"
            if not os.path.exists(config_path):
                messagebox.showerror("Error", f"Configuration file not found: {config_path}")
                return
                
            self.recognition_system = MultiCameraFaceRecognition(config_path)
            self.main_app.recognition_system = self.recognition_system
            
            # Start in a separate thread
            self.system_thread = threading.Thread(
                target=self.recognition_system.start,
                daemon=True
            )
            self.system_thread.start()
            
            self.status_var.set("System: Running")
            self.status_label.config(foreground="green")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Enable auto-refresh in logs view
            if hasattr(self.main_app, 'logs_view'):
                self.main_app.logs_view.start_auto_refresh()
            
            print("[INFO] Recognition system started successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to start recognition system: {str(e)}")
            messagebox.showerror(
                "System Error",
                f"Failed to start recognition system:\n{str(e)}"
            )

    def stop_system(self):
        """Stop the face recognition system"""
        try:
            if self.recognition_system:
                # Stop the recognition system gracefully
                self.recognition_system = None
                self.main_app.recognition_system = None
                
            # Stop auto-refresh in logs view
            if hasattr(self.main_app, 'logs_view'):
                self.main_app.logs_view.stop_auto_refresh()
                
            self.status_var.set("System: Stopped")
            self.status_label.config(foreground="red")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            print("[INFO] Recognition system stopped")
            
        except Exception as e:
            print(f"[ERROR] Error stopping recognition system: {str(e)}")

    def export_logs(self):
        """Export attendance logs to CSV"""
        try:
            from app.db.crud import export_to_csv
            file_path = export_to_csv()
            messagebox.showinfo(
                "Export Successful",
                f"Logs exported to:\n{file_path}"
            )
        except Exception as e:
            print(f"[ERROR] Export failed: {str(e)}")
            messagebox.showerror(
                "Export Failed",
                f"Error exporting logs:\n{str(e)}"
            )