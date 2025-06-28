# gui/controls.py
import tkinter as tk
from tkinter import ttk, messagebox
from app.recognition import MultiCameraFaceRecognition

class SystemControls:
    def __init__(self, parent_frame, main_app):
        self.parent = parent_frame
        self.main_app = main_app
        self.frame = ttk.Frame(self.parent)
        self.recognition_system = None
        
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
            self.recognition_system = MultiCameraFaceRecognition("config/camera_config.yaml")
            self.main_app.recognition_system = self.recognition_system
            
            # Start in a separate thread
            import threading
            self.system_thread = threading.Thread(
                target=self.recognition_system.start,
                daemon=True
            )
            self.system_thread.start()
            
            self.status_var.set("System: Running")
            self.status_label.config(foreground="green")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror(
                "System Error",
                f"Failed to start recognition system:\n{str(e)}"
            )

    def stop_system(self):
        """Stop the face recognition system"""
        if self.recognition_system:
            self.recognition_system.stop()
            self.status_var.set("System: Stopped")
            self.status_label.config(foreground="red")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def export_logs(self):
        """Export attendance logs to CSV"""
        try:
            from app.db.crud import export_to_csv
            file_path = export_to_csv()  # Implement this in crud.py
            messagebox.showinfo(
                "Export Successful",
                f"Logs exported to:\n{file_path}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Failed",
                f"Error exporting logs:\n{str(e)}"
            )