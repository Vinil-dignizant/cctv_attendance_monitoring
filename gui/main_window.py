# gui/main_window.py
import time
import tkinter as tk
from tkinter import ttk
from .camera_feed import CameraFeed
from .logs_view import LogsView
from .controls import SystemControls
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x800")
        self.recognition_system = None
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TNotebook.Tab', font=('Helvetica', 10, 'bold'))
        self.style.configure('TButton', font=('Helvetica', 10))
        
        self.setup_ui()
        self.protocols()

    def setup_ui(self):
        """Initialize all UI components"""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control Panel (at top)
        self.control_panel = ttk.LabelFrame(self.main_frame, text="System Controls")
        self.control_panel.pack(fill=tk.X, pady=(0, 10))
        self.system_controls = SystemControls(self.control_panel, self)
        
        # Tab system
        self.notebook = ttk.Notebook(self.main_frame)
        
        # Camera Tab
        self.camera_tab = ttk.Frame(self.notebook)
        self.camera_feed = CameraFeed(self.camera_tab, self)
        
        # Logs Tab
        self.logs_tab = ttk.Frame(self.notebook)
        self.logs_view = LogsView(self.logs_tab, self)
        
        # Add tabs
        self.notebook.add(self.camera_tab, text="Live Camera Feed")
        self.notebook.add(self.logs_tab, text="Attendance Logs")
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        print("[INFO] GUI initialized successfully")

    def protocols(self):
        """Handle window close events"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """Clean up resources on window close"""
        print("[INFO] Shutting down application...")
        
        # Stop recognition system
        if self.recognition_system:
            try:
                self.system_controls.stop_system()
            except Exception as e:
                print(f"[WARNING] Error stopping recognition system: {e}")
        
        # Stop camera feed
        try:
            self.camera_feed.release()
        except Exception as e:
            print(f"[WARNING] Error releasing camera: {e}")
            
        # Stop auto refresh
        if hasattr(self, 'logs_view'):
            try:
                self.logs_view.stop_auto_refresh()
            except Exception as e:
                print(f"[WARNING] Error stopping auto refresh: {e}")
        
        self.root.destroy()

def main():
    """Main function to start the GUI application"""
    root = tk.Tk()
    app = MainApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] Application interrupted by user")
        app.on_close()

if __name__ == "__main__":
    main()