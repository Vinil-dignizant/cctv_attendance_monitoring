# gui/main_window.py
import tkinter as tk
from tkinter import ttk
from gui.camera_feed import CameraFeed
from gui.logs_view import LogsView
from gui.controls import SystemControls
from app.recognition import MultiCameraFaceRecognition

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
        
        # Tab system
        self.notebook = ttk.Notebook(self.main_frame)
        
        # Camera Tab
        self.camera_tab = ttk.Frame(self.notebook)
        self.camera_feed = CameraFeed(self.camera_tab)
        self.camera_feed.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Logs Tab
        self.logs_tab = ttk.Frame(self.notebook)
        self.logs_view = LogsView(self.logs_tab)
        self.logs_view.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add tabs
        self.notebook.add(self.camera_tab, text="Live Camera Feed")
        self.notebook.add(self.logs_tab, text="Attendance Logs")
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel
        self.control_panel = ttk.LabelFrame(self.main_frame, text="System Controls")
        self.system_controls = SystemControls(self.control_panel, self)
        self.control_panel.pack(fill=tk.X, pady=10)

    def protocols(self):
        """Handle window close events"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """Clean up resources on window close"""
        if self.recognition_system:
            self.recognition_system.stop()
        self.camera_feed.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()