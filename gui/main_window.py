# gui/main_window.py
import tkinter as tk
from tkinter import ttk
from gui.components.config_ui import ConfigView
from gui.components.controls import SystemControls
from gui.components.camera_feed import CameraFeed
from gui.components.logs_view import LogsView
from gui.components.summary_view import SummaryView
from gui.styles import configure_styles
from app.multi_camera_face_recognition import MultiCameraFaceRecognition
from gui.components.person_management import PersonManagementView
from gui.components.camera_management import CameraManagementView
from app.db.crud import init_db
from typing import List, Dict
import threading
import yaml
import os


class MainApp:
    def __init__(self, root, config_path: str = "config/camera_config.yaml"):
        self.root = root
        self.config_path = config_path
        
        self.recognition_system = None
        self.system_thread = None
        # self.camera_configs = self.load_camera_config()
        self.camera_configs = self.load_camera_config_from_db()  # Changed to load from DB
        
        # Configure styles
        configure_styles()
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        
        # Initialize database
        init_db()

    def load_camera_config(self) -> List[Dict]:
        """Load camera configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Filter only enabled cameras and add camera_name if not present
            enabled_cameras = []
            for cam in config['cameras']:
                if cam.get('enabled', True):
                    if 'camera_name' not in cam:
                        cam['camera_name'] = f"{cam['camera_id']} ({cam.get('event_type', 'N/A')})"
                    enabled_cameras.append(cam)
            return enabled_cameras
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return []
        
    def load_camera_config_from_db(self) -> List[Dict]:
        """Load camera configuration from database"""
        try:
            from app.db.crud import get_all_cameras
            from app.db.database import get_db
            
            db = next(get_db())
            cameras = get_all_cameras(db)
            
            enabled_cameras = []
            for camera in cameras:
                if camera.is_enabled:
                    enabled_cameras.append({
                        'camera_id': camera.camera_id,
                        'camera_name': camera.camera_name,
                        'url': camera.url,
                        'enabled': camera.is_enabled,
                        'location': camera.location,
                        'event_type': camera.event_type
                    })
            return enabled_cameras
        except Exception as e:
            print(f"[ERROR] Failed to load camera config from database: {e}")
            # Fallback to YAML if database fails
            return self.load_camera_config_from_yaml()

    def load_camera_config_from_yaml(self) -> List[Dict]:
        """Fallback to load camera configuration from YAML file"""
        try:
            import yaml
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            enabled_cameras = []
            for cam in config['cameras']:
                if cam.get('enabled', True):
                    if 'camera_name' not in cam:
                        cam['camera_name'] = f"{cam['camera_id']} ({cam.get('event_type', 'N/A')})"
                    enabled_cameras.append(cam)
            return enabled_cameras
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return []

    def setup_ui(self):
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x800")
        
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)  # Notebook gets most space
        
        # Control panel
        self.control_panel = ttk.LabelFrame(self.main_frame, text="System Controls")
        self.control_panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.system_controls = SystemControls(
            self.control_panel,
            start_callback=self.start_system,
            stop_callback=self.stop_system
        )
        self.system_controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Tab system
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        
        # Camera Tab
        self.setup_camera_tab()
        
        # Logs Tab
        self.logs_tab = LogsView(self.notebook)
        self.notebook.add(self.logs_tab, text="Attendance Logs")
        
        # Summary Tab
        self.summary_tab = SummaryView(self.notebook)
        self.notebook.add(self.summary_tab, text="Daily Summaries")

        # Add Person Management Tab
        self.person_tab = PersonManagementView(self.notebook)
        self.notebook.add(self.person_tab, text="Person Management")

        # Add Camera Management Tab
        self.camera_mgmt_tab = CameraManagementView(self.notebook)
        self.notebook.add(self.camera_mgmt_tab, text="Camera Management")

    # setup database config
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        
        # Configuration menu
        config_menu = tk.Menu(menubar, tearoff=0)
        config_menu.add_command(label="Database Settings", command=self.show_config)
        menubar.add_cascade(label="Settings", menu=config_menu)
        
        self.root.config(menu=menubar)

    def show_config(self):
        """Show configuration window"""
        config_window = tk.Toplevel(self.root)
        config_window.title("Database Configuration")
        ConfigView(config_window).pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


    def setup_camera_tab(self):
        self.camera_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_tab, text="Live Camera Feed")
        
        # Create a single camera feed with selection dropdown
        self.camera_feed = CameraFeed(self.camera_tab, self.camera_configs)
        self.camera_feed.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # # Start the system if not already running
        # if self.recognition_system is None:
        #     self.start_system()

    def start_system(self):
        """Start the recognition system in a separate thread"""
        if self.recognition_system is None:
            self.recognition_system = MultiCameraFaceRecognition(self.config_path)
            self.system_thread = threading.Thread(
                target=self.recognition_system.start,
                daemon=True
            )
            self.system_thread.start()

    def stop_system(self):
        """Stop the recognition system"""
        if self.recognition_system:
            print("[DEBUG] Stopping recognition system...")
            self.recognition_system.stop()
            
            # Wait for thread to finish with timeout
            if self.system_thread:
                self.system_thread.join(timeout=3)  # Increased timeout
                if self.system_thread.is_alive():
                    print("[WARNING] Recognition thread did not stop gracefully")
            
            self.recognition_system = None
            self.system_thread = None
            print("[DEBUG] Recognition system stopped")


    def on_close(self):
        """Clean up resources on window close"""
        self.stop_system()
        self.root.destroy()

def run_gui():
    root = tk.Tk()
    app = MainApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    run_gui()