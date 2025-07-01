# gui/camera_feed.py
import cv2
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from queue import Queue
import numpy as np
import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.config_loader import load_camera_config

class CameraFeed:
    def __init__(self, parent_frame, main_app):
        self.parent = parent_frame
        self.main_app = main_app
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.video_queue = Queue(maxsize=1)
        self.running = False
        self.cap = None
        self.current_camera = None
        self.update_job = None
        
        # Video display
        self.video_label = tk.Label(self.frame, text="Select a camera and start the feed", 
                                   font=("Arial", 14), bg="lightgray")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera selection frame
        self.control_frame = ttk.Frame(self.frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera selection
        ttk.Label(self.control_frame, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.StringVar()
        self.camera_selector = ttk.Combobox(
            self.control_frame, 
            textvariable=self.camera_var,
            state='readonly',
            width=30
        )
        self.camera_selector.pack(side=tk.LEFT, padx=5)
        self.load_camera_options()
        
        # Start/stop buttons
        self.start_btn = ttk.Button(
            self.control_frame, 
            text="Start Feed", 
            command=self.start_stream
        )
        self.stop_btn = ttk.Button(
            self.control_frame, 
            text="Stop Feed", 
            command=self.stop_stream,
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_var = tk.StringVar(value="Camera: Ready")
        self.status_label = ttk.Label(
            self.frame,
            textvariable=self.status_var,
            foreground="green"
        )
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

    def load_camera_options(self):
        """Load available camera sources from config file"""
        try:
            cameras = load_camera_config()
            options = []
            
            for cam in cameras:
                if cam.get('enabled', False):
                    display_name = f"{cam['camera_id']} ({cam.get('event_type', 'unknown')})"
                    options.append((display_name, cam['url'], cam['camera_id']))
            
            if not options:
                messagebox.showwarning("No Cameras", "No enabled cameras found in configuration")
                self.start_btn.config(state=tk.DISABLED)
                return
            
            self.camera_selector['values'] = [opt[0] for opt in options]
            self.camera_urls = {opt[0]: (opt[1], opt[2]) for opt in options}  # (url, camera_id)
            
            if options:
                self.camera_selector.current(0)
                
        except Exception as e:
            print(f"[ERROR] Failed to load camera config: {str(e)}")
            messagebox.showerror("Configuration Error", f"Failed to load camera configuration:\n{str(e)}")
            self.start_btn.config(state=tk.DISABLED)

    def start_stream(self):
        """Start video streaming thread"""
        if not self.running:
            selected_camera = self.camera_selector.get()
            if not selected_camera:
                messagebox.showerror("Error", "No camera selected")
                return
                
            self.current_camera_url, self.current_camera_id = self.camera_urls[selected_camera]
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Camera: Connecting to {selected_camera}...")
            
            # Start video capture thread
            self.capture_thread = threading.Thread(
                target=self._video_capture,
                daemon=True
            )
            self.capture_thread.start()
            
            # Start video update loop
            self._update_video()

    def _video_capture(self):
        """Thread for capturing video frames"""
        try:
            # Handle webcam (integer index) vs RTSP URLs
            camera_source = (
                int(self.current_camera_url) 
                if str(self.current_camera_url).isdigit() 
                else self.current_camera_url
            )
            
            self.cap = cv2.VideoCapture(camera_source)
            
            if not self.cap.isOpened():
                print(f"[ERROR] Failed to open camera: {self.current_camera_url}")
                messagebox.showerror(
                    "Camera Error", 
                    f"Failed to open camera: {self.current_camera_url}"
                )
                self.running = False
                self.status_var.set("Camera: Connection failed")
                return
                
            self.status_var.set(f"Camera: {self.camera_selector.get()} - Running")
            print(f"[INFO] Camera {self.current_camera_id} connected successfully")
            
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    # Check if recognition system is running and has processed frames
                    display_frame = frame
                    if (self.main_app.recognition_system and 
                        hasattr(self.main_app.recognition_system, 'camera_data')):
                        
                        with self.main_app.recognition_system.data_lock:
                            cam_data = self.main_app.recognition_system.camera_data.get(
                                self.current_camera_id, {}
                            )
                            if 'raw_image' in cam_data:
                                display_frame = cam_data['raw_image']
                    
                    # Also try to get the latest frame from shared state
                    try:
                        from shared_state import latest_frames, frame_lock
                        with frame_lock:
                            if self.current_camera_id in latest_frames:
                                display_frame = latest_frames[self.current_camera_id]
                    except ImportError:
                        pass  # shared_state not available
                    
                    if self.video_queue.empty():
                        self.video_queue.put(display_frame)
                else:
                    print(f"[WARNING] Failed to read frame from camera {self.current_camera_id}")
        
        except Exception as e:
            print(f"[ERROR] Camera error: {str(e)}")
            messagebox.showerror("Camera Error", f"Camera error: {str(e)}")
            self.running = False
            self.status_var.set("Camera: Error occurred")
        finally:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()

    def _update_video(self):
        """Update the video display with recognition data"""
        if not self.running:
            return
            
        try:
            if not self.video_queue.empty():
                frame = self.video_queue.get()
                
                # Convert to RGB and resize
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # Maintain aspect ratio
                width, height = img.size
                max_width, max_height = 800, 600
                ratio = min(max_width/width, max_height/height)
                new_size = (int(width*ratio), int(height*ratio))
                img = img.resize(new_size, Image.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk, text="")
        
        except Exception as e:
            print(f"[WARNING] Error updating video display: {str(e)}")
        
        if self.running:
            self.update_job = self.frame.after(30, self._update_video)

    def stop_stream(self):
        """Stop video streaming"""
        self.running = False
        if self.update_job:
            self.frame.after_cancel(self.update_job)
            self.update_job = None
            
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Camera: Stopped")
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            
        # Reset video display
        self.video_label.config(image="", text="Camera feed stopped")
        if hasattr(self.video_label, 'imgtk'):
            delattr(self.video_label, 'imgtk')

    def release(self):
        """Release all resources"""
        self.stop_stream()
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1)