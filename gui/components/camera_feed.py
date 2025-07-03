# gui/components/camera_feed.py
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from typing import Dict, List
import numpy as np
from .shared_state import get_latest_frame

class CameraFeed(ttk.Frame):
    def __init__(self, parent, available_cameras: List[Dict]):
        super().__init__(parent)
        self.available_cameras = available_cameras
        self.current_camera_id = None
        self.imgtk = None  # Keep reference to prevent garbage collection
        self.setup_ui()
        
        # Select first camera by default if available
        if self.available_cameras:
            self.camera_combobox.current(0)
            self.on_camera_select()

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Camera selection controls
        self.controls_frame = ttk.Frame(self)
        self.controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(self.controls_frame, text="Select Camera:").pack(side=tk.LEFT, padx=5)
        
        self.camera_combobox = ttk.Combobox(
            self.controls_frame,
            values=[cam['camera_name'] for cam in self.available_cameras],
            state="readonly"
        )
        self.camera_combobox.pack(side=tk.LEFT, padx=5)
        self.camera_combobox.bind("<<ComboboxSelected>>", self.on_camera_select)
        
        # Camera display frame
        self.camera_frame = ttk.LabelFrame(self, text="Camera Feed")
        self.camera_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.camera_frame.columnconfigure(0, weight=1)
        self.camera_frame.rowconfigure(0, weight=1)
        
        # Video display label
        self.video_label = ttk.Label(self.camera_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")
    
    def on_camera_select(self, event=None):
        selected_name = self.camera_combobox.get()
        selected_camera = next(
            (cam for cam in self.available_cameras if cam['camera_name'] == selected_name),
            None
        )
        
        if selected_camera:
            self.current_camera_id = selected_camera['camera_id']
            self.camera_frame.config(text=f"Camera: {selected_name}")
            self.update_feed()

    def update_feed(self):
        if self.current_camera_id:
            frame = get_latest_frame(self.current_camera_id)
            if frame is not None and frame.size > 0:
                try:
                    # Convert frame to RGB and resize
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    
                    # Maintain aspect ratio
                    width, height = img.size
                    max_width, max_height = 640, 480
                    ratio = min(max_width/width, max_height/height)
                    new_size = (int(width*ratio), int(height*ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                    
                    # Update label
                    self.imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.config(image=self.imgtk)
                except Exception as e:
                    print(f"[WARNING] Failed to update camera feed: {e}")
        
        self.after(50, self.update_feed)