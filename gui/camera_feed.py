# gui/camera_feed.py
import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from queue import Queue

class CameraFeed:
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.frame = ttk.Frame(self.parent)
        self.video_queue = Queue(maxsize=1)
        self.running = False
        
        # Video display
        self.video_label = tk.Label(self.frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Camera selection
        self.camera_var = tk.StringVar()
        self.camera_selector = ttk.Combobox(
            self.frame, 
            textvariable=self.camera_var,
            state='readonly'
        )
        self.camera_selector.pack(fill=tk.X, padx=5, pady=5)
        self.load_camera_options()
        
        # Start/stop buttons
        self.btn_frame = ttk.Frame(self.frame)
        self.start_btn = ttk.Button(
            self.btn_frame, 
            text="Start Feed", 
            command=self.start_stream
        )
        self.stop_btn = ttk.Button(
            self.btn_frame, 
            text="Stop Feed", 
            command=self.stop_stream,
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.btn_frame.pack(fill=tk.X, pady=5)

    def load_camera_options(self):
        """Load available camera sources"""
        options = [
            "Camera 1 (Main Entrance)", 
            "Camera 2 (Exit Gate)",
            "Webcam"
        ]
        self.camera_selector['values'] = options
        self.camera_selector.current(0)

    def start_stream(self):
        """Start video streaming thread"""
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Start video capture thread
            self.capture_thread = threading.Thread(
                target=self._video_capture,
                daemon=True
            )
            self.capture_thread.start()
            
            # Start video update loop
            self._update_video()

    def stop_stream(self):
        """Stop video streaming"""
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _video_capture(self):
        """Thread for capturing video frames"""
        cap = cv2.VideoCapture(0)  # Replace with your camera source
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.video_queue.empty():
                    self.video_queue.put(frame)
        
        cap.release()

    def _update_video(self):
        """Update the video display"""
        if not self.video_queue.empty():
            frame = self.video_queue.get()
            
            # Convert to RGB and resize
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            # Maintain aspect ratio
            width, height = img.size
            ratio = min(800/width, 600/height)
            new_size = (int(width*ratio), int(height*ratio))
            img = img.resize(new_size, Image.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        
        if self.running:
            self.frame.after(30, self._update_video)

    def release(self):
        """Release all resources"""
        self.stop_stream()
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1)