# gui/shared_state.py
"""
Shared state module for communication between recognition system and GUI
"""
import threading
import numpy as np

# Global variables for sharing camera frames between recognition system and GUI
latest_frames = {}
frame_lock = threading.Lock()

def get_latest_frame(camera_id):
    """Get the latest frame for a specific camera"""
    with frame_lock:
        return latest_frames.get(camera_id, None)

def set_latest_frame(camera_id, frame):
    """Set the latest frame for a specific camera"""
    with frame_lock:
        latest_frames[camera_id] = frame.copy() if frame is not None else None

def initialize_camera_frames(camera_ids):
    """Initialize frames dict for given camera IDs"""
    with frame_lock:
        for camera_id in camera_ids:
            if camera_id not in latest_frames:
                latest_frames[camera_id] = np.zeros((480, 640, 3), dtype=np.uint8)