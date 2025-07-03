import threading
import numpy as np
from typing import Dict, Optional

# Shared camera frames
latest_frames: Dict[str, np.ndarray] = {}
frame_lock = threading.Lock()

# System status
system_status = {
    'running': False,
    'cameras_connected': [],
    'last_error': None
}
status_lock = threading.Lock()

def update_system_status(running: Optional[bool] = None, 
                       cameras: Optional[list] = None,
                       error: Optional[str] = None):
    with status_lock:
        if running is not None:
            system_status['running'] = running
        if cameras is not None:
            system_status['cameras_connected'] = cameras
        if error is not None:
            system_status['last_error'] = error

def get_system_status() -> dict:
    with status_lock:
        return system_status.copy()

def get_latest_frame(camera_id: str) -> Optional[np.ndarray]:
    with frame_lock:
        return latest_frames.get(camera_id, None)

def set_latest_frame(camera_id: str, frame: np.ndarray):
    """Get the latest frame for a camera"""
    with frame_lock:
        latest_frames[camera_id] = frame.copy()