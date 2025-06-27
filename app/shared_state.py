import threading

latest_frames = {}
frame_lock = threading.Lock()