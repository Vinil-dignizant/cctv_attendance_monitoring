from .api import app
from .recognition import MultiCameraFaceRecognition
import threading

def run_recognition():
    system = MultiCameraFaceRecognition("config/camera_config.yaml")
    system.start()

if __name__ == "__main__":
    # Start recognition system in background thread
    threading.Thread(target=run_recognition, daemon=True).start()
    
    # Start FastAPI server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)