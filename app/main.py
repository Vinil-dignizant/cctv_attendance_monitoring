import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.api import app
from app.recognition import MultiCameraFaceRecognition
import threading

def run_recognition():
    config_path = os.path.join(project_root, 'config', 'camera_config.yaml')
    system = MultiCameraFaceRecognition(config_path)
    system.start()

def main():
    threading.Thread(target=run_recognition, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()