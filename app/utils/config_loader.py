# app/utils/config_loader.py
import yaml
from pathlib import Path

def load_camera_config():
    config_path = Path("config") / "camera_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config.get('cameras', [])
    except Exception as e:
        print(f"Error loading camera config: {e}")
        return []