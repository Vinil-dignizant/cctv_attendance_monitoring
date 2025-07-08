import json
import os
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    CONFIG_FILE = "database_config.json"
    DEFAULT_CONFIG = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "face_recognition_db",
        "DB_USER": "face_recognition_user",
        "DB_PASSWORD": "vinil123",
        "PGTZ": "Asia/Kolkata",
        "TZ": "Asia/Kolkata"
    }

    def __init__(self):
        self.config_path = Path(__file__).parent.parent / self.CONFIG_FILE
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load config from file or create default"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            return self._create_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.DEFAULT_CONFIG.copy()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default config file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.DEFAULT_CONFIG, f, indent=4)
        return self.DEFAULT_CONFIG.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update and save configuration"""
        self.config.update(new_config)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get_db_url(self) -> str:
        """Construct database URL from config"""
        return (
            f"postgresql://{self.config['DB_USER']}:{self.config['DB_PASSWORD']}@"
            f"{self.config['DB_HOST']}:{self.config['DB_PORT']}/{self.config['DB_NAME']}"
        )