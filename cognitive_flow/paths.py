"""
App data directory and file paths for Cognitive Flow.
Stores config, history, and statistics in the appropriate OS location.
"""

import os
import sys
from pathlib import Path


def get_app_data_dir() -> Path:
    """Get the application data directory, creating it if needed.
    
    Windows: %APPDATA%/CognitiveFlow
    Linux/Mac: ~/.cognitive_flow
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
        app_dir = base / "CognitiveFlow"
    else:
        app_dir = Path.home() / ".cognitive_flow"
    
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


# App data directory and file paths
APP_DATA_DIR = get_app_data_dir()
CONFIG_FILE = APP_DATA_DIR / "config.json"
STATS_FILE = APP_DATA_DIR / "statistics.json"
HISTORY_FILE = APP_DATA_DIR / "history.json"
DEBUG_LOG_FILE = APP_DATA_DIR / "debug_transcriptions.log"

# Audio archive directory
AUDIO_ARCHIVE_DIR = APP_DATA_DIR / "audio"
AUDIO_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
