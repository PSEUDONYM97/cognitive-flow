"""
Shared fixtures for Cognitive Flow tests.

Provides isolated temp directories and mock objects so tests never
touch real config files, audio hardware, or GPU resources.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def tmp_config(tmp_path):
    """Provide a temporary config file path and helper to write config JSON."""
    config_file = tmp_path / "config.json"

    def _write(data: dict):
        config_file.write_text(json.dumps(data, indent=2))
        return config_file

    return config_file, _write


@pytest.fixture
def tmp_stats_file(tmp_path):
    """Provide a temporary statistics file path."""
    return tmp_path / "statistics.json"


@pytest.fixture
def sample_stats():
    """Return a realistic statistics dict for testing."""
    return {
        "total_seconds": 120.5,
        "total_records": 10,
        "total_words": 350,
        "total_characters": 1800,
        "last_used": "2025-06-15T10:30:00",
        "total_processing_time": 15.2,
        "session_stats": {
            "records": 3,
            "words": 100,
            "characters": 520,
            "processing_time": 4.5,
        },
        "performance_history": [
            {
                "timestamp": "2025-06-15T10:30:00",
                "audio_duration": 12.0,
                "processing_time": 1.5,
                "words": 35,
                "speed_ratio": 0.125,
            }
        ],
    }


@pytest.fixture
def text_processor():
    """Return a fresh TextProcessor instance."""
    from cognitive_flow.app import TextProcessor
    return TextProcessor()
