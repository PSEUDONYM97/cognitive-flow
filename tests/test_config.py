"""
Tests for configuration loading, saving, and validation logic.

Covers paths.py (app data directory resolution) and the config/stats
persistence in app.py. Uses tmp_path fixtures so real config is never touched.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# paths.py - get_app_data_dir()
# ---------------------------------------------------------------------------

class TestGetAppDataDir:

    def test_returns_path_object(self):
        from cognitive_flow.paths import get_app_data_dir
        result = get_app_data_dir()
        assert isinstance(result, Path)

    def test_directory_exists(self):
        from cognitive_flow.paths import get_app_data_dir
        result = get_app_data_dir()
        assert result.exists()
        assert result.is_dir()

    @patch("sys.platform", "win32")
    def test_windows_uses_appdata(self, tmp_path):
        with patch.dict("os.environ", {"APPDATA": str(tmp_path)}):
            import importlib
            import cognitive_flow.paths as paths_mod
            importlib.reload(paths_mod)
            result = paths_mod.get_app_data_dir()
            assert result == tmp_path / "CognitiveFlow"
            assert result.exists()

    @patch("sys.platform", "linux")
    def test_linux_uses_home_dot_dir(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            import importlib
            import cognitive_flow.paths as paths_mod
            importlib.reload(paths_mod)
            result = paths_mod.get_app_data_dir()
            assert result == tmp_path / ".cognitive_flow"
            assert result.exists()


class TestPathConstants:

    def test_config_file_is_json(self):
        from cognitive_flow.paths import CONFIG_FILE
        assert CONFIG_FILE.suffix == ".json"
        assert CONFIG_FILE.name == "config.json"

    def test_stats_file_is_json(self):
        from cognitive_flow.paths import STATS_FILE
        assert STATS_FILE.suffix == ".json"

    def test_history_file_is_json(self):
        from cognitive_flow.paths import HISTORY_FILE
        assert HISTORY_FILE.suffix == ".json"

    def test_debug_log_is_log(self):
        from cognitive_flow.paths import DEBUG_LOG_FILE
        assert DEBUG_LOG_FILE.suffix == ".log"

    def test_audio_archive_dir_exists(self):
        from cognitive_flow.paths import AUDIO_ARCHIVE_DIR
        assert AUDIO_ARCHIVE_DIR.exists()
        assert AUDIO_ARCHIVE_DIR.is_dir()

    def test_all_paths_share_parent(self):
        from cognitive_flow.paths import (
            APP_DATA_DIR, CONFIG_FILE, STATS_FILE, HISTORY_FILE,
        )
        assert CONFIG_FILE.parent == APP_DATA_DIR
        assert STATS_FILE.parent == APP_DATA_DIR
        assert HISTORY_FILE.parent == APP_DATA_DIR


# ---------------------------------------------------------------------------
# Statistics - load, record, compute
# ---------------------------------------------------------------------------

class TestStatistics:

    def test_fresh_stats_defaults(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        assert stats.stats["total_records"] == 0
        assert stats.stats["total_words"] == 0
        assert stats.stats["total_seconds"] == 0

    def test_load_existing_stats(self, tmp_stats_file, sample_stats):
        tmp_stats_file.write_text(json.dumps(sample_stats))
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        assert stats.stats["total_records"] == 10
        assert stats.stats["total_words"] == 350

    def test_record_increments_counts(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        stats.record(duration_seconds=5.0, text="hello world test", processing_time=0.5)

        assert stats.stats["total_records"] == 1
        assert stats.stats["total_words"] == 3
        assert stats.stats["total_characters"] == 16
        assert stats.stats["total_seconds"] == 5.0
        assert stats.stats["total_processing_time"] == 0.5

    def test_record_persists_to_disk(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        stats.record(duration_seconds=2.0, text="saved", processing_time=0.1)

        saved = json.loads(tmp_stats_file.read_text())
        assert saved["total_records"] == 1
        assert saved["total_words"] == 1

    def test_record_accumulates(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        stats.record(duration_seconds=3.0, text="one two", processing_time=0.2)
        stats.record(duration_seconds=4.0, text="three four five", processing_time=0.3)

        assert stats.stats["total_records"] == 2
        assert stats.stats["total_words"] == 5
        assert stats.stats["total_seconds"] == 7.0

    def test_performance_history_capped_at_100(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        for i in range(110):
            stats.record(duration_seconds=1.0, text=f"word{i}", processing_time=0.1)

        assert len(stats.stats["performance_history"]) == 100

    def test_performance_history_newest_first(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        stats.record(duration_seconds=1.0, text="first", processing_time=0.1)
        stats.record(duration_seconds=2.0, text="second entry", processing_time=0.2)

        history = stats.stats["performance_history"]
        assert history[0]["words"] == 2  # "second entry"
        assert history[1]["words"] == 1  # "first"

    def test_speed_ratio_calculated(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        stats.record(duration_seconds=10.0, text="test", processing_time=2.0)

        entry = stats.stats["performance_history"][0]
        assert entry["speed_ratio"] == pytest.approx(0.2)

    def test_speed_ratio_zero_duration(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        stats.record(duration_seconds=0.0, text="edge case", processing_time=0.5)

        entry = stats.stats["performance_history"][0]
        assert entry["speed_ratio"] == 0

    def test_get_time_saved(self, tmp_stats_file, sample_stats):
        tmp_stats_file.write_text(json.dumps(sample_stats))
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        saved = stats.get_time_saved()
        assert "minutes" in saved or "hours" in saved

    def test_speaking_speed_wpm(self, tmp_stats_file, sample_stats):
        tmp_stats_file.write_text(json.dumps(sample_stats))
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        wpm = stats.get_speaking_speed_wpm()
        # 350 words / (120.5 seconds / 60) = ~174 wpm
        assert 170 < wpm < 180

    def test_speaking_speed_wpm_zero_seconds(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        assert stats.get_speaking_speed_wpm() == 0

    def test_seconds_per_word(self, tmp_stats_file, sample_stats):
        tmp_stats_file.write_text(json.dumps(sample_stats))
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        spw = stats.get_seconds_per_word()
        assert 0.34 < spw < 0.35

    def test_summary_format(self, tmp_stats_file, sample_stats):
        tmp_stats_file.write_text(json.dumps(sample_stats))
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        summary = stats.summary()
        assert "Recordings:" in summary
        assert "Words:" in summary
        assert "Time saved:" in summary

    def test_get_avg_speed_ratio(self, tmp_stats_file):
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        assert stats.get_avg_speed_ratio() == 0

        stats.record(duration_seconds=10.0, text="a", processing_time=2.0)
        stats.record(duration_seconds=10.0, text="b", processing_time=4.0)
        assert stats.get_avg_speed_ratio() == pytest.approx(0.3)

    def test_typing_vs_speaking_comparison(self, tmp_stats_file, sample_stats):
        tmp_stats_file.write_text(json.dumps(sample_stats))
        from cognitive_flow.app import Statistics
        stats = Statistics(stats_file=str(tmp_stats_file))
        comparison = stats.get_typing_vs_speaking_comparison()
        assert "typing_time" in comparison
        assert "speaking_time" in comparison
        assert "time_saved" in comparison
        assert "efficiency_ratio" in comparison
        assert comparison["time_saved"] > 0


# ---------------------------------------------------------------------------
# UpdateChecker.parse_version()
# ---------------------------------------------------------------------------

class TestUpdateCheckerParseVersion:

    def test_basic_version(self):
        from cognitive_flow.app import UpdateChecker
        assert UpdateChecker.parse_version("1.2.3") == (1, 2, 3)

    def test_strips_v_prefix(self):
        from cognitive_flow.app import UpdateChecker
        assert UpdateChecker.parse_version("v1.21.1") == (1, 21, 1)

    def test_invalid_returns_zeros(self):
        from cognitive_flow.app import UpdateChecker
        assert UpdateChecker.parse_version("not-a-version") == (0, 0, 0)
        assert UpdateChecker.parse_version("") == (0, 0, 0)

    def test_comparison_logic(self):
        from cognitive_flow.app import UpdateChecker
        parse = UpdateChecker.parse_version
        assert parse("1.21.1") > parse("1.20.0")
        assert parse("1.21.1") > parse("1.2.0")  # 21 > 2, not string comparison
        assert parse("2.0.0") > parse("1.99.99")
        assert parse("1.0.0") == parse("v1.0.0")

    @pytest.mark.parametrize("version,expected", [
        ("0.0.1", (0, 0, 1)),
        ("10.20.30", (10, 20, 30)),
        ("v0.0.0", (0, 0, 0)),
    ])
    def test_parametrized_versions(self, version, expected):
        from cognitive_flow.app import UpdateChecker
        assert UpdateChecker.parse_version(version) == expected
