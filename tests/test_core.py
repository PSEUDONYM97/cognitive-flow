"""
Tests for core application logic - TextProcessor, VirtualKeyboard sanitization,
and the ColoredLogger.

These test pure-function paths that do not require audio hardware,
GUI frameworks, or Windows APIs.
"""

import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# TextProcessor - spoken punctuation, filler removal, artifact correction
# ---------------------------------------------------------------------------

class TestTextProcessorPunctuation:

    @pytest.mark.parametrize("spoken,symbol", [
        ("period", "."),
        ("full stop", "."),
        ("comma", ","),
        ("question mark", "?"),
        ("exclamation mark", "!"),
        ("exclamation point", "!"),
        ("colon", ":"),
        ("semicolon", ";"),
        ("dash", "-"),
        ("hyphen", "-"),
        ("ellipsis", "..."),
    ])
    def test_spoken_punctuation_replaced(self, text_processor, spoken, symbol):
        result = text_processor.process(f"hello {spoken} world")
        assert symbol in result

    def test_semicolon_single_word_only(self, text_processor):
        """'semicolon' (one word) maps to ; but 'semi colon' (two words)
        gets split - 'colon' maps to ':' independently."""
        result = text_processor.process("hello semicolon world")
        assert ";" in result

    def test_newline_consumed_by_whitespace_cleanup(self, text_processor):
        """'new line' -> newline char, but final \\s+ cleanup collapses it.
        This is expected behavior - spoken newlines only work with VirtualKeyboard."""
        result = text_processor.process("hello new line world")
        # The newline gets produced but then collapsed by whitespace normalization
        assert "hello" in result and "world" in result

    def test_punctuation_spacing_cleaned(self, text_processor):
        result = text_processor.process("hello comma world")
        assert result == "hello, world"

    def test_multiple_punctuation(self, text_processor):
        result = text_processor.process("yes period no comma maybe")
        assert result == "yes. no, maybe"

    def test_case_insensitive_punctuation(self, text_processor):
        result = text_processor.process("hello PERIOD world")
        assert result == "hello. world"


class TestTextProcessorFillerWords:

    @pytest.mark.parametrize("filler", [
        "um", "uh", "uhh", "umm", "ummm", "er", "err", "ah", "ahh",
        "hmm", "hmmm", "mm", "mmm",
    ])
    def test_filler_removed(self, text_processor, filler):
        result = text_processor.process(f"I {filler} think so")
        assert filler not in result.lower().split()
        assert "think" in result

    def test_filler_at_start(self, text_processor):
        result = text_processor.process("Um hello world")
        assert result.lower().startswith("hello")

    def test_filler_at_end(self, text_processor):
        result = text_processor.process("hello world uh")
        assert result == "hello world"

    def test_multiple_fillers(self, text_processor):
        result = text_processor.process("um well uh I uh think um so")
        assert "um" not in result.lower().split()
        assert "uh" not in result.lower().split()

    def test_filler_with_trailing_punctuation(self, text_processor):
        result = text_processor.process("I um, think so")
        assert "think" in result


class TestTextProcessorWhisperArtifacts:

    def test_comma_nd_to_command(self, text_processor):
        result = text_processor.process("run the ,nd")
        assert "command" in result.lower()

    def test_comma_nds_to_commands(self, text_processor):
        result = text_processor.process("multiple ,nds")
        assert "commands" in result.lower()

    def test_comma_nt_to_comment(self, text_processor):
        result = text_processor.process("add a ,nt")
        assert "comment" in result.lower()

    def test_period_riod_artifact_corrected(self, text_processor):
        """'.riod' is a Whisper artifact that gets corrected to ' period',
        then 'period' is converted to '.' by spoken punctuation replacement."""
        result = text_processor.process("the .riod ended")
        # The correction chain: .riod -> " period" -> "."
        assert "." in result
        assert ".riod" not in result

    def test_question_estion_to_question(self, text_processor):
        result = text_processor.process("ask a ?estion")
        assert "question" in result.lower()


class TestTextProcessorCharNormalization:
    """TextProcessor.CHAR_NORMALIZE handles ASCII-range normalizations
    (double-dash to dash, etc.). Unicode smart quotes are only handled
    by VirtualKeyboard, not by TextProcessor."""

    def test_double_dash_normalized(self, text_processor):
        result = text_processor.process("hello -- world")
        assert "--" not in result
        assert "-" in result

    def test_ascii_ellipsis_preserved(self, text_processor):
        result = text_processor.process("wait...")
        assert "..." in result

    def test_unicode_smart_quotes_pass_through(self, text_processor):
        """TextProcessor does NOT normalize unicode smart quotes -
        that is VirtualKeyboard's job."""
        # Use chr() to construct the actual unicode chars at runtime
        # to avoid encoding issues in the test file itself
        left_single = chr(0x2018)
        result = text_processor.process(f"he said {left_single}hello{left_single}")
        # Smart quotes pass through TextProcessor untouched
        assert left_single in result


class TestTextProcessorCustomReplacements:

    def test_custom_replacement(self, text_processor):
        text_processor.REPLACEMENTS = {"claw": "CLAUDE"}
        result = text_processor.process("ask claw about it")
        assert "CLAUDE" in result
        assert "claw" not in result

    def test_replacement_respects_word_boundaries(self, text_processor):
        text_processor.REPLACEMENTS = {"claw": "CLAUDE"}
        result = text_processor.process("the outlaw escaped")
        assert "outlaw" in result
        assert "CLAUDE" not in result

    def test_empty_replacements(self, text_processor):
        text_processor.REPLACEMENTS = {}
        result = text_processor.process("nothing changes")
        assert result == "nothing changes"


class TestTextProcessorHallucinationLoop:

    def test_detects_hallucination(self, text_processor):
        hallucinated = " ".join(["word"] * 15) + " fine"
        result = text_processor.process(hallucinated)
        assert result.count("word") < 5

    def test_short_repetition_preserved(self, text_processor):
        normal = "really really really important"
        result = text_processor.process(normal)
        assert result.count("really") == 3


class TestTextProcessorEdgeCases:

    def test_empty_string(self, text_processor):
        assert text_processor.process("") == ""

    def test_whitespace_only(self, text_processor):
        assert text_processor.process("   ") == ""

    def test_only_filler_words(self, text_processor):
        result = text_processor.process("um uh er")
        assert result == ""

    def test_preserves_normal_text(self, text_processor):
        result = text_processor.process("The quick brown fox jumps over the lazy dog")
        assert result == "The quick brown fox jumps over the lazy dog"

    def test_multipass_integration(self, text_processor):
        """Test that all processing passes work together:
        filler removal, custom replacements, spoken punctuation."""
        text_processor.REPLACEMENTS = {"claude": "CLAUDE"}
        result = text_processor.process(
            "um ask claude comma that is great period"
        )
        # Should: remove "um", replace "claude" -> CLAUDE,
        # replace "comma" -> ",", replace "period" -> "."
        assert "um" not in result.split()
        assert "CLAUDE" in result
        assert "," in result
        assert result.endswith(".")


# ---------------------------------------------------------------------------
# VirtualKeyboard.sanitize_text() - character safety
# ---------------------------------------------------------------------------

class TestVirtualKeyboardSanitize:

    def test_normal_text_passes_through(self):
        from cognitive_flow.app import VirtualKeyboard
        assert VirtualKeyboard.sanitize_text("Hello, world!") == "Hello, world!"

    def test_backtick_replaced(self):
        from cognitive_flow.app import VirtualKeyboard
        result = VirtualKeyboard.sanitize_text("use `code` here")
        assert "`" not in result
        assert "'" in result

    def test_control_chars_removed(self):
        from cognitive_flow.app import VirtualKeyboard
        text = "hello\x00\x01\x02world"
        result = VirtualKeyboard.sanitize_text(text)
        assert result == "helloworld"

    def test_newline_and_tab_preserved(self):
        from cognitive_flow.app import VirtualKeyboard
        result = VirtualKeyboard.sanitize_text("line1\nline2\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_zero_width_chars_removed(self):
        from cognitive_flow.app import VirtualKeyboard
        text = "hello\u200bworld"
        result = VirtualKeyboard.sanitize_text(text)
        assert "\u200b" not in result

    def test_smart_quotes_replaced(self):
        from cognitive_flow.app import VirtualKeyboard
        text = "he said \u201chello\u201d"
        result = VirtualKeyboard.sanitize_text(text)
        assert '"' in result
        assert "\u201c" not in result
        assert "\u201d" not in result

    def test_em_dash_replaced(self):
        from cognitive_flow.app import VirtualKeyboard
        result = VirtualKeyboard.sanitize_text("word\u2014word")
        assert "\u2014" not in result
        assert "-" in result

    def test_ellipsis_char_replaced(self):
        from cognitive_flow.app import VirtualKeyboard
        result = VirtualKeyboard.sanitize_text("wait\u2026")
        assert "\u2026" not in result
        assert "..." in result

    def test_non_breaking_space_replaced(self):
        from cognitive_flow.app import VirtualKeyboard
        result = VirtualKeyboard.sanitize_text("hello\u00a0world")
        assert "\u00a0" not in result
        assert " " in result

    def test_empty_string(self):
        from cognitive_flow.app import VirtualKeyboard
        assert VirtualKeyboard.sanitize_text("") == ""

    def test_only_dangerous_chars_returns_empty(self):
        from cognitive_flow.app import VirtualKeyboard
        result = VirtualKeyboard.sanitize_text("\x00\x01\x02\x7f")
        assert result == ""

    @pytest.mark.parametrize("char", [
        "\x1b",    # Escape
        "\x7f",    # DEL
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\ufeff",  # BOM
        "\u2028",  # Line separator
        "\u2029",  # Paragraph separator
    ])
    def test_dangerous_chars_parametrized(self, char):
        from cognitive_flow.app import VirtualKeyboard
        result = VirtualKeyboard.sanitize_text(f"before{char}after")
        assert char not in result


# ---------------------------------------------------------------------------
# ColoredLogger - formatting and file output
# ---------------------------------------------------------------------------

class TestColoredLogger:

    def test_format_without_colors(self):
        from cognitive_flow.logger import ColoredLogger
        log = ColoredLogger(use_colors=False)
        formatted = log._format("INFO", "Test", "hello")
        assert "[Test]" in formatted
        assert "hello" in formatted

    def test_file_logging(self, tmp_path):
        from cognitive_flow.logger import ColoredLogger
        log_file = tmp_path / "test.log"
        log = ColoredLogger(use_colors=False)
        log.set_log_file(log_file)

        log.info("Cat", "message one")
        log.error("Cat", "message two")

        content = log_file.read_text()
        assert "message one" in content
        assert "message two" in content
        assert "SESSION" in content

    def test_timer_context_manager(self, tmp_path):
        from cognitive_flow.logger import ColoredLogger
        log_file = tmp_path / "timer.log"
        log = ColoredLogger(use_colors=False)
        log.set_log_file(log_file)

        with log.timer("Bench", "test_op"):
            time.sleep(0.01)

        content = log_file.read_text()
        assert "test_op" in content
        assert "ms" in content

    def test_all_log_levels(self, tmp_path):
        from cognitive_flow.logger import ColoredLogger
        log_file = tmp_path / "levels.log"
        log = ColoredLogger(use_colors=False)
        log.set_log_file(log_file)

        log.debug("D", "debug msg")
        log.info("I", "info msg")
        log.success("S", "success msg")
        log.warning("W", "warning msg")
        log.error("E", "error msg")
        log.timing("T", "operation", 42.5)

        content = log_file.read_text()
        assert "debug msg" in content
        assert "info msg" in content
        assert "success msg" in content
        assert "warning msg" in content
        assert "error msg" in content
        assert "42.5ms" in content

    def test_log_transcription_suspects(self, tmp_path):
        from cognitive_flow.logger import ColoredLogger
        log_file = tmp_path / "transcription.log"
        log = ColoredLogger(use_colors=False)
        log.set_log_file(log_file)

        raw = "hello\x05world\u0080"
        log.log_transcription(raw=raw, processed="helloworld", audio_sec=2.5)

        content = log_file.read_text()
        assert "SUSPECT" in content
        assert "TRANSCRIPTION" in content

    def test_log_transcription_no_file_noop(self):
        from cognitive_flow.logger import ColoredLogger
        log = ColoredLogger(use_colors=False)
        # Should not raise
        log.log_transcription(raw="text", processed="text", audio_sec=1.0)
