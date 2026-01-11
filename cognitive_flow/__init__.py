"""
Cognitive Flow - Local voice-to-text with GPU acceleration.

A free, privacy-focused alternative to WhisperTyping.
Press ~ (tilde) to record, speak, press ~ again - text is typed directly into any application.
"""

__version__ = "1.8.4"
__author__ = "Jared Williams"

from .app import CognitiveFlowApp, main

__all__ = ["CognitiveFlowApp", "main"]
