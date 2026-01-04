"""
Color-coded logging for Cognitive Flow
Clean, readable debug output with optional timing
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager


class ColoredLogger:
    """Windows-compatible colored logging with timing support"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[37m',      # White
        'SUCCESS': '\033[32m',   # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'TIMING': '\033[35m',    # Magenta
        'RESET': '\033[0m',
        'DIM': '\033[2m'
    }
    
    def __init__(self, use_colors=True):
        self.use_colors = use_colors and self._supports_color()
        self._session_start = time.perf_counter()
        self._log_file = None
        
        # Enable ANSI colors on Windows
        if sys.platform == 'win32' and self.use_colors:
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                self.use_colors = False
    
    def set_log_file(self, path: Path):
        """Enable file logging."""
        self._log_file = path
        self._write_file(f"\n{'='*60}")
        self._write_file(f"SESSION: {datetime.now().isoformat()}")
        self._write_file(f"{'='*60}")
    
    def _supports_color(self):
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self._session_start) * 1000
    
    def _format(self, level: str, category: str, message: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if self.use_colors:
            color = self.COLORS.get(level, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            dim = self.COLORS['DIM']
            return f"{dim}[{timestamp}]{reset} {color}[{category}]{reset} {message}"
        else:
            return f"[{timestamp}] [{category}] {message}"
    
    def _write_file(self, msg: str):
        if self._log_file:
            try:
                # Strip ANSI codes
                clean = msg
                for code in self.COLORS.values():
                    clean = clean.replace(code, '')
                with open(self._log_file, 'a', encoding='utf-8') as f:
                    f.write(clean + '\n')
            except:
                pass
    
    def _log(self, level: str, category: str, message: str):
        formatted = self._format(level, category, message)
        print(formatted)
        self._write_file(formatted)
    
    def debug(self, category: str, message: str):
        self._log('DEBUG', category, message)
    
    def info(self, category: str, message: str):
        self._log('INFO', category, message)
    
    def success(self, category: str, message: str):
        self._log('SUCCESS', category, message)
    
    def warning(self, category: str, message: str):
        self._log('WARNING', category, message)
    
    def error(self, category: str, message: str):
        self._log('ERROR', category, message)
    
    def timing(self, category: str, name: str, ms: float):
        self._log('TIMING', category, f"{name}: {ms:.1f}ms")
    
    def separator(self, char='=', length=60):
        line = char * length
        print(line)
        self._write_file(line)
    
    @contextmanager
    def timer(self, category: str, name: str):
        """Context manager for timing a block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - start) * 1000
            self.timing(category, name, ms)
    
    def log_transcription(self, raw: str, processed: str, audio_sec: float, timings: dict = None):
        """Log transcription details to file only."""
        if not self._log_file:
            return
        
        self._write_file(f"\n{'-'*40}")
        self._write_file(f"TRANSCRIPTION @ {datetime.now().strftime('%H:%M:%S')}")
        self._write_file(f"audio: {audio_sec:.2f}s")
        
        if timings:
            for k, v in timings.items():
                self._write_file(f"  {k}: {v:.1f}ms")
        
        self._write_file(f"raw ({len(raw)}): {raw!r}")
        self._write_file(f"out ({len(processed)}): {processed!r}")
        
        # Flag suspicious
        for i, c in enumerate(raw):
            code = ord(c)
            if code < 32 and c not in '\n\t':
                self._write_file(f"  SUSPECT pos {i}: 0x{code:02X} control")
            elif code > 126:
                self._write_file(f"  SUSPECT pos {i}: U+{code:04X} '{c}'")
        self._write_file(f"{'-'*40}")


# Global instance
logger = ColoredLogger()
