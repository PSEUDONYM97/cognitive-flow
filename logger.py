"""
Color-coded logging for Cognitive Flow
Clean, readable debug output
"""

import sys
from datetime import datetime
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ColoredLogger:
    """Windows-compatible colored logging"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[37m',      # White
        'SUCCESS': '\033[32m',   # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m'
    }
    
    # Emoji/symbols for each level
    SYMBOLS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️ ',
        'SUCCESS': '✓',
        'WARNING': '⚠️ ',
        'ERROR': '✗'
    }
    
    def __init__(self, use_colors=True, use_symbols=False):
        self.use_colors = use_colors and self._supports_color()
        self.use_symbols = use_symbols
        
        # Enable ANSI colors on Windows
        if sys.platform == 'win32' and self.use_colors:
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                self.use_colors = False
    
    def _supports_color(self):
        """Check if terminal supports colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def _format_message(self, level: str, category: str, message: str) -> str:
        """Format log message with colors and structure"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if self.use_colors:
            color = self.COLORS.get(level, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            dim = self.COLORS['DIM']
            
            # Structured format: [TIME] [CATEGORY] Message
            return f"{dim}[{timestamp}]{reset} {color}[{category}]{reset} {message}"
        else:
            # Plain format
            symbol = self.SYMBOLS.get(level, '') if self.use_symbols else ''
            return f"[{timestamp}] [{category}] {symbol}{message}"
    
    def debug(self, category: str, message: str):
        """Debug level - development info"""
        print(self._format_message('DEBUG', category, message))
    
    def info(self, category: str, message: str):
        """Info level - general information"""
        print(self._format_message('INFO', category, message))
    
    def success(self, category: str, message: str):
        """Success level - completed actions"""
        print(self._format_message('SUCCESS', category, message))
    
    def warning(self, category: str, message: str):
        """Warning level - potential issues"""
        print(self._format_message('WARNING', category, message))
    
    def error(self, category: str, message: str):
        """Error level - failures"""
        print(self._format_message('ERROR', category, message))
    
    def separator(self, char='=', length=60):
        """Print a separator line"""
        print(char * length)
    
    def header(self, text: str):
        """Print a formatted header"""
        self.separator()
        print(f"  {text}")
        self.separator()


# Global logger instance
logger = ColoredLogger(use_colors=True, use_symbols=False)
