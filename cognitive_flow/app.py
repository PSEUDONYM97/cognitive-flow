"""
Cognitive Flow - Main application module.
Local voice-to-text with global hotkey.
"""

import ctypes
import ctypes.wintypes
import threading
import queue
import json
import os
import wave
import tempfile
import time
import sys
from datetime import datetime
from pathlib import Path

# These get set during init_app()
GPU_AVAILABLE = False
HAS_TRAY = False
HAS_UI = False
CognitiveFlowUI = None
WhisperModel = None
pyaudio = None
pystray = None
Image = None
ImageDraw = None
logger = None


def init_app():
    """Initialize app - load libraries after banner is shown."""
    global GPU_AVAILABLE, HAS_TRAY, HAS_UI, CognitiveFlowUI
    global WhisperModel, pyaudio, pystray, Image, ImageDraw, logger
    
    from .logger import logger as _logger
    from .paths import CONFIG_FILE, STATS_FILE
    logger = _logger
    
    # Add CUDA libraries for GPU support (MUST be before importing faster_whisper)
    try:
        import site
        user_site = site.USER_SITE
        if user_site:
            cuda_path = Path(user_site) / "nvidia"
            if cuda_path.exists():
                cudnn_bin = cuda_path / "cudnn" / "bin"
                cublas_bin = cuda_path / "cublas" / "bin"
                
                if cudnn_bin.exists():
                    os.add_dll_directory(str(cudnn_bin))
                if cublas_bin.exists():
                    os.add_dll_directory(str(cublas_bin))
                
                os.environ['PATH'] = f"{cudnn_bin};{cublas_bin};" + os.environ.get('PATH', '')
                
                dll_count = 0
                for dll_dir in [cublas_bin, cudnn_bin]:
                    if dll_dir.exists():
                        for dll in dll_dir.glob("*.dll"):
                            try:
                                ctypes.CDLL(str(dll))
                                dll_count += 1
                            except Exception:
                                pass
                
                if dll_count > 0:
                    GPU_AVAILABLE = True
                    print(f"[CUDA] Loaded {dll_count} GPU libraries")
    except Exception as e:
        print(f"[CUDA] GPU setup failed: {e} - using CPU")
    
    import pyaudio as _pyaudio
    pyaudio = _pyaudio
    
    print("[Model] Loading Whisper engine...", end=" ", flush=True)
    from faster_whisper import WhisperModel as _WhisperModel
    WhisperModel = _WhisperModel
    print("done")
    
    # Optional: pystray for system tray
    try:
        import pystray as _pystray
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        pystray = _pystray
        Image = _Image
        ImageDraw = _ImageDraw
        HAS_TRAY = True
    except ImportError:
        print("[Note] Install pystray and pillow for system tray: pip install pystray pillow")
    
    # UI module
    try:
        from .ui import CognitiveFlowUI as _CognitiveFlowUI
        CognitiveFlowUI = _CognitiveFlowUI
        HAS_UI = True
        print("[UI] Using PyQt6")
    except ImportError as e:
        print(f"[Note] No UI module: {e}")

# Windows API constants
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_OEM_3 = 0xC0  # Tilde key

INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.wintypes.DWORD),
        ("scanCode", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG))
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG))
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.wintypes.DWORD),
        ("wParamL", ctypes.wintypes.WORD),
        ("wParamH", ctypes.wintypes.WORD)
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG))
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT)
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("union", INPUT_UNION)
    ]


user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

SetWindowsHookExW = user32.SetWindowsHookExW
UnhookWindowsHookEx = user32.UnhookWindowsHookEx
CallNextHookEx = user32.CallNextHookEx
GetMessageW = user32.GetMessageW
SendInput = user32.SendInput
GetForegroundWindow = user32.GetForegroundWindow

HOOKPROC = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.POINTER(KBDLLHOOKSTRUCT))


class Statistics:
    """Track usage statistics"""
    
    def __init__(self, stats_file: str | None = None):
        from .paths import STATS_FILE
        self.stats_file = Path(stats_file) if stats_file else STATS_FILE
        self.stats: dict = self._load_stats()
    
    def _load_stats(self) -> dict:
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        
        # Try to import from WhisperTyping
        wt_stats = Path(os.environ.get('LOCALAPPDATA', '')) / "WhisperTyping" / "statistics.json"
        if wt_stats.exists():
            print(f"[Stats] Importing from WhisperTyping...")
            with open(wt_stats, 'r') as f:
                imported = json.load(f)
            stats = {
                "total_seconds": imported.get("TotalSeconds", 0),
                "total_records": imported.get("TotalRecords", 0),
                "total_words": imported.get("TranscriptionWords", 0),
                "total_characters": imported.get("TranscriptionCharacters", 0),
                "last_used": imported.get("LastTranscriptionTime", ""),
                "imported_from_whispertyping": True
            }
            self._save(stats)
            return stats
        
        return {
            "total_seconds": 0,
            "total_records": 0,
            "total_words": 0,
            "total_characters": 0,
            "last_used": "",
            "imported_from_whispertyping": False,
            "total_processing_time": 0,
            "session_stats": {"records": 0, "words": 0, "characters": 0, "processing_time": 0},
            "performance_history": []
        }
    
    def _save(self, stats: dict | None = None):
        if stats is None:
            stats = self.stats
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def record(self, duration_seconds: float, text: str, processing_time: float = 0):
        words = len(text.split())
        chars = len(text)
        
        self.stats["total_seconds"] += duration_seconds
        self.stats["total_records"] += 1
        self.stats["total_words"] += words
        self.stats["total_characters"] += chars
        self.stats["total_processing_time"] = self.stats.get("total_processing_time", 0) + processing_time
        self.stats["last_used"] = datetime.now().isoformat()
        
        if "session_stats" not in self.stats:
            self.stats["session_stats"] = {"records": 0, "words": 0, "characters": 0, "processing_time": 0}
        
        self.stats["session_stats"]["records"] += 1
        self.stats["session_stats"]["words"] += words
        self.stats["session_stats"]["characters"] += chars
        self.stats["session_stats"]["processing_time"] += processing_time
        
        if "performance_history" not in self.stats:
            self.stats["performance_history"] = []
        
        perf_entry = {
            "timestamp": datetime.now().isoformat(),
            "audio_duration": duration_seconds,
            "processing_time": processing_time,
            "words": words,
            "speed_ratio": processing_time / duration_seconds if duration_seconds > 0 else 0
        }
        
        self.stats["performance_history"].insert(0, perf_entry)
        self.stats["performance_history"] = self.stats["performance_history"][:100]
        
        self._save()
    
    def get_time_saved(self) -> str:
        words = self.stats["total_words"]
        audio_seconds = self.stats["total_seconds"]
        typing_time_minutes = words / 30
        speaking_time_minutes = audio_seconds / 60
        saved_minutes = typing_time_minutes - speaking_time_minutes
        
        if saved_minutes < 60:
            return f"{saved_minutes:.0f} minutes"
        else:
            return f"{saved_minutes / 60:.1f} hours"
    
    def get_typing_vs_speaking_comparison(self) -> dict:
        words = self.stats["total_words"]
        audio_seconds = self.stats["total_seconds"]
        typing_time_minutes = words / 30
        speaking_time_minutes = audio_seconds / 60
        
        return {
            "typing_time": typing_time_minutes,
            "speaking_time": speaking_time_minutes,
            "time_saved": typing_time_minutes - speaking_time_minutes,
            "efficiency_ratio": typing_time_minutes / speaking_time_minutes if speaking_time_minutes > 0 else 0
        }
    
    def get_speaking_speed_wpm(self) -> float:
        words = self.stats["total_words"]
        audio_seconds = self.stats["total_seconds"]
        if audio_seconds == 0:
            return 0
        return words / (audio_seconds / 60)
    
    def get_seconds_per_word(self) -> float:
        words = self.stats["total_words"]
        audio_seconds = self.stats["total_seconds"]
        return audio_seconds / words if words > 0 else 0
    
    def summary(self) -> str:
        return (
            f"Recordings: {self.stats['total_records']:,} | "
            f"Words: {self.stats['total_words']:,} | "
            f"Time saved: ~{self.get_time_saved()}"
        )
    
    def get_avg_speed_ratio(self) -> float:
        history = self.stats.get("performance_history", [])
        if not history:
            return 0
        ratios = [h["speed_ratio"] for h in history if h.get("speed_ratio", 0) > 0]
        return sum(ratios) / len(ratios) if ratios else 0
    
    def get_avg_words_per_recording(self) -> float:
        records = self.stats.get("total_records", 0)
        if records == 0:
            return 0
        return self.stats.get("total_words", 0) / records


class TextProcessor:
    """Process transcribed text with correction pass for Whisper artifacts"""
    
    PUNCTUATION = {
        "period": ".", "full stop": ".", "comma": ",", "question mark": "?",
        "exclamation mark": "!", "exclamation point": "!", "colon": ":",
        "semicolon": ";", "semi colon": ";", "dash": "-", "hyphen": "-",
        "open parenthesis": "(", "close parenthesis": ")", "open bracket": "[",
        "close bracket": "]", "open brace": "{", "close brace": "}",
        "apostrophe": "'", "quote": '"', "open quote": '"', "close quote": '"',
        "ellipsis": "...", "new line": "\n", "newline": "\n",
        "new paragraph": "\n\n", "enter": "\n",
    }
    
    REPLACEMENTS = {"hashtag": "#", "clod": "CLAUDE"}
    
    CHAR_NORMALIZE = {
        "'": "'", "'": "'", '"': '"', '"': '"',
        "-": "-", "--": "-", "...": "...",
    }
    
    # Whisper sometimes outputs punctuation directly when it "hears" the word
    # e.g., "command" becomes ",nd" because Whisper thinks you said "comma" + "nd"
    # These patterns catch the most common artifacts
    # Order matters - more specific patterns first
    WHISPER_CORRECTIONS = [
        # Punctuation fused with following text
        # Handles: ",nd", "the,nd", " ,nd" -> "command", "the command", " command"
        (r",nd\b", " command"),           # ,nd -> command (add space, clean later)
        (r",nds\b", " commands"),         # ,nds -> commands (plural)
        (r",nding\b", " commanding"),     # ,nding -> commanding
        (r",nt\b", " comment"),           # ,nt -> comment  
        (r",nts\b", " comments"),         # ,nts -> comments
        (r",n\b", " common"),             # ,n -> common
        (r",nly\b", " commonly"),         # ,nly -> commonly
        (r"\.riod\b", " period"),         # .riod -> period
        (r":lon\b", " colon"),            # :lon -> colon  
        (r";micolon\b", " semicolon"),    # ;micolon -> semicolon
        (r"\?estion\b", " question"),     # ?estion -> question
        (r"!xclamation\b", " exclamation"), # !xclamation -> exclamation
        
        # Common Whisper mishearings
        (r"\bkama\b", "comma"),          # kama -> comma (phonetic mishearing)
    ]
    
    def _correct_whisper_artifacts(self, text: str) -> str:
        """First pass: fix Whisper's tendency to output literal punctuation"""
        import re
        
        for pattern, replacement in self.WHISPER_CORRECTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def process(self, text: str) -> str:
        if not text:
            return text
        
        import re
        
        # Pass 1: Fix Whisper artifacts (e.g., ",nd" -> "command")
        text = self._correct_whisper_artifacts(text)
        
        # Pass 2: Normalize fancy characters
        for fancy, simple in self.CHAR_NORMALIZE.items():
            text = text.replace(fancy, simple)
        
        # Pass 3: Custom word replacements
        for word, replacement in self.REPLACEMENTS.items():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub(replacement, text)
        
        # Pass 4: Convert spoken punctuation to symbols
        # Only replace when it's a standalone word
        for spoken, punct in self.PUNCTUATION.items():
            pattern = re.compile(r'\b' + re.escape(spoken) + r'\b', re.IGNORECASE)
            text = pattern.sub(punct, text)
        
        # Pass 5: Clean up spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


class SoundEffects:
    @staticmethod
    def play_start():
        import winsound
        winsound.Beep(800, 150)
    
    @staticmethod
    def play_stop():
        import winsound
        winsound.Beep(400, 150)
    
    @staticmethod
    def play_error():
        import winsound
        winsound.Beep(200, 300)


class SystemTray:
    def __init__(self, app: "CognitiveFlowApp"):
        self.app = app
        self.icon = None
    
    def create_icon(self, recording: bool = False, loading: bool = False) -> "Image.Image":
        size = 64
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        if loading:
            draw.ellipse([4, 4, size-4, size-4], fill=(255, 200, 0, 255))
        elif recording:
            draw.ellipse([4, 4, size-4, size-4], fill=(255, 50, 50, 255))
        else:
            draw.ellipse([4, 4, size-4, size-4], fill=(50, 200, 50, 255))
        
        return img
    
    def update_icon(self, recording: bool = False, loading: bool = False):
        if self.icon:
            self.icon.icon = self.create_icon(recording, loading)
    
    def on_quit(self, icon, item):
        print("\n[Exit] Quit from system tray...")
        self.app.running = False
        
        # Stop icon first
        try:
            icon.visible = False
            icon.stop()
        except:
            pass
        
        # Unhook keyboard
        try:
            if self.app.hook:
                UnhookWindowsHookEx(self.app.hook)
                self.app.hook = None
        except:
            pass
        
        # Terminate audio
        try:
            if self.app.audio:
                self.app.audio.terminate()
        except:
            pass
        
        # Post quit to Windows message loop
        try:
            ctypes.windll.user32.PostQuitMessage(0)
        except:
            pass
        
        print("[Exit] Goodbye!")
        
        # Force immediate exit - use sys.exit in a thread to ensure it runs
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
    
    def on_show_stats(self, icon, item):
        if self.app.ui:
            self.app.ui.show_settings()
        else:
            stats = self.app.stats.summary()
            ctypes.windll.user32.MessageBoxW(None, stats, "Cognitive Flow Stats", 0x40)
    
    def run(self):
        if not HAS_TRAY:
            return
        
        menu = pystray.Menu(
            pystray.MenuItem("Settings", self.on_show_stats),
            pystray.MenuItem("Quit", self.on_quit)
        )
        
        self.icon = pystray.Icon(
            "CognitiveFlow",
            self.create_icon(loading=True),
            "Cognitive Flow - Loading...",
            menu
        )
        
        threading.Thread(target=self.icon.run, daemon=True).start()


class VirtualKeyboard:
    # Characters that can break terminals or trigger unwanted behavior
    DANGEROUS_CHARS = {
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',  # Control chars
        '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12',
        '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
        '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',  # Escape and more control
        '\x7f',  # DEL
        '`',  # Backtick - can trigger terminal escapes
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # BOM / zero-width no-break space
        '\u00ad',  # Soft hyphen
        '\u2028',  # Line separator
        '\u2029',  # Paragraph separator
    }
    
    # Characters to replace with safe equivalents
    CHAR_REPLACEMENTS = {
        '`': "'",           # Backtick -> single quote
        '\u00a0': ' ',      # Non-breaking space -> regular space
        '\u2018': "'",      # Left single quote
        '\u2019': "'",      # Right single quote  
        '\u201c': '"',      # Left double quote
        '\u201d': '"',      # Right double quote
        '\u2013': '-',      # En dash
        '\u2014': '-',      # Em dash
        '\u2026': '...',    # Ellipsis
    }
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove or replace characters that can break terminals."""
        # First, apply replacements
        for bad, good in VirtualKeyboard.CHAR_REPLACEMENTS.items():
            text = text.replace(bad, good)
        # Then remove dangerous chars
        text = ''.join(c for c in text if c not in VirtualKeyboard.DANGEROUS_CHARS)
        # Finally, ensure only printable ASCII + common extended chars
        # Keep: printable ASCII (32-126), newline, tab, and basic extended Latin
        return ''.join(c for c in text if (32 <= ord(c) <= 126) or c in '\n\t' or (192 <= ord(c) <= 255))
    
    @staticmethod
    def type_text(text: str):
        if not text:
            return
        
        # Sanitize before typing
        text = VirtualKeyboard.sanitize_text(text)
        
        BATCH_SIZE = 100
        BATCH_DELAY = 0
        
        for i in range(0, len(text), BATCH_SIZE):
            batch = text[i:i + BATCH_SIZE]
            VirtualKeyboard._type_batch(batch)
            if i + BATCH_SIZE < len(text):
                time.sleep(BATCH_DELAY)
    
    @staticmethod
    def _type_batch(text: str):
        num_chars = len(text)
        if num_chars == 0:
            return
        
        inputs = (INPUT * (num_chars * 2))()
        
        for i, char in enumerate(text):
            idx = i * 2
            
            inputs[idx].type = INPUT_KEYBOARD
            inputs[idx].union.ki.wVk = 0
            inputs[idx].union.ki.wScan = ord(char)
            inputs[idx].union.ki.dwFlags = KEYEVENTF_UNICODE
            inputs[idx].union.ki.time = 0
            inputs[idx].union.ki.dwExtraInfo = None
            
            inputs[idx + 1].type = INPUT_KEYBOARD
            inputs[idx + 1].union.ki.wVk = 0
            inputs[idx + 1].union.ki.wScan = ord(char)
            inputs[idx + 1].union.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
            inputs[idx + 1].union.ki.time = 0
            inputs[idx + 1].union.ki.dwExtraInfo = None
        
        SendInput(num_chars * 2, inputs, ctypes.sizeof(INPUT))


class CognitiveFlowApp:
    """Main application"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.is_recording = False
        self.frames: list[bytes] = []
        self.record_start_time: float = 0.0
        self.audio_queue = queue.Queue()
        
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.audio = pyaudio.PyAudio()
        
        self.stats = Statistics()
        self.processor = TextProcessor()
        self.model: WhisperModel | None = None
        self.model_loading = False
        self.using_gpu = False
        
        # For retry functionality
        self.last_audio = None
        self.last_duration = 0.0
        
        self.tray: SystemTray | None = None
        if HAS_TRAY:
            self.tray = SystemTray(self)
        
        self.ui = None
        if HAS_UI:
            self.ui = CognitiveFlowUI(self)
        
        self.hook = None
        self.hook_proc = None
        self.running = True
        
        if self.debug:
            logger.info("Stats", self.stats.summary())
            logger.separator()
            logger.info("Controls", "~ (tilde) - Start/stop recording")
            logger.info("Controls", "Tray > Quit - Exit application")
            logger.info("Controls", "Right-click - Open settings")
            logger.separator()
        
        from .paths import CONFIG_FILE
        self.config_file = CONFIG_FILE
        self._load_config()
        if self.debug:
            logger.info("Model", f"Loading Whisper model ({self.model_name})...")
    
    def _load_config(self):
        # Defaults
        self.model_name = 'medium'
        self.add_trailing_space = True  # Add space after each transcription
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.model_name = config.get('model_name', 'medium')
                    self.add_trailing_space = config.get('add_trailing_space', True)
            except:
                pass
    
    def save_config(self):
        config = {
            'model_name': self.model_name,
            'add_trailing_space': self.add_trailing_space,
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[Config] Saved preferences")
    
    def _log_transcription(self, raw_text: str, processed_text: str):
        """Log transcription details to file for debugging terminal breaks."""
        from .paths import DEBUG_LOG_FILE
        from datetime import datetime
        
        def char_dump(s: str) -> str:
            """Create hex dump of characters."""
            return ' '.join(f'{ord(c):02x}' for c in s)
        
        def flag_suspicious(s: str) -> list:
            """Flag any non-standard ASCII characters."""
            suspicious = []
            for i, c in enumerate(s):
                code = ord(c)
                if code < 32 and c not in '\n\t':
                    suspicious.append(f"  pos {i}: CONTROL char {code:02x}")
                elif code > 126 and code < 192:
                    suspicious.append(f"  pos {i}: EXTENDED char {code:02x} '{c}'")
                elif code > 255:
                    suspicious.append(f"  pos {i}: UNICODE {code:04x} '{c}'")
            return suspicious
        
        try:
            with open(DEBUG_LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Raw text ({len(raw_text)} chars):\n")
                f.write(f"  {raw_text!r}\n")
                f.write(f"Processed text ({len(processed_text)} chars):\n")
                f.write(f"  {processed_text!r}\n")
                
                suspicious = flag_suspicious(raw_text)
                if suspicious:
                    f.write(f"SUSPICIOUS CHARS IN RAW:\n")
                    f.write('\n'.join(suspicious) + '\n')
                
                suspicious = flag_suspicious(processed_text)
                if suspicious:
                    f.write(f"SUSPICIOUS CHARS IN PROCESSED:\n")
                    f.write('\n'.join(suspicious) + '\n')
                    
        except Exception as e:
            if self.debug:
                print(f"[Debug] Failed to log transcription: {e}")
    
    def load_model(self):
        self.model_loading = True
        
        def _load():
            try:
                if GPU_AVAILABLE:
                    if self.debug:
                        print(f"[GPU] Loading {self.model_name} model on CUDA...")
                    try:
                        self.model = WhisperModel(
                            self.model_name,
                            device="cuda",
                            compute_type="float32",
                            device_index=0
                        )
                        self.using_gpu = True
                        if self.debug:
                            print(f"[GPU] Model loaded successfully!")
                        
                        # Warmup
                        if self.debug:
                            print("[GPU] Warming up model...")
                        import numpy as np
                        warmup_audio = tempfile.mktemp(suffix=".wav")
                        samples = np.zeros(16000, dtype=np.int16)
                        import wave as wav_module
                        wf = wav_module.open(warmup_audio, 'wb')
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(samples.tobytes())
                        wf.close()
                        list(self.model.transcribe(warmup_audio, beam_size=1, language="en"))
                        os.unlink(warmup_audio)
                        if self.debug:
                            print("[GPU] Warmup complete")
                            print(f"[Ready] Model: {self.model_name} (GPU)")
                        
                        if self.tray:
                            self.tray.update_icon(recording=False, loading=False)
                            self.tray.icon.title = "Cognitive Flow - Ready (GPU)"
                        if self.ui:
                            self.ui.set_state("idle", "Ready (GPU)")
                        return
                    except Exception as gpu_err:
                        if self.debug:
                            print(f"[GPU] Failed: {gpu_err}")
                
                if self.debug:
                    print(f"[CPU] Loading {self.model_name} model...")
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                    num_workers=1
                )
                self.using_gpu = False
                if self.debug:
                    print(f"[Ready] Model: {self.model_name} (CPU)")
                if self.tray:
                    self.tray.update_icon(recording=False, loading=False)
                    self.tray.icon.title = "Cognitive Flow - Ready (CPU)"
                if self.ui:
                    self.ui.set_state("idle", "Ready (CPU)")
                    
            except Exception as e:
                print(f"[Error] Failed to load: {e}")
                try:
                    self.model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
                    self.model_name = "base"
                    print("[Ready] Using base model (CPU)")
                except Exception as e2:
                    print(f"[Fatal] Could not load any model: {e2}")
            finally:
                self.model_loading = False
        
        threading.Thread(target=_load, daemon=True).start()
    
    def keyboard_callback(self, nCode, wParam, lParam):
        if nCode >= 0:
            kb = lParam.contents
            if kb.vkCode == VK_OEM_3:
                if wParam == WM_KEYDOWN:
                    self.toggle_recording()
                    return 1
        return CallNextHookEx(self.hook, nCode, wParam, lParam)
    
    def install_hook(self):
        ctypes.windll.kernel32.SetLastError(0)
        self.hook_proc = HOOKPROC(self.keyboard_callback)
        self.hook = SetWindowsHookExW(WH_KEYBOARD_LL, self.hook_proc, None, 0)
        
        if not self.hook:
            error = ctypes.windll.kernel32.GetLastError()
            raise RuntimeError(f"Failed to install keyboard hook (error: {error})")
        
        print("[Hook] Global keyboard hook installed")
    
    def message_loop(self):
        msg = ctypes.wintypes.MSG()
        PeekMessageW = user32.PeekMessageW
        TranslateMessage = user32.TranslateMessage
        DispatchMessageW = user32.DispatchMessageW
        PM_REMOVE = 0x0001
        
        while self.running:
            if self.ui and self.ui.qt_app:
                self.ui.qt_app.processEvents()
            
            if PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == 0x0012:
                    break
                TranslateMessage(ctypes.byref(msg))
                DispatchMessageW(ctypes.byref(msg))
            else:
                time.sleep(0.001)
    
    def toggle_recording(self):
        if self.model_loading:
            logger.warning("Record", "Model still loading...")
            return
        
        if not self.model:
            logger.error("Record", "Model not loaded")
            SoundEffects.play_error()
            return
        
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.record_start_time = time.time()
        
        SoundEffects.play_start()
        if self.debug:
            logger.info("Record", "Listening...")
        
        if self.tray:
            self.tray.update_icon(recording=True)
        if self.ui:
            self.ui.set_state("recording", "Recording...")
        
        def _record():
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            while self.is_recording:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"[Error] Recording: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
        
        threading.Thread(target=_record, daemon=True).start()
    
    def stop_recording(self):
        self.is_recording = False
        time.sleep(0.1)  # Give recording thread time to finish last read
        duration = time.time() - self.record_start_time
        
        SoundEffects.play_stop()
        if self.debug:
            logger.info("Processing", f"{duration:.1f}s of audio...")
        
        if self.tray:
            self.tray.update_icon(recording=False)
        if self.ui:
            self.ui.set_state("processing", "Processing...")
        
        def _transcribe():
            try:
                import numpy as np
                start_time = time.time()
                
                audio_data = b''.join(self.frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Store for retry
                self.last_audio = audio_array
                self.last_duration = duration
                
                # Check for audio issues
                max_amp = np.max(np.abs(audio_array))
                if max_amp < 0.01:
                    logger.warning("Audio", "No audio detected - check microphone")
                    SoundEffects.play_error()
                    if self.ui:
                        self.ui.set_state("idle", "No audio!")
                        threading.Timer(2.0, lambda: self.ui and self.ui.set_state("idle", "Ready")).start()
                    return
                
                if self.model is None:
                    raise RuntimeError("Model not loaded")
                
                segments, info = self.model.transcribe(
                    audio_array,
                    beam_size=5,
                    language="en",
                    vad_filter=False,
                    word_timestamps=False,
                    condition_on_previous_text=True,
                    no_speech_threshold=0.9,
                    hallucination_silence_threshold=None,
                )
                
                segment_list = list(segments)
                segment_texts = [seg.text.strip() for seg in segment_list]
                raw_text = " ".join(segment_texts)
                
                if self.debug:
                    logger.info("Raw", f'Whisper output: "{raw_text}"')
                
                processed_text = self.processor.process(raw_text)
                
                if self.debug and processed_text != raw_text:
                    logger.info("Processed", f'After cleanup: "{processed_text}"')
                
                # Log to file for debugging terminal breaks
                self._log_transcription(raw_text, processed_text)
                
                if processed_text:
                    total_pipeline = time.time() - start_time
                    
                    self.stats.record(duration, processed_text, total_pipeline)
                    
                    if self.ui:
                        self.ui.add_transcription(processed_text, duration)
                    
                    # Add trailing space if enabled (helps separate consecutive transcriptions)
                    output_text = processed_text + " " if self.add_trailing_space else processed_text
                    VirtualKeyboard.type_text(output_text)
                    
                    # Rich logging
                    words = len(processed_text.split())
                    chars = len(processed_text)
                    speed_ratio = duration / total_pipeline if total_pipeline > 0 else 0
                    preview = processed_text[:60] + "..." if len(processed_text) > 60 else processed_text
                    preview = preview.replace('\n', ' ')  # Single line preview
                    
                    if self.debug:
                        # Verbose debug output
                        logger.success("Done", f"{words} words, {chars} chars in {total_pipeline:.2f}s ({speed_ratio:.1f}x realtime)")
                        logger.info("Text", f'"{preview}"')
                    else:
                        # Concise but useful
                        logger.success("Typed", f'{words}w/{chars}c in {total_pipeline:.1f}s | "{preview}"')
                    
                    if self.ui:
                        self.ui.set_state("idle", f"{words} words")
                        threading.Timer(2.0, lambda: self.ui and self.ui.set_state("idle", "Ready")).start()
                else:
                    logger.warning("Audio", "No speech detected")
                    if self.ui:
                        self.ui.set_state("idle", "Ready")
                
            except Exception as e:
                print(f"[Error] Transcription failed: {e}")
                SoundEffects.play_error()
                if self.ui:
                    self.ui.set_state("idle", "Ready")
        
        threading.Thread(target=_transcribe, daemon=True).start()
    
    def run(self):
        if self.ui:
            self.ui.start()
            self.ui.set_state("loading")
        
        if self.tray:
            self.tray.run()
        
        self.load_model()
        self.install_hook()
        
        try:
            self.message_loop()
        except KeyboardInterrupt:
            print("\n[Exit] Shutting down...")
        finally:
            if self.hook:
                UnhookWindowsHookEx(self.hook)
            if self.tray and self.tray.icon:
                self.tray.icon.stop()
            if self.ui:
                self.ui.destroy()
            self.audio.terminate()
            print("[Goodbye]")


def main():
    import signal
    import argparse
    import subprocess
    from . import __version__
    
    parser = argparse.ArgumentParser(description="Cognitive Flow - Local voice-to-text")
    parser.add_argument("--debug", action="store_true", help="Run in foreground with debug output")
    parser.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)  # Internal flag
    args = parser.parse_args()
    
    # If not debug and not already in foreground, respawn detached and exit
    if not args.debug and not args.foreground:
        # Spawn detached process with pythonw (no console)
        subprocess.Popen(
            ["pythonw", "-m", "cognitive_flow", "--foreground"],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
            close_fds=True
        )
        print("Cognitive Flow started in background")
        return
    
    # Debug mode: show banner
    if args.debug:
        print("=" * 60)
        print(f"  Cognitive Flow v{__version__}")
        print("=" * 60)
        print()
        print("  CHANGELOG:")
        print("    v1.2.0 - Debug logging for terminal break investigation")
        print("           - Logs to %APPDATA%\\CognitiveFlow\\debug_transcriptions.log")
        print("           - Enhanced character sanitization")
        print("           - beam_size=5 for better accuracy")
        print("    v1.1.0 - GUI mode by default, --debug for console")
        print("           - Fixed audio capture race condition") 
        print("    v1.0.0 - Initial release as proper package")
        print()
        print("=" * 60)
    
    # Now load all the heavy libraries
    init_app()
    
    app = CognitiveFlowApp(debug=args.debug)
    
    def handle_sigint(sig, frame):
        print("\n[Exit] Ctrl+C received...")
        app.running = False
        if app.hook:
            UnhookWindowsHookEx(app.hook)
            app.hook = None
        if app.tray and app.tray.icon:
            try:
                app.tray.icon.stop()
            except:
                pass
        if app.ui:
            app.ui.destroy()
        app.audio.terminate()
        print("[Goodbye]")
        os._exit(0)
    
    signal.signal(signal.SIGINT, handle_sigint)
    
    app.run()
