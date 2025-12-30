"""
Cognitive Flow - Local voice-to-text with global hotkey
Press tilde (~) to start/stop recording, transcribes and types into focused app
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

# Import our logger
from logger import logger

# Add CUDA libraries for GPU support (MUST be before importing faster_whisper)
# Pre-load ALL DLLs explicitly - os.add_dll_directory alone doesn't work reliably
GPU_AVAILABLE = False
try:
    import site
    user_site = site.USER_SITE
    if user_site:
        cuda_path = Path(user_site) / "nvidia"
        if cuda_path.exists():
            cudnn_bin = cuda_path / "cudnn" / "bin"
            cublas_bin = cuda_path / "cublas" / "bin"
            
            # Add to DLL search paths
            if cudnn_bin.exists():
                os.add_dll_directory(str(cudnn_bin))
            if cublas_bin.exists():
                os.add_dll_directory(str(cublas_bin))
            
            # Add to PATH
            os.environ['PATH'] = f"{cudnn_bin};{cublas_bin};" + os.environ.get('PATH', '')
            
            # Pre-load ALL DLLs explicitly (critical for ctranslate2 to find them)
            dll_count = 0
            for dll_dir in [cublas_bin, cudnn_bin]:
                if dll_dir.exists():
                    for dll in dll_dir.glob("*.dll"):
                        try:
                            ctypes.CDLL(str(dll))
                            dll_count += 1
                        except Exception:
                            pass  # Some DLLs have dependencies, skip failures
            
            if dll_count > 0:
                GPU_AVAILABLE = True
                print(f"[CUDA] Loaded {dll_count} GPU libraries")
except Exception as e:
    print(f"[CUDA] GPU setup failed: {e} - using CPU")
    GPU_AVAILABLE = False

import pyaudio
from faster_whisper import WhisperModel

# Optional: pystray for system tray (graceful fallback if not installed)
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False
    print("[Note] Install pystray and pillow for system tray: pip install pystray pillow")

# UI module - try PyQt first, fallback to tkinter
HAS_UI = False
CognitiveFlowUI = None

try:
    from cognitive_flow_ui_qt import CognitiveFlowUI
    HAS_UI = True
    print("[UI] Using PyQt6 (smooth graphics)")
except ImportError:
    try:
        from cognitive_flow_ui import CognitiveFlowUI
        HAS_UI = True
        print("[UI] Using tkinter (basic graphics)")
    except ImportError:
        print("[Note] No UI module found, running in console mode")

# Windows API constants
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_OEM_3 = 0xC0  # Tilde/backtick key

# For virtual keyboard input
INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002

# Windows structures for keyboard hook
class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.wintypes.DWORD),
        ("scanCode", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG))
    ]

# For SendInput
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

# Windows API functions
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

SetWindowsHookExW = user32.SetWindowsHookExW
UnhookWindowsHookEx = user32.UnhookWindowsHookEx
CallNextHookEx = user32.CallNextHookEx
GetMessageW = user32.GetMessageW
SendInput = user32.SendInput
GetForegroundWindow = user32.GetForegroundWindow

# Hook procedure type
HOOKPROC = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.POINTER(KBDLLHOOKSTRUCT))


class Statistics:
    """Track usage statistics, ported from WhisperTyping"""
    
    def __init__(self, stats_file: str | None = None):
        if stats_file is None:
            self.stats_file = Path(__file__).parent / "statistics.json"
        else:
            self.stats_file = Path(stats_file)
        
        self.stats: dict = self._load_stats()
    
    def _load_stats(self) -> dict:
        """Load stats, importing from WhisperTyping if this is first run"""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        
        # Try to import from WhisperTyping
        wt_stats = Path(os.environ['LOCALAPPDATA']) / "WhisperTyping" / "statistics.json"
        if wt_stats.exists():
            print(f"[Stats] Importing from WhisperTyping...")
            with open(wt_stats, 'r') as f:
                imported = json.load(f)
            # Convert to our format
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
        
        # Fresh start with enhanced metrics
        return {
            "total_seconds": 0,
            "total_records": 0,
            "total_words": 0,
            "total_characters": 0,
            "last_used": "",
            "imported_from_whispertyping": False,
            "total_processing_time": 0,  # Total time spent processing
            "session_stats": {
                "records": 0,
                "words": 0,
                "characters": 0,
                "processing_time": 0
            },
            "performance_history": []  # Last 100 transcriptions
        }
    
    def _save(self, stats: dict | None = None):
        if stats is None:
            stats = self.stats
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def record(self, duration_seconds: float, text: str, processing_time: float = 0):
        """Record a transcription with enhanced metrics"""
        words = len(text.split())
        chars = len(text)
        
        # Update totals
        self.stats["total_seconds"] += duration_seconds
        self.stats["total_records"] += 1
        self.stats["total_words"] += words
        self.stats["total_characters"] += chars
        self.stats["total_processing_time"] = self.stats.get("total_processing_time", 0) + processing_time
        self.stats["last_used"] = datetime.now().isoformat()
        
        # Update session stats
        if "session_stats" not in self.stats:
            self.stats["session_stats"] = {"records": 0, "words": 0, "characters": 0, "processing_time": 0}
        
        self.stats["session_stats"]["records"] += 1
        self.stats["session_stats"]["words"] += words
        self.stats["session_stats"]["characters"] += chars
        self.stats["session_stats"]["processing_time"] += processing_time
        
        # Track performance history (last 100)
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
        """Calculate time saved using YOUR actual typing speed (30 WPM)"""
        words = self.stats["total_words"]
        audio_seconds = self.stats["total_seconds"]
        
        # Your actual typing speed: 30 WPM (from test)
        typing_time_minutes = words / 30
        
        # Actual speaking time (from recorded audio)
        speaking_time_minutes = audio_seconds / 60
        
        # Time saved = what typing would have taken - what speaking took
        saved_minutes = typing_time_minutes - speaking_time_minutes
        
        if saved_minutes < 60:
            return f"{saved_minutes:.0f} minutes"
        else:
            hours = saved_minutes / 60
            return f"{hours:.1f} hours"
    
    def get_typing_vs_speaking_comparison(self) -> dict:
        """Get detailed comparison of typing time vs speaking time"""
        words = self.stats["total_words"]
        audio_seconds = self.stats["total_seconds"]
        
        typing_time_minutes = words / 30  # Your 30 WPM
        speaking_time_minutes = audio_seconds / 60
        
        return {
            "typing_time": typing_time_minutes,
            "speaking_time": speaking_time_minutes,
            "time_saved": typing_time_minutes - speaking_time_minutes,
            "efficiency_ratio": typing_time_minutes / speaking_time_minutes if speaking_time_minutes > 0 else 0
        }
    
    def get_speaking_speed_wpm(self) -> float:
        """Calculate your average speaking speed in words per minute"""
        words = self.stats["total_words"]
        audio_seconds = self.stats["total_seconds"]
        
        if audio_seconds == 0:
            return 0
        
        minutes = audio_seconds / 60
        return words / minutes if minutes > 0 else 0
    
    def get_seconds_per_word(self) -> float:
        """Calculate average seconds per word when speaking"""
        words = self.stats["total_words"]
        audio_seconds = self.stats["total_seconds"]
        
        return audio_seconds / words if words > 0 else 0
    
    def summary(self) -> str:
        """Get a summary string"""
        return (
            f"Recordings: {self.stats['total_records']:,} | "
            f"Words: {self.stats['total_words']:,} | "
            f"Time saved: ~{self.get_time_saved()}"
        )
    
    def get_avg_speed_ratio(self) -> float:
        """Get average processing speed ratio"""
        history = self.stats.get("performance_history", [])
        if not history:
            return 0
        ratios = [h["speed_ratio"] for h in history if h.get("speed_ratio", 0) > 0]
        return sum(ratios) / len(ratios) if ratios else 0
    
    def get_avg_words_per_recording(self) -> float:
        """Get average words per recording"""
        records = self.stats.get("total_records", 0)
        if records == 0:
            return 0
        return self.stats.get("total_words", 0) / records
    
    def reset_session_stats(self):
        """Reset session statistics"""
        self.stats["session_stats"] = {
            "records": 0,
            "words": 0,
            "characters": 0,
            "processing_time": 0
        }
        self._save()


class TextProcessor:
    """Process transcribed text - punctuation, replacements, etc."""
    
    # Spoken punctuation mappings
    PUNCTUATION = {
        "period": ".",
        "full stop": ".",
        "comma": ",",
        "question mark": "?",
        "exclamation mark": "!",
        "exclamation point": "!",
        "colon": ":",
        "semicolon": ";",
        "semi colon": ";",
        "dash": "-",
        "hyphen": "-",
        "open parenthesis": "(",
        "close parenthesis": ")",
        "open bracket": "[",
        "close bracket": "]",
        "open brace": "{",
        "close brace": "}",
        "apostrophe": "'",
        "quote": '"',
        "open quote": '"',
        "close quote": '"',
        "ellipsis": "...",
        "new line": "\n",
        "newline": "\n",
        "new paragraph": "\n\n",
        "enter": "\n",
    }
    
    # Custom replacements (from WhisperTyping settings)
    REPLACEMENTS = {
        "hashtag": "#",
        "clod": "CLAUDE",
    }
    
    def process(self, text: str) -> str:
        """Process transcribed text"""
        if not text:
            return text
        
        # Apply custom replacements first (case-insensitive)
        for word, replacement in self.REPLACEMENTS.items():
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub(replacement, text)
        
        # Apply spoken punctuation
        for spoken, punct in self.PUNCTUATION.items():
            # Replace spoken punctuation with actual punctuation
            # Handle "word period" -> "word."
            text = text.replace(f" {spoken}", punct)
            text = text.replace(f"{spoken} ", f"{punct} ")
            # Handle at end of text
            if text.lower().endswith(spoken):
                text = text[:-len(spoken)] + punct
        
        return text.strip()


class SoundEffects:
    """Simple sound effects using winsound"""
    
    @staticmethod
    def play_start():
        """Play start recording sound"""
        import winsound
        # High beep for start
        winsound.Beep(800, 150)
    
    @staticmethod
    def play_stop():
        """Play stop recording sound"""
        import winsound
        # Lower beep for stop
        winsound.Beep(400, 150)
    
    @staticmethod
    def play_error():
        """Play error sound"""
        import winsound
        winsound.Beep(200, 300)


class SystemTray:
    """System tray icon with status indicator"""
    
    def __init__(self, app: "WhisperTypingApp"):
        self.app = app
        self.icon = None
        
    def create_icon(self, recording: bool = False, loading: bool = False) -> "Image.Image":
        """Create a tray icon image"""
        size = 64
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        if loading:
            # Yellow circle while loading
            draw.ellipse([4, 4, size-4, size-4], fill=(255, 200, 0, 255))
        elif recording:
            # Red circle when recording
            draw.ellipse([4, 4, size-4, size-4], fill=(255, 50, 50, 255))
        else:
            # Green circle when ready
            draw.ellipse([4, 4, size-4, size-4], fill=(50, 200, 50, 255))
        
        return img
    
    def update_icon(self, recording: bool = False, loading: bool = False):
        """Update the tray icon state"""
        if self.icon:
            self.icon.icon = self.create_icon(recording, loading)
    
    def on_quit(self, icon, item):
        """Handle quit from tray menu"""
        import sys
        import os
        import threading
        
        print("\n[Exit] Quit from system tray - starting cleanup...")
        self.app.running = False
        
        # Stop icon first (this is blocking the exit)
        print("[Exit] Stopping tray icon...")
        try:
            icon.visible = False
            icon.stop()
        except Exception as e:
            print(f"[Exit] Tray icon error: {e}")
        
        # Unhook keyboard
        print("[Exit] Removing keyboard hook...")
        try:
            if self.app.hook:
                UnhookWindowsHookEx(self.app.hook)
                self.app.hook = None
        except Exception as e:
            print(f"[Exit] Hook error: {e}")
        
        # Close Qt UI - SKIP GRACEFUL SHUTDOWN, just force exit
        print("[Exit] Closing Qt UI...")
        # Don't even try to clean up Qt - it hangs
        # Just let os._exit() kill everything
        
        # Terminate audio
        print("[Exit] Terminating audio...")
        try:
            if self.app.audio:
                self.app.audio.terminate()
        except Exception as e:
            print(f"[Exit] Audio error: {e}")
        
        # Post quit to Windows message loop
        print("[Exit] Posting quit message...")
        try:
            ctypes.windll.user32.PostQuitMessage(0)
        except Exception as e:
            print(f"[Exit] PostQuit error: {e}")
        
        # Force exit NOW - no waiting
        print("[Exit] Goodbye!")
        os._exit(0)  # Nuclear option - immediate exit
    
    def on_show_stats(self, icon, item):
        """Show settings dialog (not old MessageBox)"""
        if self.app.ui:
            self.app.ui.show_settings()
        else:
            # Fallback to old message box if no UI
            stats = self.app.stats.summary()
            ctypes.windll.user32.MessageBoxW(
                None, 
                stats, 
                "Cognitive Flow Stats", 
                0x40  # MB_ICONINFORMATION
            )
    
    def run(self):
        """Run the system tray icon"""
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
        
        # Run in background thread
        threading.Thread(target=self.icon.run, daemon=True).start()


class VirtualKeyboard:
    """Type text using Windows SendInput API - works in admin windows"""
    
    @staticmethod
    def type_text(text: str):
        """Type text character by character using SendInput - FAST"""
        # Batch characters for speed (no delay between chars)
        for char in text:
            VirtualKeyboard._send_unicode_char(char)
            # NO SLEEP - let it rip at full speed
    
    @staticmethod
    def _send_unicode_char(char: str):
        """Send a single unicode character"""
        # Key down
        inputs = (INPUT * 2)()
        
        inputs[0].type = INPUT_KEYBOARD
        inputs[0].union.ki.wVk = 0
        inputs[0].union.ki.wScan = ord(char)
        inputs[0].union.ki.dwFlags = KEYEVENTF_UNICODE
        inputs[0].union.ki.time = 0
        inputs[0].union.ki.dwExtraInfo = None
        
        # Key up
        inputs[1].type = INPUT_KEYBOARD
        inputs[1].union.ki.wVk = 0
        inputs[1].union.ki.wScan = ord(char)
        inputs[1].union.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
        inputs[1].union.ki.time = 0
        inputs[1].union.ki.dwExtraInfo = None
        
        SendInput(2, inputs, ctypes.sizeof(INPUT))


class WhisperTypingApp:
    """Main application - global hotkey, recording, transcription, typing"""
    
    def __init__(self):
        self.is_recording = False
        self.frames: list[bytes] = []
        self.record_start_time: float = 0.0
        self.audio_queue = queue.Queue()
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.audio = pyaudio.PyAudio()
        
        # Components
        self.stats = Statistics()
        self.processor = TextProcessor()
        self.model: WhisperModel | None = None
        self.model_loading = False
        self.using_gpu = False
        
        # System tray
        self.tray: SystemTray | None = None
        if HAS_TRAY:
            self.tray = SystemTray(self)
        
        # UI
        self.ui = None
        if HAS_UI:
            self.ui = CognitiveFlowUI(self)
        
        # Hook handle
        self.hook = None
        self.hook_proc = None  # Must keep reference to prevent GC
        
        # For clean shutdown
        self.running = True
        
        logger.header("Cognitive Flow - Local Voice-to-Text")
        logger.info("Stats", self.stats.summary())
        logger.separator()
        logger.info("Controls", "~ (tilde) - Start/stop recording")
        logger.info("Controls", "Tray > Quit - Exit application")
        logger.info("Controls", "Right-click - Open settings")
        logger.separator()
        
        # Load config (model preference)
        self.config_file = Path(__file__).parent / "config.json"
        self.model_name = self._load_config()
        logger.info("Model", f"Loading Whisper model ({self.model_name})...")
    
    def _load_config(self) -> str:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('model_name', 'medium')
            except:
                pass
        return 'medium'
    
    def save_config(self):
        """Save configuration to file"""
        config = {'model_name': self.model_name}
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[Config] Saved model preference: {self.model_name}")
    
    def load_model(self):
        """Load Whisper model in background - GPU if available, CPU fallback"""
        self.model_loading = True
        
        def _load():
            try:
                # Try GPU first if available
                if GPU_AVAILABLE:
                    print(f"[GPU] Loading {self.model_name} model on CUDA...")
                    try:
                        self.model = WhisperModel(
                            self.model_name,
                            device="cuda",
                            compute_type="float32",  # Most compatible, still fast
                            device_index=0
                        )
                        self.using_gpu = True
                        print(f"[GPU] Model loaded successfully!")
                        print(f"[Ready] Model: {self.model_name} (GPU) | Press ~ to record")
                        print(f"[Stats] {self.stats.summary()}")
                        if self.tray:
                            self.tray.update_icon(recording=False, loading=False)
                            self.tray.icon.title = "Cognitive Flow - Ready (GPU)"
                        if self.ui:
                            self.ui.set_state("idle", "Ready (GPU) - Press ~ to record")
                        return
                    except Exception as gpu_err:
                        print(f"[GPU] Failed: {gpu_err}")
                        print("[GPU] Falling back to CPU...")
                
                # CPU fallback
                print(f"[CPU] Loading {self.model_name} model on CPU...")
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                    num_workers=1
                )
                self.using_gpu = False
                print("[CPU] Model loaded successfully!")
                print(f"[Ready] Model: {self.model_name} (CPU) | Press ~ to record")
                print(f"[Stats] {self.stats.summary()}")
                if self.tray:
                    self.tray.update_icon(recording=False, loading=False)
                    self.tray.icon.title = "Cognitive Flow - Ready (CPU)"
                if self.ui:
                    self.ui.set_state("idle", "Ready (CPU) - Press ~ to record")
                    
            except Exception as e:
                print(f"[Error] Failed to load {self.model_name}: {e}")
                print("[Fallback] Trying 'base' model on CPU...")
                try:
                    self.model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
                    self.model_name = "base"
                    self.using_gpu = False
                    print("[Ready] Using base model (CPU). Press ~ to record")
                except Exception as e2:
                    print(f"[Fatal] Could not load any model: {e2}")
            finally:
                self.model_loading = False
        
        threading.Thread(target=_load, daemon=True).start()
    
    def keyboard_callback(self, nCode, wParam, lParam):
        """Low-level keyboard hook callback"""
        if nCode >= 0:
            kb = lParam.contents
            
            # Check for tilde key (OEM_3)
            if kb.vkCode == VK_OEM_3:
                if wParam == WM_KEYDOWN:
                    # Toggle recording on key down
                    self.toggle_recording()
                    # Return 1 to block the key from reaching other apps
                    return 1
        
        return CallNextHookEx(self.hook, nCode, wParam, lParam)
    
    def install_hook(self):
        """Install the low-level keyboard hook"""
        # Need to use SetLastError pattern for proper error reporting
        ctypes.windll.kernel32.SetLastError(0)
        
        self.hook_proc = HOOKPROC(self.keyboard_callback)
        
        # For low-level hooks, hMod should be NULL (0) - this works better with Python
        # Error 126 means GetModuleHandleW returned something invalid
        self.hook = SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self.hook_proc,
            None,  # NULL for low-level hooks
            0
        )
        
        if not self.hook:
            error = ctypes.windll.kernel32.GetLastError()
            print(f"[Error] Failed to install keyboard hook (error code: {error})")
            print("[Note] This app must be run from a terminal window, not as a subprocess")
            print("[Tip] Open a new terminal and run: python cognitive_flow.py")
            raise RuntimeError(f"Failed to install keyboard hook (error: {error})")
        
        print("[Hook] Global keyboard hook installed")
    
    def message_loop(self):
        """
        Hybrid Windows + Qt message loop
        Process Windows messages (for keyboard hook) AND Qt events (for UI)
        """
        msg = ctypes.wintypes.MSG()
        PeekMessageW = user32.PeekMessageW
        TranslateMessage = user32.TranslateMessage
        DispatchMessageW = user32.DispatchMessageW
        PM_REMOVE = 0x0001
        
        while self.running:
            # Process Qt events (non-blocking)
            if self.ui and self.ui.qt_app:
                self.ui.qt_app.processEvents()
            
            # Check for Windows messages (non-blocking peek)
            if PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == 0x0012:  # WM_QUIT
                    break
                TranslateMessage(ctypes.byref(msg))
                DispatchMessageW(ctypes.byref(msg))
            else:
                # No messages - sleep briefly to avoid CPU spin
                import time
                time.sleep(0.001)  # 1ms sleep
    
    def toggle_recording(self):
        """Toggle recording state"""
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
        """Start recording audio"""
        self.is_recording = True
        self.frames = []
        self.record_start_time = time.time()
        
        SoundEffects.play_start()
        logger.info("Record", "Listening...")
        
        if self.tray:
            self.tray.update_icon(recording=True)
        if self.ui:
            self.ui.set_state("recording", "● Recording...")
        
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
        """Stop recording and transcribe"""
        self.is_recording = False
        duration = time.time() - self.record_start_time
        
        SoundEffects.play_stop()
        logger.info("Processing", f"{duration:.1f}s of audio...")
        
        if self.tray:
            self.tray.update_icon(recording=False)
        if self.ui:
            self.ui.set_state("processing", "Processing...")
        
        # Transcribe in background
        def _transcribe():
            try:
                import time as timer
                start_time = timer.time()
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                
                wf = wave.open(temp_path, 'wb')
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(self.frames))
                wf.close()
                
                print(f"[Debug] Audio saved: {temp_path} ({os.path.getsize(temp_path)} bytes)")
                
                # Transcribe with optimizations
                if self.model is None:
                    raise RuntimeError("Model not loaded")
                
                transcribe_start = timer.time()
                print(f"[Debug] Starting transcription...")
                
                segments, info = self.model.transcribe(
                    temp_path,
                    beam_size=1,        # Greedy decoding (fastest)
                    language="en",      # Skip language detection
                    vad_filter=False,   # Disable VAD for speed (we record clean audio)
                    word_timestamps=False,  # Don't need word-level timing
                    condition_on_previous_text=False  # Faster, each segment independent
                )
                
                transcribe_time = timer.time() - transcribe_start
                print(f"[Debug] Transcription completed in {transcribe_time:.2f}s")
                print(f"[Debug] Detected language: {info.language}, probability: {info.language_probability:.2f}")
                
                # Collect segments
                segment_texts = []
                for i, seg in enumerate(segments):
                    print(f"[Debug] Segment {i}: '{seg.text.strip()}'")
                    segment_texts.append(seg.text.strip())
                
                raw_text = " ".join(segment_texts)
                print(f"[Debug] Raw transcription: '{raw_text[:100]}...'")
                print(f"[Debug] Total time: {timer.time() - start_time:.2f}s")
                
                # Process text (punctuation, replacements)
                processed_text = self.processor.process(raw_text)
                
                # Clean up
                os.unlink(temp_path)
                
                if processed_text:
                    total_pipeline = timer.time() - start_time
                    
                    # Update stats with processing time
                    self.stats.record(duration, processed_text, total_pipeline)
                    
                    # Add to UI history
                    if self.ui:
                        self.ui.add_transcription(processed_text, duration)
                    
                    # Type into focused app
                    logger.info("Typing", f"{len(processed_text)} chars: {processed_text[:50]}{'...' if len(processed_text) > 50 else ''}")
                    typing_start = timer.time()
                    VirtualKeyboard.type_text(processed_text)
                    typing_time = timer.time() - typing_start
                    
                    # Performance metrics
                    words = len(processed_text.split())
                    speed_ratio = total_pipeline / duration if duration > 0 else 0
                    
                    logger.debug("Perf", f"Typing: {typing_time:.2f}s ({len(processed_text)/typing_time:.0f} chars/sec)")
                    logger.success("Pipeline", f"{total_pipeline:.2f}s total | {speed_ratio:.2f}x realtime | {words} words")
                    logger.info("Stats", self.stats.summary())
                    
                    # Show success briefly
                    if self.ui:
                        self.ui.set_state("idle", f"✓ {len(processed_text.split())} words")
                        # Reset to ready after 2 seconds
                        threading.Timer(2.0, lambda: self.ui and self.ui.set_state("idle", "Ready")).start()
                else:
                    print("[Empty] No speech detected")
                
            except Exception as e:
                print(f"[Error] Transcription failed: {e}")
                SoundEffects.play_error()
        
        threading.Thread(target=_transcribe, daemon=True).start()
    
    def run(self):
        """Run the application"""
        # Start UI
        if self.ui:
            self.ui.start()
            self.ui.set_state("loading")
        
        # Start system tray
        if self.tray:
            self.tray.run()
        
        # Load model in background
        self.load_model()
        
        # Install keyboard hook
        self.install_hook()
        
        try:
            # Run message loop (blocks)
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
    app = WhisperTypingApp()
    app.run()


if __name__ == "__main__":
    main()
