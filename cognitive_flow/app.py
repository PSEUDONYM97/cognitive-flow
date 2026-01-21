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
import hashlib
from datetime import datetime
from pathlib import Path

# These get set during init_app()
GPU_AVAILABLE = False
HAS_TRAY = False
HAS_UI = False
CognitiveFlowUI = None
pyaudio = None
pystray = None
Image = None
ImageDraw = None
logger = None


def init_app(debug=False):
    """Initialize app - load libraries after banner is shown."""
    global GPU_AVAILABLE, HAS_TRAY, HAS_UI, CognitiveFlowUI
    global pyaudio, pystray, Image, ImageDraw, logger
    
    import time
    _t = time.perf_counter
    _timings = {}
    
    from .logger import logger as _logger
    logger = _logger
    
    # In debug mode, enable file logging
    if debug:
        from .paths import DEBUG_LOG_FILE
        logger.set_log_file(DEBUG_LOG_FILE)
    
    # Check if NVIDIA GPU libraries are available (actual loading done in backends.py)
    _start = _t()
    try:
        import site
        user_site = site.USER_SITE
        if user_site:
            cuda_path = Path(user_site) / "nvidia"
            if cuda_path.exists() and any(cuda_path.iterdir()):
                GPU_AVAILABLE = True
                print("[CUDA] NVIDIA libraries detected")
    except Exception:
        pass
    _timings['cuda_check'] = (_t() - _start) * 1000

    _start = _t()
    import pyaudio as _pyaudio
    pyaudio = _pyaudio
    _timings['pyaudio_import'] = (_t() - _start) * 1000
    
    # Optional: pystray for system tray
    _start = _t()
    try:
        import pystray as _pystray
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        pystray = _pystray
        Image = _Image
        ImageDraw = _ImageDraw
        HAS_TRAY = True
    except ImportError:
        print("[Note] Install pystray and pillow for system tray: pip install pystray pillow")
    _timings['pystray_import'] = (_t() - _start) * 1000
    
    # UI module
    _start = _t()
    try:
        from .ui import CognitiveFlowUI as _CognitiveFlowUI
        CognitiveFlowUI = _CognitiveFlowUI
        HAS_UI = True
        print("[UI] Using PyQt6")
    except ImportError as e:
        print(f"[Note] No UI module: {e}")
    _timings['ui_import'] = (_t() - _start) * 1000
    
    # Log timings in debug mode
    if debug:
        logger.info("Init", "Library load times:")
        for name, ms in _timings.items():
            logger.timing("Init", name, ms)

# Windows API constants
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_OEM_3 = 0xC0  # Tilde key
VK_ESCAPE = 0x1B  # Escape key

# Wake detection
WAKE_THRESHOLD_SECONDS = 30  # If loop blocked for 30s+, assume sleep/wake

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


class AudioArchive:
    """Save audio recordings as compressed FLAC files for future training data"""

    @staticmethod
    def save(audio_array, sample_rate: int, label: str = "") -> str | None:
        """Save audio synchronously. Returns filename or None on failure."""
        try:
            from .paths import AUDIO_ARCHIVE_DIR
            import soundfile as sf

            # Generate filename: timestamp_hash.flac
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Hash from first 1000 samples + label for uniqueness
            hash_input = audio_array[:1000].tobytes() + label.encode()
            short_hash = hashlib.md5(hash_input).hexdigest()[:8]
            filename = f"{timestamp}_{short_hash}.flac"
            filepath = AUDIO_ARCHIVE_DIR / filename

            # Save as FLAC (lossless compression, ~50% size of WAV)
            sf.write(str(filepath), audio_array, sample_rate, format='FLAC')
            return filename

        except ImportError:
            print("[Audio] soundfile not installed - skipping audio archive")
            return None
        except Exception as e:
            print(f"[Audio] Failed to save: {e}")
            return None

    @staticmethod
    def save_async(audio_array, sample_rate: int, text: str, callback=None):
        """Save audio in background thread. Calls callback(filename) when done."""
        def _save():
            filename = AudioArchive.save(audio_array, sample_rate, text)
            if callback and filename:
                callback(filename)

        threading.Thread(target=_save, daemon=True).start()

    @staticmethod
    def load(filename: str):
        """Load audio from archive. Returns (audio_array, sample_rate) or (None, None)."""
        try:
            from .paths import AUDIO_ARCHIVE_DIR
            import soundfile as sf

            filepath = AUDIO_ARCHIVE_DIR / filename
            if not filepath.exists():
                print(f"[Audio] File not found: {filename}")
                return None, None

            audio_array, sample_rate = sf.read(str(filepath), dtype='float32')
            return audio_array, sample_rate
        except ImportError:
            print("[Audio] soundfile not installed")
            return None, None
        except Exception as e:
            print(f"[Audio] Failed to load: {e}")
            return None, None

    @staticmethod
    def get_latest() -> str | None:
        """Get the filename of the most recent audio file."""
        try:
            from .paths import AUDIO_ARCHIVE_DIR

            files = list(AUDIO_ARCHIVE_DIR.glob("*.flac"))
            if not files:
                return None

            # Sort by modification time, newest first
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return files[0].name
        except Exception as e:
            print(f"[Audio] Failed to get latest: {e}")
            return None


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
    
    # User-configurable word replacements (loaded from config)
    REPLACEMENTS = {}

    # Filler words to remove (vocal pauses)
    FILLER_WORDS = {
        "um", "uh", "uhh", "umm", "ummm", "uhhh",
        "er", "err", "errr", "ah", "ahh", "ahhh",
        "hmm", "hmmm", "hmmmm", "mm", "mmm", "mmmm",
    }

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
    
    def _remove_filler_words(self, text: str) -> str:
        """Remove filler words like um, uh, er, etc."""
        import re

        words = text.split()
        filtered = []

        for word in words:
            # Strip punctuation for comparison but keep original
            clean_word = re.sub(r'[.,!?;:\'"]+$', '', word).lower()
            if clean_word not in self.FILLER_WORDS:
                filtered.append(word)

        return ' '.join(filtered)

    def _detect_hallucination_loop(self, text: str) -> str | None:
        """Detect and remove Whisper hallucination loops (e.g., 'I'm I'm I'm...' 50 times)"""
        import re
        
        # Find any word/phrase repeated 10+ times consecutively
        # High threshold to avoid catching intentional repetition for emphasis
        # Matches: "I'm I'm I'm..." (10+ times) but not "really really really"
        pattern = r'\b(\w+(?:\'\w+)?)\s+(?:\1\s+){9,}'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            repeated = match.group(1)
            # Remove all but one instance of the repeated word
            cleaned = re.sub(pattern, repeated + ' ', text, flags=re.IGNORECASE)
            print(f"[Warning] Hallucination loop detected: '{repeated}' repeated, cleaned up")
            return cleaned.strip()
        
        return None
    
    def process(self, text: str) -> str:
        if not text:
            return text
        
        import re
        
        # Pass 0: Detect and fix hallucination loops
        loop_fix = self._detect_hallucination_loop(text)
        if loop_fix:
            text = loop_fix

        # Pass 1: Remove filler words (um, uh, er, etc.)
        text = self._remove_filler_words(text)

        # Pass 2: Fix Whisper artifacts (e.g., ",nd" -> "command")
        text = self._correct_whisper_artifacts(text)

        # Pass 3: Normalize fancy characters
        for fancy, simple in self.CHAR_NORMALIZE.items():
            text = text.replace(fancy, simple)

        # Pass 4: Custom word replacements
        for word, replacement in self.REPLACEMENTS.items():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub(replacement, text)

        # Pass 5: Convert spoken punctuation to symbols
        # Only replace when it's a standalone word
        for spoken, punct in self.PUNCTUATION.items():
            pattern = re.compile(r'\b' + re.escape(spoken) + r'\b', re.IGNORECASE)
            text = pattern.sub(punct, text)

        # Pass 6: Clean up spacing around punctuation
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


class MediaControl:
    """Control media playback via Windows media keys."""
    VK_MEDIA_PLAY_PAUSE = 0xB3
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002

    @staticmethod
    def send_play_pause():
        """Send media play/pause key to pause/resume media players."""
        try:
            user32 = ctypes.windll.user32
            # Key down
            user32.keybd_event(MediaControl.VK_MEDIA_PLAY_PAUSE, 0, MediaControl.KEYEVENTF_EXTENDEDKEY, 0)
            # Key up
            user32.keybd_event(MediaControl.VK_MEDIA_PLAY_PAUSE, 0, MediaControl.KEYEVENTF_EXTENDEDKEY | MediaControl.KEYEVENTF_KEYUP, 0)
        except Exception as e:
            print(f"[Media] Failed to send play/pause: {e}")

    @staticmethod
    def is_audio_playing() -> bool:
        """Check if any audio session is actively playing using pycaw.

        Returns True if any app is actively outputting audio, False otherwise.
        This checks session STATE (playing/paused), not audio levels.
        """
        try:
            from pycaw.pycaw import AudioUtilities

            # Get all audio sessions
            sessions = AudioUtilities.GetAllSessions()

            for session in sessions:
                # State 1 = AudioSessionStateActive (actually playing audio)
                # State 0 = AudioSessionStateInactive (open but not playing)
                # State 2 = AudioSessionStateExpired
                if session.State == 1:
                    return True

            return False

        except ImportError:
            # pycaw not installed - fall back to assuming no audio
            print("[Media] pycaw not installed - install with: pip install pycaw")
            return False
        except Exception:
            # If detection fails, assume audio is NOT playing (safer)
            return False


class UpdateChecker:
    """Check GitHub for new versions."""
    GITHUB_REPO = "PSEUDONYM97/cognitive-flow"
    RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    TAGS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/tags"

    @staticmethod
    def parse_version(version_str: str) -> tuple:
        """Parse version string like '1.11.0' into tuple (1, 11, 0)."""
        # Strip 'v' prefix if present
        version_str = version_str.lstrip('v')
        try:
            parts = version_str.split('.')
            return tuple(int(p) for p in parts)
        except (ValueError, AttributeError):
            return (0, 0, 0)

    @staticmethod
    def check_for_update(current_version: str) -> dict | None:
        """
        Check GitHub for newer version.
        Returns dict with 'version' and 'url' if update available, None otherwise.
        """
        import urllib.request
        import urllib.error

        try:
            # Try releases first (preferred)
            req = urllib.request.Request(
                UpdateChecker.RELEASES_URL,
                headers={'User-Agent': 'CognitiveFlow-UpdateChecker'}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                latest_version = data.get('tag_name', '').lstrip('v')
                html_url = data.get('html_url', f'https://github.com/{UpdateChecker.GITHUB_REPO}')

                if UpdateChecker.parse_version(latest_version) > UpdateChecker.parse_version(current_version):
                    return {'version': latest_version, 'url': html_url}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # No releases yet, try tags
                try:
                    req = urllib.request.Request(
                        UpdateChecker.TAGS_URL,
                        headers={'User-Agent': 'CognitiveFlow-UpdateChecker'}
                    )
                    with urllib.request.urlopen(req, timeout=5) as response:
                        tags = json.loads(response.read().decode())
                        if tags:
                            latest_tag = tags[0].get('name', '').lstrip('v')
                            if UpdateChecker.parse_version(latest_tag) > UpdateChecker.parse_version(current_version):
                                return {
                                    'version': latest_tag,
                                    'url': f'https://github.com/{UpdateChecker.GITHUB_REPO}'
                                }
                except Exception:
                    pass
        except Exception:
            # Network error, timeout, etc - silently ignore
            pass

        return None

    @staticmethod
    def check_async(current_version: str, callback=None):
        """Check for updates in background thread."""
        def _check():
            result = UpdateChecker.check_for_update(current_version)
            if result and callback:
                callback(result)
            elif result:
                print(f"[Update] New version available: v{result['version']}")
                print(f"[Update] Download: {result['url']}")

        threading.Thread(target=_check, daemon=True).start()


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
    
    def on_toggle_overlay(self, icon, item):
        self.app.show_overlay = not self.app.show_overlay
        self.app.save_config()
        if self.app.ui:
            if self.app.show_overlay:
                self.app.ui.show()
            else:
                self.app.ui.hide()
        print(f"[Settings] Overlay: {'visible' if self.app.show_overlay else 'hidden'}")

    def on_reset_overlay(self, icon, item):
        """Reset overlay position and visibility - use if indicator goes missing"""
        if self.app.ui and self.app.ui.indicator:
            self.app.show_overlay = True
            self.app.ui.indicator.ensure_visible()
            print("[Settings] Overlay position reset")
    
    def run(self):
        if not HAS_TRAY:
            return
        
        menu = pystray.Menu(
            pystray.MenuItem("Settings", self.on_show_stats),
            pystray.MenuItem(
                "Show Overlay",
                self.on_toggle_overlay,
                checked=lambda item: self.app.show_overlay if self.app else True
            ),
            pystray.MenuItem("Reset Overlay Position", self.on_reset_overlay),
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
        """Type text by posting WM_CHAR messages directly to the focused control.
        
        This bypasses the keyboard input queue and sends characters directly,
        which should be faster and more reliable for long strings.
        """
        if not text:
            return
        
        # Sanitize before typing
        text = VirtualKeyboard.sanitize_text(text)
        
        if not text:
            return
        
        WM_CHAR = 0x0102
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        
        # Get the foreground window
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return
        
        # Get the focused control (requires thread attachment)
        current_thread = kernel32.GetCurrentThreadId()
        fg_thread = user32.GetWindowThreadProcessId(hwnd, None)
        
        # Attach to the foreground thread to get focus
        attached = False
        if current_thread != fg_thread:
            attached = user32.AttachThreadInput(current_thread, fg_thread, True)
        
        focus = user32.GetFocus()
        target = focus if focus else hwnd
        
        # Post WM_CHAR for each character
        for char in text:
            user32.PostMessageW(target, WM_CHAR, ord(char), 0)
        
        # Detach from thread
        if attached:
            user32.AttachThreadInput(current_thread, fg_thread, False)


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
        self.input_device_index = None  # None = system default
        
        self.stats = Statistics()
        self.processor = TextProcessor()
        self.backend = None  # TranscriptionBackend instance
        self.model = None    # Legacy compatibility (points to backend)
        self.model_loading = False
        self.using_gpu = False
        
        # For retry functionality
        self.last_audio = None
        self.last_duration = 0.0
        self.last_audio_file: str | None = None

        # State reset timer (to prevent race conditions)
        self._state_reset_timer: threading.Timer | None = None

        # Double-escape tracking for cancel
        self._last_escape_time: float = 0.0
        self._escape_window: float = 0.5  # 500ms window for double-escape
        
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
            model = self.parakeet_model if self.backend_type == 'parakeet' else self.model_name
            logger.info("Model", f"Loading {self.backend_type} model ({model})...")
    
    def _load_config(self):
        # Defaults
        self.backend_type = 'whisper'  # 'whisper' or 'parakeet'
        self.model_name = 'medium'     # Whisper: tiny/base/small/medium/large
        self.parakeet_model = 'nemo-parakeet-tdt-0.6b-v2'  # Parakeet model
        self.add_trailing_space = True  # Add space after each transcription
        self.input_device_index = None  # None = system default
        self.show_overlay = True  # Show floating indicator
        self.archive_audio = True  # Save audio recordings for future training
        self.text_replacements = {}  # User's text replacements (from -> to)
        self.pause_media = False  # Pause media playback during recording

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.backend_type = config.get('backend_type', 'whisper')
                    self.model_name = config.get('model_name', 'medium')
                    self.parakeet_model = config.get('parakeet_model', 'nemo-parakeet-tdt-0.6b-v2')
                    self.add_trailing_space = config.get('add_trailing_space', True)
                    self.input_device_index = config.get('input_device_index', None)
                    self.show_overlay = config.get('show_overlay', True)
                    self.archive_audio = config.get('archive_audio', True)
                    self.text_replacements = config.get('text_replacements', {})
                    self.pause_media = config.get('pause_media', False)
            except:
                pass

        # Apply replacements to processor
        self.processor.REPLACEMENTS = dict(self.text_replacements)
    
    def save_config(self):
        config = {
            'backend_type': self.backend_type,
            'model_name': self.model_name,
            'parakeet_model': self.parakeet_model,
            'add_trailing_space': self.add_trailing_space,
            'input_device_index': self.input_device_index,
            'show_overlay': self.show_overlay,
            'archive_audio': self.archive_audio,
            'text_replacements': self.text_replacements,
            'pause_media': self.pause_media,
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[Config] Saved preferences")
    
    def get_input_devices(self) -> list[tuple[int, str]]:
        """Get list of available input devices as (index, name) tuples."""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                name = info.get('name', f'Device {i}')
                devices.append((i, name))
        return devices
    

    
    def load_model(self):
        self.model_loading = True

        def _load():
            from .backends import get_backend, WhisperBackend, ParakeetBackend
            _t = time.perf_counter

            try:
                # Determine model name based on backend
                if self.backend_type == 'parakeet':
                    model_name = self.parakeet_model
                    backend_class = ParakeetBackend
                else:
                    model_name = self.model_name
                    backend_class = WhisperBackend

                if self.debug:
                    logger.info("Model", f"Loading {self.backend_type}: {model_name}...")

                # Create backend instance
                self.backend = backend_class()

                # Try GPU first
                _start = _t()
                use_gpu = GPU_AVAILABLE

                if use_gpu:
                    if self.debug:
                        logger.info("GPU", f"Loading {model_name} on CUDA...")

                success = self.backend.load(model_name, use_gpu=use_gpu)

                if not success:
                    raise RuntimeError(f"Failed to load {model_name}")

                self.using_gpu = self.backend.using_gpu
                self.model = self.backend  # Legacy compatibility

                load_time = (_t() - _start) * 1000
                device = "GPU" if self.using_gpu else "CPU"

                if self.debug:
                    logger.timing(device, "model_load", load_time)

                # Warmup for Whisper (Parakeet doesn't need it)
                if isinstance(self.backend, WhisperBackend) and self.using_gpu:
                    if self.debug:
                        logger.info("GPU", "Warming up model...")
                    _start = _t()
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
                    list(self.backend._model.transcribe(warmup_audio, beam_size=1, language="en"))
                    os.unlink(warmup_audio)
                    if self.debug:
                        logger.timing("GPU", "warmup", (_t() - _start) * 1000)

                # Update status
                status = f"Ready ({device})"
                if self.debug:
                    logger.success("Ready", f"Model: {model_name} ({device})")

                if self.tray:
                    self.tray.update_icon(recording=False, loading=False)
                    self.tray.icon.title = f"Cognitive Flow - {status}"
                if self.ui:
                    self.ui.set_state("idle", status)

            except Exception as e:
                print(f"[Error] Failed to load {self.backend_type}: {e}")

                # If Parakeet failed, clean up corrupted downloads and fall back to Whisper
                original_backend = self.backend_type
                if original_backend == 'parakeet':
                    print("[Fallback] Parakeet failed - reverting to Whisper")
                    # Clean up any corrupted/partial downloads
                    from .backends import ParakeetBackend
                    if ParakeetBackend.cleanup_failed_download(self.parakeet_model):
                        print("[Cleanup] Removed corrupted Parakeet cache files")

                # Fallback to Whisper on CPU
                try:
                    from .backends import WhisperBackend
                    self.backend = WhisperBackend()

                    # Try user's preferred Whisper model first
                    fallback_model = self.model_name if self.model_name else "small"
                    if not self.backend.load(fallback_model, use_gpu=GPU_AVAILABLE):
                        # Last resort: base on CPU
                        self.backend.load("base", use_gpu=False)
                        fallback_model = "base"

                    self.model = self.backend
                    self.model_name = fallback_model
                    self.backend_type = "whisper"
                    self.using_gpu = self.backend.using_gpu

                    # Save config so we don't try failed backend on next startup
                    if original_backend != 'whisper':
                        self.save_config()
                        print(f"[Config] Reverted to Whisper ({fallback_model})")

                    device = "GPU" if self.using_gpu else "CPU"
                    print(f"[Ready] Using Whisper {fallback_model} ({device})")
                    if self.ui:
                        self.ui.set_state("idle", f"Ready ({device})")
                    if self.tray:
                        self.tray.update_icon(recording=False, loading=False)

                except Exception as e2:
                    print(f"[Fatal] Could not load any model: {e2}")
                    if self.ui:
                        self.ui.set_state("idle", "No model!")
            finally:
                self.model_loading = False

        threading.Thread(target=_load, daemon=True).start()

    def warmup_gpu(self):
        """Run a silent warmup transcription to re-initialize GPU after sleep/wake."""
        if self.model_loading:
            return  # Already loading
        if not self.backend or not self.backend.is_loaded:
            return  # No model loaded
        if not self.using_gpu:
            return  # CPU doesn't need warmup

        def _warmup():
            try:
                import numpy as np
                print("[Warmup] Re-initializing GPU after wake...")

                if self.ui:
                    self.ui.set_state("processing", "Warming up...")

                # Generate 1 second of silence
                warmup_audio = np.zeros(16000, dtype=np.float32)

                _start = time.perf_counter()
                self.backend.transcribe(warmup_audio, sample_rate=16000)
                warmup_time = (time.perf_counter() - _start) * 1000

                print(f"[Warmup] GPU ready in {warmup_time:.0f}ms")

                if self.ui:
                    self.ui.set_state("idle", "Ready")

            except Exception as e:
                print(f"[Warmup] Failed: {e}")
                if self.ui:
                    self.ui.set_state("idle", "Ready")

        threading.Thread(target=_warmup, daemon=True).start()

    def keyboard_callback(self, nCode, wParam, lParam):
        if nCode >= 0:
            kb = lParam.contents
            if kb.vkCode == VK_OEM_3:
                if wParam == WM_KEYDOWN:
                    self.toggle_recording()
                    return 1
            elif kb.vkCode == VK_ESCAPE:
                if wParam == WM_KEYDOWN and self.is_recording:
                    current_time = time.time()
                    if current_time - self._last_escape_time < self._escape_window:
                        # Double-escape: cancel recording
                        self.cancel_recording()
                        self._last_escape_time = 0.0  # Reset
                        return 1
                    else:
                        # First escape: just note the time
                        self._last_escape_time = current_time
                        print("[Record] Press Escape again to cancel")
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

        last_loop_time = time.time()

        while self.running:
            current_time = time.time()

            # Detect wake from sleep: if loop was blocked for 30+ seconds
            time_gap = current_time - last_loop_time
            if time_gap > WAKE_THRESHOLD_SECONDS:
                print(f"[Wake] Detected system resume (gap: {time_gap:.1f}s)")
                self.warmup_gpu()

            last_loop_time = current_time

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
    
    def _cancel_state_reset(self):
        """Cancel any pending state reset timer"""
        if self._state_reset_timer:
            self._state_reset_timer.cancel()
            self._state_reset_timer = None

    def _schedule_state_reset(self, delay: float = 2.0):
        """Schedule state reset to 'Ready' after delay, cancelling any existing timer"""
        self._cancel_state_reset()
        def _reset():
            if self.ui and not self.is_recording:
                self.ui.set_state("idle", "Ready")
        self._state_reset_timer = threading.Timer(delay, _reset)
        self._state_reset_timer.start()

    def start_recording(self):
        # Cancel any pending state reset before changing to recording state
        self._cancel_state_reset()

        self.is_recording = True
        self.frames = []
        self.record_start_time = time.time()

        # Pause media if enabled - but only if audio is actually playing
        self._media_was_paused = False
        if self.pause_media:
            if MediaControl.is_audio_playing():
                MediaControl.send_play_pause()
                self._media_was_paused = True
                if self.debug:
                    print("[Media] Paused (audio was playing)")
            elif self.debug:
                print("[Media] Skipped pause (no audio playing)")

        SoundEffects.play_start()
        if self.debug:
            logger.info("Record", "Listening...")
        
        if self.tray:
            self.tray.update_icon(recording=True)
        if self.ui:
            self.ui.set_state("recording", "Recording...")
        
        def _record():
            # Use selected input device or system default
            stream_kwargs = {
                'format': self.FORMAT,
                'channels': self.CHANNELS,
                'rate': self.RATE,
                'input': True,
                'frames_per_buffer': self.CHUNK,
            }
            if self.input_device_index is not None:
                stream_kwargs['input_device_index'] = self.input_device_index
            
            stream = self.audio.open(**stream_kwargs)
            
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
    
    def cancel_recording(self):
        """Cancel recording without transcribing (Escape key)."""
        self.is_recording = False
        self.frames = []  # Discard recorded audio

        # Resume media if we paused it
        if getattr(self, '_media_was_paused', False):
            MediaControl.send_play_pause()
            self._media_was_paused = False
            if self.debug:
                print("[Media] Resumed")

        SoundEffects.play_error()  # Different sound to indicate cancel
        print("[Record] Cancelled")

        if self.tray:
            self.tray.update_icon(recording=False)
        if self.ui:
            self.ui.set_state("idle", "Cancelled")
            self._schedule_state_reset(delay=1.0)

    def stop_recording(self):
        self.is_recording = False
        time.sleep(0.1)  # Give recording thread time to finish last read
        duration = time.time() - self.record_start_time

        # Resume media if we paused it
        if getattr(self, '_media_was_paused', False):
            MediaControl.send_play_pause()
            self._media_was_paused = False
            if self.debug:
                print("[Media] Resumed")

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
                _t = time.perf_counter
                _timings = {}
                pipeline_start = _t()

                _start = _t()
                audio_data = b''.join(self.frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                _timings['audio_convert'] = (_t() - _start) * 1000

                # Store for retry
                self.last_audio = audio_array
                self.last_duration = duration

                # Save audio EARLY (before transcription) so we don't lose it on failure
                if self.archive_audio:
                    self.last_audio_file = AudioArchive.save(audio_array, 16000, "pending")
                    if self.last_audio_file and self.debug:
                        logger.info("Audio", f"Saved: {self.last_audio_file}")

                # Check for audio issues
                max_amp = np.max(np.abs(audio_array))
                num_samples = len(audio_array)
                
                if self.debug:
                    logger.info("Audio", f"samples={num_samples} max_amp={max_amp:.4f} duration={duration:.2f}s")
                
                if max_amp < 0.01:
                    logger.warning("Audio", "No audio detected - check microphone")
                    SoundEffects.play_error()
                    if self.ui:
                        self.ui.set_state("idle", "No audio!")
                        self._schedule_state_reset()
                    return
                
                if self.backend is None or not self.backend.is_loaded:
                    raise RuntimeError("Model not loaded")

                _start = _t()
                result = self.backend.transcribe(audio_array, sample_rate=16000)
                _timings['transcribe'] = result.duration_ms
                raw_text = result.raw_text

                if self.debug:
                    backend_name = self.backend.name.capitalize()
                    logger.info("Raw", f'{backend_name} output: "{raw_text}"')
                    if result.segments:
                        logger.info("Segments", f"count={len(result.segments)}")
                
                _start = _t()
                processed_text = self.processor.process(raw_text)
                _timings['text_process'] = (_t() - _start) * 1000
                
                if self.debug and processed_text != raw_text:
                    logger.info("Processed", f'After cleanup: "{processed_text}"')
                
                if processed_text:
                    _start = _t()
                    # Add trailing space if enabled (helps separate consecutive transcriptions)
                    output_text = processed_text + " " if self.add_trailing_space else processed_text
                    VirtualKeyboard.type_text(output_text)
                    _timings['typing'] = (_t() - _start) * 1000
                    
                    total_pipeline = (_t() - pipeline_start) * 1000  # ms
                    _timings['total'] = total_pipeline
                    
                    self.stats.record(duration, processed_text, total_pipeline / 1000)
                    
                    if self.ui:
                        self.ui.add_transcription(processed_text, duration, self.last_audio_file)
                    
                    # Log to file for debugging terminal breaks (debug mode only)
                    if self.debug:
                        logger.log_transcription(raw_text, processed_text, duration, _timings)
                    
                    # Rich logging
                    words = len(processed_text.split())
                    chars = len(processed_text)
                    preview = processed_text[:60] + "..." if len(processed_text) > 60 else processed_text
                    preview = preview.replace('\n', ' ')  # Single line preview
                    
                    if self.debug:
                        # Verbose debug output with timings
                        logger.success("Done", f"{words} words, {chars} chars")
                        for name, ms in _timings.items():
                            logger.timing("Pipeline", name, ms)
                        logger.info("Text", f'"{preview}"')
                    else:
                        # Concise but useful
                        logger.success("Typed", f'{words}w/{chars}c in {total_pipeline:.1f}s | "{preview}"')
                    
                    if self.ui:
                        self.ui.set_state("idle", f"{words} words")
                        self._schedule_state_reset()
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

    def retry_last(self, audio_file: str | None = None):
        """Retry transcription from saved audio file.

        If audio_file is None, uses the most recent saved audio.
        """
        if self.model_loading:
            logger.warning("Retry", "Model still loading...")
            return

        if not self.backend or not self.backend.is_loaded:
            logger.error("Retry", "Model not loaded")
            SoundEffects.play_error()
            return

        # Find the audio file to retry
        target_file = audio_file or self.last_audio_file or AudioArchive.get_latest()
        if not target_file:
            logger.warning("Retry", "No audio file to retry")
            if self.ui:
                self.ui.set_state("idle", "No audio!")
                self._schedule_state_reset()
            return

        logger.info("Retry", f"Loading {target_file}...")
        if self.ui:
            self.ui.set_state("processing", "Retrying...")

        def _retry():
            try:
                audio_array, sample_rate = AudioArchive.load(target_file)
                if audio_array is None:
                    logger.error("Retry", "Failed to load audio")
                    SoundEffects.play_error()
                    if self.ui:
                        self.ui.set_state("idle", "Load failed!")
                        self._schedule_state_reset()
                    return

                duration = len(audio_array) / sample_rate
                logger.info("Retry", f"Transcribing {duration:.1f}s of audio...")

                result = self.backend.transcribe(audio_array, sample_rate=sample_rate)
                raw_text = result.raw_text
                processed_text = self.processor.process(raw_text)

                if processed_text:
                    output_text = processed_text + " " if self.add_trailing_space else processed_text
                    VirtualKeyboard.type_text(output_text)

                    self.stats.record(duration, processed_text, 0)
                    if self.ui:
                        self.ui.add_transcription(processed_text, duration, target_file)

                    words = len(processed_text.split())
                    logger.success("Retry", f"{words} words from {target_file}")
                    if self.ui:
                        self.ui.set_state("idle", f"{words} words")
                        self._schedule_state_reset()
                else:
                    logger.warning("Retry", "No speech detected")
                    if self.ui:
                        self.ui.set_state("idle", "No speech")
                        self._schedule_state_reset()

            except Exception as e:
                logger.error("Retry", f"Failed: {e}")
                SoundEffects.play_error()
                if self.ui:
                    self.ui.set_state("idle", "Retry failed!")
                    self._schedule_state_reset()

        threading.Thread(target=_retry, daemon=True).start()

    def run(self):
        if self.ui:
            self.ui.start()
            if self.show_overlay:
                self.ui.set_state("loading")
            else:
                self.ui.hide()
        
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
        print("    v1.13.2 - Use pycaw for reliable audio session state detection")
        print("            - Checks actual session state (playing/paused) instead of audio levels")
        print("    v1.13.0 - Fix pause media playing music when already paused")
        print("            - Now detects if audio is playing before toggling (via Windows Audio API)")
        print("    v1.12.0 - Update checker")
        print("            - Checks GitHub for new versions on startup, notifies if update available")
        print("    v1.11.0 - Pause media during recording")
        print("            - Toggle in Settings to auto-pause Spotify/YouTube/etc while recording")
        print("    v1.10.0 - Custom text replacements in Settings")
        print("            - Add/remove word corrections via UI (no built-in defaults)")
        print("    v1.9.2 - Double-Escape to cancel recording (prevents accidental cancel)")
        print("           - Remove filler words (um, uh, er, ah, hmm) from transcriptions")
        print("           - Use EXHAUSTIVE cuDNN algo search for best GPU performance")
        print("    v1.9.1 - Fix overlay not appearing until mouse wiggle")
        print("           - Force Windows compositor refresh with position nudge")
        print("    v1.9.0 - GPU warmup on wake: auto-reinitialize after sleep/resume")
        print("           - Detects system wake and runs silent warmup transcription")
        print("    v1.8.9 - Disable dropdowns during model loading (prevent lockups)")
        print("           - Auto-cleanup corrupted Parakeet downloads on load failure")
        print("    v1.8.8 - Disable scroll wheel on settings dropdowns (prevent accidental changes)")
        print("    v1.8.7 - Fix recording state not showing: force immediate color update")
        print("           - Use QueuedConnection for thread-safe UI updates")
        print("           - Add processEvents() after state change for immediate visibility")
        print("    v1.8.6 - Fix state race condition: cancel pending timers on new recording")
        print("    v1.8.5 - Fix disappearing overlay: ensure_visible(), refresh geometry")
        print("           - Add 'Reset Overlay Position' to system tray menu")
        print("    v1.8.4 - Lazy backend loading: 50x faster startup (~8s -> 0.16s)")
        print("    v1.8.3 - Add nvrtc DLL support, suppress onnxruntime warnings")
        print("    v1.8.2 - Shared CUDA path setup fixes cuDNN loading for both backends")
        print("    v1.8.1 - Graceful fallback: Parakeet failure auto-reverts to Whisper")
        print("           - Better error handling for missing CUDA libs")
        print("    v1.8.0 - NVIDIA Parakeet backend via onnx-asr (~50x faster)")
        print("           - Backend selector in Settings: Whisper vs Parakeet")
        print("           - TranscriptionBackend abstraction for swappable engines")
        print("    v1.7.1 - Fix OOM crash when rapidly switching models in Settings")
        print("    v1.7.0 - Audio saved before transcription (survives failures)")
        print("           - Retry Last Recording in right-click menu")
        print("    v1.6.3 - Hallucination loop detection (10+ repeats)")
        print("    v1.6.2 - Initial prompt for better contraction accuracy")
        print("    v1.6.1 - Settings UI fixes, live model switching")
        print("    v1.6.0 - Audio archive: saves recordings as FLAC for training data")
        print("           - Paired with transcriptions in history.json")
        print("    v1.5.1 - Reduced collapse delay to 3s, fix repaint on show")
        print("    v1.5.0 - Collapsible indicator (shrinks to dot when idle)")
        print("           - Expands on hover or when recording/processing")
        print("    v1.4.0 - Show Overlay toggle in system tray menu")
        print("    v1.3.3 - WM_CHAR direct posting (bypass keyboard queue)")
        print("    v1.3.0 - Microphone input device selector in Settings")
        print("    v1.2.1 - Comprehensive timing + debug logging")
        print("           - Logs to %APPDATA%\\CognitiveFlow\\debug_transcriptions.log")
        print("    v1.2.0 - Enhanced character sanitization, beam_size=5")
        print("    v1.1.0 - GUI mode by default, --debug for console")
        print("    v1.0.0 - Initial release as proper package")
        print()
        print("=" * 60)
    
    # Now load all the heavy libraries
    init_app(debug=args.debug)

    app = CognitiveFlowApp(debug=args.debug)

    # Check for updates in background
    UpdateChecker.check_async(__version__)

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
