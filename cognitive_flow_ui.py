"""
Cognitive Flow UI - Neural Calm aesthetic
Floating indicator + Settings window with distinctive design
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import json
from datetime import datetime
from pathlib import Path
from typing import Callable
import threading
import math

# =============================================================================
# DESIGN SYSTEM - "Neural Calm"
# =============================================================================

COLORS = {
    # Base palette - deep, calm darkness
    "bg_deep": "#0A0A0B",       # Deepest background
    "bg_primary": "#111113",    # Primary surfaces
    "bg_elevated": "#1A1A1D",   # Cards, elevated surfaces
    "bg_hover": "#242428",      # Hover states
    
    # Text hierarchy
    "text_primary": "#FAFAFA",   # Headlines, important
    "text_secondary": "#A1A1A6", # Body text
    "text_muted": "#5C5C66",     # Hints, timestamps
    
    # Accent - warm amber (cognition, thought, warmth)
    "accent": "#E8A838",         # Primary accent
    "accent_dim": "#B8842D",     # Dimmed accent
    "accent_glow": "#E8A83822",  # Glow effect (with alpha)
    
    # States
    "recording": "#C73E3E",      # Recording - soft crimson
    "recording_glow": "#C73E3E33",
    "success": "#3EC76A",        # Success states
    "idle": "#3D3D42",           # Idle/inactive
}

FONTS = {
    "mono": ("JetBrains Mono", "Consolas", "Monaco", "monospace"),
    "display": ("Georgia", "Cambria", "Times New Roman", "serif"),
}

# =============================================================================
# FLOATING INDICATOR - Substantial, informative presence
# =============================================================================

class FloatingIndicator:
    """
    Floating indicator with substance - shows state with visual presence.
    Rounded card with glow, status text, and breathing animation.
    """
    
    def __init__(self, on_click: Callable | None = None, on_right_click: Callable | None = None):
        self.on_click = on_click
        self.on_right_click = on_right_click
        self.root: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None
        self.status_label: tk.Label | None = None
        
        self.state = "loading"  # loading, idle, recording, processing
        self.status_text = "Loading..."
        self._breath_phase = 0.0
        self._animation_id = None
        
        # Size - much larger, substantial
        self.width = 180
        self.height = 72
        
    def create(self):
        """Create the floating indicator window"""
        self.root = tk.Tk()
        self.root.title("")
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.95)
        
        # Position - bottom right, above taskbar with proper margin
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = screen_w - self.width - 30
        y = screen_h - self.height - 80  # More clearance for taskbar
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
        
        # Background frame
        self.root.configure(bg=COLORS["bg_elevated"])
        
        # Main container
        container = tk.Frame(self.root, bg=COLORS["bg_elevated"], padx=12, pady=10)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Status indicator (left side) - breathing circle with smooth rendering
        self.canvas = tk.Canvas(
            container,
            width=50,
            height=50,
            bg=COLORS["bg_elevated"],
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, padx=(0, 12))
        
        # Status text (right side)
        try:
            status_font = tkfont.Font(family="Segoe UI", size=10, weight="normal")
        except:
            status_font = tkfont.Font(size=10)
        
        self.status_label = tk.Label(
            container,
            text=self.status_text,
            font=status_font,
            bg=COLORS["bg_elevated"],
            fg=COLORS["text_secondary"],
            anchor="w",
            justify=tk.LEFT
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bindings for entire window
        self.root.bind('<Button-1>', self._handle_click)
        self.root.bind('<Button-3>', self._handle_right_click)
        container.bind('<Button-1>', self._handle_click)
        container.bind('<Button-3>', self._handle_right_click)
        self.canvas.bind('<Button-1>', self._handle_click)
        self.canvas.bind('<Button-3>', self._handle_right_click)
        if self.status_label:
            self.status_label.bind('<Button-1>', self._handle_click)
            self.status_label.bind('<Button-3>', self._handle_right_click)
        
        # Dragging with right-click-drag
        self._drag_start = None
        self.root.bind('<ButtonPress-1>', self._start_drag)
        self.root.bind('<B1-Motion>', self._do_drag)
        
        # Initial draw
        self._draw()
        
        # Start breathing animation
        self._animate()
        
    def _draw(self):
        """Draw the indicator circle with smooth rendering"""
        if not self.canvas:
            return
            
        self.canvas.delete("all")
        center = 25  # Canvas is now 50x50
        base_radius = 16  # Larger for better appearance
        
        # Calculate breath effect
        breath = math.sin(self._breath_phase) * 0.15 + 1.0
        
        # State colors and sizing
        if self.state == "loading":
            color = COLORS["accent"]
            radius = base_radius * breath
            show_glow = True
        elif self.state == "recording":
            color = COLORS["recording"]
            radius = base_radius * (breath * 1.1)
            show_glow = True
        elif self.state == "processing":
            color = COLORS["accent"]
            radius = base_radius
            show_glow = True
        else:  # idle
            color = COLORS["success"]
            radius = base_radius * 0.9
            show_glow = False
        
        # Outer glow ring
        if show_glow:
            glow_radius = radius + 4
            self.canvas.create_oval(
                center - glow_radius, center - glow_radius,
                center + glow_radius, center + glow_radius,
                fill="", outline=color, width=2
            )
        
        # Main circle
        self.canvas.create_oval(
            center - radius, center - radius,
            center + radius, center + radius,
            fill=color, outline=""
        )
        
        # Subtle highlight
        highlight_r = radius * 0.3
        highlight_offset = -radius * 0.35
        self.canvas.create_oval(
            center + highlight_offset - highlight_r,
            center + highlight_offset - highlight_r,
            center + highlight_offset + highlight_r,
            center + highlight_offset + highlight_r,
            fill=self._lighten_color(color, 0.4), outline=""
        )
        
        # Update status label color
        if self.status_label:
            self.status_label.configure(fg=COLORS["text_primary"] if show_glow else COLORS["text_secondary"])
    
    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Lighten a hex color"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _animate(self):
        """Breathing animation loop"""
        if not self.root:
            return
        
        # Only animate when not idle
        if self.state != "idle":
            # Breathing speed varies by state
            speed = 0.15 if self.state == "recording" else 0.08
            self._breath_phase += speed
            if self._breath_phase > math.pi * 2:
                self._breath_phase = 0
            self._draw()
            interval = 33  # ~30fps
        else:
            interval = 100  # Slower when idle
        
        self._animation_id = self.root.after(interval, self._animate)
    
    def set_state(self, state: str, status_text: str | None = None):
        """Update indicator state and optional status text"""
        old_state = self.state
        self.state = state
        
        # Update status text
        if status_text:
            self.status_text = status_text
            if self.status_label:
                self.status_label.configure(text=status_text)
        else:
            # Default status messages
            status_map = {
                "loading": "Loading model...",
                "idle": "Ready",
                "recording": "● Recording...",
                "processing": "Processing..."
            }
            self.status_text = status_map.get(state, "Ready")
            if self.status_label:
                self.status_label.configure(text=self.status_text)
        
        if old_state == "idle" and state != "idle":
            self._breath_phase = 0  # Reset breathing
        
        if self.root:
            self.root.after(0, self._draw)
    
    def _handle_click(self, event):
        if self.on_click:
            self.on_click()
    
    def _handle_right_click(self, event):
        if self.on_right_click:
            self.on_right_click()
    
    def _start_drag(self, event):
        self._drag_start = (event.x, event.y)
    
    def _do_drag(self, event):
        if self._drag_start and self.root:
            x = self.root.winfo_x() + (event.x - self._drag_start[0])
            y = self.root.winfo_y() + (event.y - self._drag_start[1])
            self.root.geometry(f"+{x}+{y}")
    
    def destroy(self):
        if self._animation_id and self.root:
            self.root.after_cancel(self._animation_id)
        if self.root:
            self.root.destroy()
            self.root = None


# =============================================================================
# TRANSCRIPTION HISTORY
# =============================================================================

class TranscriptionHistory:
    """Stores and retrieves transcription history"""
    
    def __init__(self, filepath: Path | None = None):
        self.filepath = filepath or Path(__file__).parent / "history.json"
        self.entries: list[dict] = self._load()
    
    def _load(self) -> list[dict]:
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)
    
    def add(self, text: str, duration: float):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "duration": round(duration, 2),
            "words": len(text.split()),
            "chars": len(text)
        }
        self.entries.insert(0, entry)
        
        # Keep last 500
        self.entries = self.entries[:500]
        self._save()
    
    def get_recent(self, n: int = 50) -> list[dict]:
        return self.entries[:n]
    
    def search(self, query: str) -> list[dict]:
        q = query.lower()
        return [e for e in self.entries if q in e["text"].lower()]
    
    def clear(self):
        self.entries = []
        self._save()


# =============================================================================
# SETTINGS WINDOW - Editorial, calm, achievement-focused
# =============================================================================

class SettingsWindow:
    """
    Full settings window with tabs.
    Design: Editorial feel - generous whitespace, clear hierarchy.
    Stats feel like achievements earned.
    """
    
    def __init__(self, app):
        self.app = app
        self.root: tk.Toplevel | None = None
        self.history = TranscriptionHistory()
        self.settings = self._load_settings()
    
    def _load_settings(self) -> dict:
        path = Path(__file__).parent / "settings.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "model": "large-v3",
            "language": "en",
            "sound_effects": True,
            "show_indicator": True,
            "replacements": {"hashtag": "#", "clod": "CLAUDE"}
        }
    
    def _save_settings(self):
        path = Path(__file__).parent / "settings.json"
        with open(path, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def show(self, parent: tk.Tk | None = None):
        """Show the settings window"""
        if self.root and self.root.winfo_exists():
            self.root.lift()
            return
        
        self.root = tk.Toplevel() if parent else tk.Tk()
        self.root.title("Cognitive Flow")
        self.root.geometry("760x580")
        self.root.configure(bg=COLORS["bg_deep"])
        self.root.minsize(600, 400)
        
        # Custom fonts
        try:
            self.font_display = tkfont.Font(family="Georgia", size=24, weight="bold")
            self.font_heading = tkfont.Font(family="Georgia", size=14, weight="bold")
            self.font_body = tkfont.Font(family="Segoe UI", size=10)
            self.font_mono = tkfont.Font(family="Consolas", size=10)
            self.font_small = tkfont.Font(family="Segoe UI", size=9)
        except:
            self.font_display = tkfont.Font(size=24, weight="bold")
            self.font_heading = tkfont.Font(size=14, weight="bold")
            self.font_body = tkfont.Font(size=10)
            self.font_mono = tkfont.Font(family="Courier", size=10)
            self.font_small = tkfont.Font(size=9)
        
        # Main container with padding
        container = tk.Frame(self.root, bg=COLORS["bg_deep"])
        container.pack(fill=tk.BOTH, expand=True, padx=32, pady=24)
        
        # Header
        header = tk.Frame(container, bg=COLORS["bg_deep"])
        header.pack(fill=tk.X, pady=(0, 24))
        
        title = tk.Label(
            header, text="Cognitive Flow",
            font=self.font_display,
            bg=COLORS["bg_deep"], fg=COLORS["text_primary"]
        )
        title.pack(side=tk.LEFT)
        
        subtitle = tk.Label(
            header, text="Voice to text, naturally",
            font=self.font_body,
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
        )
        subtitle.pack(side=tk.LEFT, padx=(16, 0), pady=(12, 0))
        
        # Tab bar
        self.tab_var = tk.StringVar(value="stats")
        tab_bar = tk.Frame(container, bg=COLORS["bg_deep"])
        tab_bar.pack(fill=tk.X, pady=(0, 20))
        
        tabs = [("stats", "Stats"), ("history", "History"), ("settings", "Settings")]
        for tab_id, tab_label in tabs:
            btn = tk.Label(
                tab_bar, text=tab_label,
                font=self.font_body,
                bg=COLORS["bg_deep"],
                fg=COLORS["text_primary"] if self.tab_var.get() == tab_id else COLORS["text_muted"],
                cursor="hand2",
                padx=16, pady=8
            )
            btn.pack(side=tk.LEFT)
            btn.bind('<Button-1>', lambda e, t=tab_id: self._switch_tab(t))
            btn.bind('<Enter>', lambda e, b=btn: b.configure(fg=COLORS["accent"]))
            btn.bind('<Leave>', lambda e, b=btn, t=tab_id: b.configure(
                fg=COLORS["text_primary"] if self.tab_var.get() == t else COLORS["text_muted"]
            ))
        
        # Separator
        sep = tk.Frame(container, bg=COLORS["bg_hover"], height=1)
        sep.pack(fill=tk.X, pady=(0, 20))
        
        # Content area
        self.content = tk.Frame(container, bg=COLORS["bg_deep"])
        self.content.pack(fill=tk.BOTH, expand=True)
        
        # Show initial tab
        self._show_stats_tab()
    
    def _switch_tab(self, tab_id: str):
        self.tab_var.set(tab_id)
        
        # Clear content
        for widget in self.content.winfo_children():
            widget.destroy()
        
        if tab_id == "stats":
            self._show_stats_tab()
        elif tab_id == "history":
            self._show_history_tab()
        elif tab_id == "settings":
            self._show_settings_tab()
    
    def _show_stats_tab(self):
        """Stats tab - achievements earned through voice"""
        stats = self.app.stats.stats
        
        # Stats grid - big numbers, editorial feel
        grid = tk.Frame(self.content, bg=COLORS["bg_deep"])
        grid.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        
        stats_display = [
            (f"{stats['total_records']:,}", "recordings", "Thoughts captured"),
            (f"{stats['total_words']:,}", "words", "Ideas transcribed"),
            (f"{stats['total_characters']:,}", "characters", "Keystrokes saved"),
            (self.app.stats.get_time_saved(), "saved", "Time reclaimed"),
        ]
        
        for i, (value, label, desc) in enumerate(stats_display):
            row, col = divmod(i, 2)
            
            card = tk.Frame(grid, bg=COLORS["bg_elevated"], padx=24, pady=20)
            card.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
            
            # Value - big, prominent
            val_label = tk.Label(
                card, text=value,
                font=tkfont.Font(family="Georgia", size=32, weight="bold"),
                bg=COLORS["bg_elevated"], fg=COLORS["accent"]
            )
            val_label.pack(anchor="w")
            
            # Label
            type_label = tk.Label(
                card, text=label.upper(),
                font=tkfont.Font(family="Consolas", size=9),
                bg=COLORS["bg_elevated"], fg=COLORS["text_muted"]
            )
            type_label.pack(anchor="w", pady=(4, 0))
            
            # Description
            desc_label = tk.Label(
                card, text=desc,
                font=self.font_small,
                bg=COLORS["bg_elevated"], fg=COLORS["text_secondary"]
            )
            desc_label.pack(anchor="w", pady=(8, 0))
        
        # Footer note
        note = tk.Label(
            self.content,
            text="Time saved estimated at 40 WPM typing vs 150 WPM speaking",
            font=self.font_small,
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
        )
        note.pack(pady=(24, 0))
        
        if stats.get("imported_from_whispertyping"):
            imported = tk.Label(
                self.content,
                text="Includes history imported from WhisperTyping",
                font=self.font_small,
                bg=COLORS["bg_deep"], fg=COLORS["success"]
            )
            imported.pack(pady=(4, 0))
    
    def _show_history_tab(self):
        """History tab - searchable transcription list"""
        # Search bar
        search_frame = tk.Frame(self.content, bg=COLORS["bg_deep"])
        search_frame.pack(fill=tk.X, pady=(0, 16))
        
        search_var = tk.StringVar()
        search_entry = tk.Entry(
            search_frame,
            textvariable=search_var,
            font=self.font_body,
            bg=COLORS["bg_elevated"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["accent"],
            relief=tk.FLAT,
            width=40
        )
        search_entry.pack(side=tk.LEFT, ipady=8, ipadx=12)
        search_entry.insert(0, "Search transcriptions...")
        search_entry.bind('<FocusIn>', lambda e: search_entry.delete(0, tk.END) if search_entry.get() == "Search transcriptions..." else None)
        
        # History list with scrollbar
        list_frame = tk.Frame(self.content, bg=COLORS["bg_deep"])
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        canvas = tk.Canvas(list_frame, bg=COLORS["bg_deep"], highlightthickness=0)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        self.history_container = tk.Frame(canvas, bg=COLORS["bg_deep"])
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas_window = canvas.create_window((0, 0), window=self.history_container, anchor="nw")
        
        def configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)
        
        self.history_container.bind('<Configure>', configure_scroll)
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def populate(filter_text: str = ""):
            # Clear
            for w in self.history_container.winfo_children():
                w.destroy()
            
            entries = self.history.search(filter_text) if filter_text and filter_text != "Search transcriptions..." else self.history.get_recent(100)
            
            if not entries:
                empty = tk.Label(
                    self.history_container,
                    text="No transcriptions yet. Press ~ to start recording.",
                    font=self.font_body,
                    bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
                )
                empty.pack(pady=40)
                return
            
            for entry in entries:
                item = tk.Frame(self.history_container, bg=COLORS["bg_elevated"], padx=16, pady=12)
                item.pack(fill=tk.X, pady=4)
                
                # Timestamp
                try:
                    dt = datetime.fromisoformat(entry["timestamp"])
                    time_str = dt.strftime("%b %d, %H:%M")
                except:
                    time_str = "Unknown"
                
                header = tk.Frame(item, bg=COLORS["bg_elevated"])
                header.pack(fill=tk.X)
                
                time_label = tk.Label(
                    header, text=time_str,
                    font=self.font_small,
                    bg=COLORS["bg_elevated"], fg=COLORS["text_muted"]
                )
                time_label.pack(side=tk.LEFT)
                
                meta = tk.Label(
                    header, text=f"{entry.get('words', 0)} words | {entry.get('duration', 0):.1f}s",
                    font=self.font_small,
                    bg=COLORS["bg_elevated"], fg=COLORS["text_muted"]
                )
                meta.pack(side=tk.RIGHT)
                
                # Text preview
                text = entry["text"][:150] + "..." if len(entry["text"]) > 150 else entry["text"]
                text = text.replace("\n", " ")
                
                text_label = tk.Label(
                    item, text=text,
                    font=self.font_body,
                    bg=COLORS["bg_elevated"], fg=COLORS["text_secondary"],
                    wraplength=600, justify=tk.LEFT, anchor="w"
                )
                text_label.pack(fill=tk.X, pady=(8, 0), anchor="w")
                
                # Click to copy
                def copy_text(t=entry["text"]):
                    self.root.clipboard_clear()
                    self.root.clipboard_append(t)
                
                item.bind('<Button-1>', lambda e, t=entry["text"]: copy_text(t))
                for child in item.winfo_children():
                    child.bind('<Button-1>', lambda e, t=entry["text"]: copy_text(t))
        
        # Search binding
        search_var.trace('w', lambda *args: populate(search_var.get()))
        
        # Initial populate
        populate()
    
    def _show_settings_tab(self):
        """Settings tab - clean, organized options"""
        # Scrollable frame
        canvas = tk.Canvas(self.content, bg=COLORS["bg_deep"], highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        settings_frame = tk.Frame(canvas, bg=COLORS["bg_deep"])
        canvas.create_window((0, 0), window=settings_frame, anchor="nw")
        
        # Model selection
        section = tk.Frame(settings_frame, bg=COLORS["bg_deep"])
        section.pack(fill=tk.X, pady=(0, 24))
        
        tk.Label(
            section, text="TRANSCRIPTION MODEL",
            font=tkfont.Font(family="Consolas", size=9),
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
        ).pack(anchor="w")
        
        model_var = tk.StringVar(value=self.settings.get("model", "large-v3"))
        models = ["tiny", "base", "small", "medium", "large-v3"]
        
        model_frame = tk.Frame(section, bg=COLORS["bg_deep"])
        model_frame.pack(fill=tk.X, pady=(8, 0))
        
        for model in models:
            rb = tk.Radiobutton(
                model_frame, text=model,
                variable=model_var, value=model,
                font=self.font_body,
                bg=COLORS["bg_deep"], fg=COLORS["text_secondary"],
                selectcolor=COLORS["bg_elevated"],
                activebackground=COLORS["bg_deep"],
                activeforeground=COLORS["accent"]
            )
            rb.pack(side=tk.LEFT, padx=(0, 16))
        
        tk.Label(
            section, text="Larger models are more accurate but slower. Restart required after change.",
            font=self.font_small,
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
        ).pack(anchor="w", pady=(4, 0))
        
        # Language
        section2 = tk.Frame(settings_frame, bg=COLORS["bg_deep"])
        section2.pack(fill=tk.X, pady=(0, 24))
        
        tk.Label(
            section2, text="LANGUAGE",
            font=tkfont.Font(family="Consolas", size=9),
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
        ).pack(anchor="w")
        
        lang_var = tk.StringVar(value=self.settings.get("language", "en"))
        lang_frame = tk.Frame(section2, bg=COLORS["bg_deep"])
        lang_frame.pack(fill=tk.X, pady=(8, 0))
        
        for lang, label in [("en", "English"), ("auto", "Auto-detect"), ("es", "Spanish"), ("fr", "French")]:
            rb = tk.Radiobutton(
                lang_frame, text=label,
                variable=lang_var, value=lang,
                font=self.font_body,
                bg=COLORS["bg_deep"], fg=COLORS["text_secondary"],
                selectcolor=COLORS["bg_elevated"],
                activebackground=COLORS["bg_deep"],
                activeforeground=COLORS["accent"]
            )
            rb.pack(side=tk.LEFT, padx=(0, 16))
        
        # Toggles
        section3 = tk.Frame(settings_frame, bg=COLORS["bg_deep"])
        section3.pack(fill=tk.X, pady=(0, 24))
        
        tk.Label(
            section3, text="OPTIONS",
            font=tkfont.Font(family="Consolas", size=9),
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
        ).pack(anchor="w", pady=(0, 8))
        
        sound_var = tk.BooleanVar(value=self.settings.get("sound_effects", True))
        indicator_var = tk.BooleanVar(value=self.settings.get("show_indicator", True))
        
        for var, text in [(sound_var, "Play sound effects"), (indicator_var, "Show floating indicator")]:
            cb = tk.Checkbutton(
                section3, text=text,
                variable=var,
                font=self.font_body,
                bg=COLORS["bg_deep"], fg=COLORS["text_secondary"],
                selectcolor=COLORS["bg_elevated"],
                activebackground=COLORS["bg_deep"],
                activeforeground=COLORS["accent"]
            )
            cb.pack(anchor="w", pady=2)
        
        # Replacements
        section4 = tk.Frame(settings_frame, bg=COLORS["bg_deep"])
        section4.pack(fill=tk.X, pady=(0, 24))
        
        tk.Label(
            section4, text="TEXT REPLACEMENTS",
            font=tkfont.Font(family="Consolas", size=9),
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
        ).pack(anchor="w")
        
        tk.Label(
            section4, text="Words that get automatically replaced in transcriptions",
            font=self.font_small,
            bg=COLORS["bg_deep"], fg=COLORS["text_muted"]
        ).pack(anchor="w", pady=(2, 8))
        
        replacements = self.settings.get("replacements", {})
        for word, replacement in replacements.items():
            row = tk.Frame(section4, bg=COLORS["bg_elevated"], padx=12, pady=8)
            row.pack(fill=tk.X, pady=2)
            
            tk.Label(
                row, text=f'"{word}"',
                font=self.font_mono,
                bg=COLORS["bg_elevated"], fg=COLORS["text_secondary"]
            ).pack(side=tk.LEFT)
            
            tk.Label(
                row, text="→",
                font=self.font_body,
                bg=COLORS["bg_elevated"], fg=COLORS["text_muted"]
            ).pack(side=tk.LEFT, padx=12)
            
            tk.Label(
                row, text=f'"{replacement}"',
                font=self.font_mono,
                bg=COLORS["bg_elevated"], fg=COLORS["accent"]
            ).pack(side=tk.LEFT)
        
        # Save button
        def save():
            self.settings["model"] = model_var.get()
            self.settings["language"] = lang_var.get()
            self.settings["sound_effects"] = sound_var.get()
            self.settings["show_indicator"] = indicator_var.get()
            self._save_settings()
            
            # Flash confirmation
            save_btn.configure(text="Saved!", fg=COLORS["success"])
            self.root.after(1500, lambda: save_btn.configure(text="Save Changes", fg=COLORS["text_primary"]))
        
        save_btn = tk.Label(
            settings_frame, text="Save Changes",
            font=self.font_body,
            bg=COLORS["accent"], fg=COLORS["bg_deep"],
            padx=24, pady=10,
            cursor="hand2"
        )
        save_btn.pack(anchor="w", pady=(8, 0))
        save_btn.bind('<Button-1>', lambda e: save())
    
    def close(self):
        if self.root:
            self.root.destroy()
            self.root = None


# =============================================================================
# UI COORDINATOR
# =============================================================================

class CognitiveFlowUI:
    """Coordinates all UI components"""
    
    def __init__(self, app):
        self.app = app
        self.indicator: FloatingIndicator | None = None
        self.settings_window: SettingsWindow | None = None
        self.history = TranscriptionHistory()
        self._tk_thread = None
    
    def start(self):
        """Start the UI"""
        def _run():
            self.indicator = FloatingIndicator(
                on_click=self.app.toggle_recording,
                on_right_click=self.show_settings
            )
            self.indicator.create()
            
            self.settings_window = SettingsWindow(self.app)
            
            if self.indicator.root:
                self.indicator.root.mainloop()
        
        self._tk_thread = threading.Thread(target=_run, daemon=True)
        self._tk_thread.start()
    
    def set_state(self, state: str, status_text: str | None = None):
        """Update UI state"""
        if self.indicator and self.indicator.root:
            self.indicator.root.after(0, lambda: self.indicator.set_state(state, status_text))
    
    def add_transcription(self, text: str, duration: float):
        """Record a transcription to history"""
        self.history.add(text, duration)
    
    def show_settings(self):
        """Show settings window"""
        if self.settings_window and self.indicator and self.indicator.root:
            self.indicator.root.after(0, lambda: self.settings_window.show(self.indicator.root))
    
    def destroy(self):
        """Clean up"""
        if self.indicator:
            if self.indicator.root:
                self.indicator.root.after(0, self.indicator.destroy)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test the UI standalone
    class MockApp:
        class MockStats:
            stats = {
                "total_records": 1483,
                "total_words": 69268,
                "total_characters": 363625,
                "total_seconds": 26419,
                "imported_from_whispertyping": True
            }
            def get_time_saved(self):
                return "21.2 hours"
        
        stats = MockStats()
        
        def toggle_recording(self):
            print("Toggle recording!")
    
    app = MockApp()
    ui = CognitiveFlowUI(app)
    ui.start()
    
    # Keep main thread alive
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
