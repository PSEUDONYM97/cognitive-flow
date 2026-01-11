"""
Cognitive Flow UI - PyQt6 floating indicator and settings dialog.
"""

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QMenu, QDialog, QPushButton, QComboBox, QSpinBox,
                              QCheckBox, QScrollArea, QFrame, QGraphicsDropShadowEffect)


from PyQt6.QtCore import (Qt, QPropertyAnimation, QEasingCurve, pyqtProperty,
                          QObject, pyqtSignal, QTimer, QPoint, QSize)
from PyQt6.QtGui import (QPainter, QColor, QRadialGradient, QFont, QAction, 
                         QPainterPath, QLinearGradient, QPen, QCursor, QGuiApplication)
import sys
import json
from datetime import datetime

from .paths import HISTORY_FILE


class NoScrollComboBox(QComboBox):
    """ComboBox that ignores scroll wheel events to prevent accidental changes."""
    def wheelEvent(self, event):
        event.ignore()  # Pass to parent (scroll area) instead of changing value


# Professional color system
COLORS = {
    # Backgrounds with proper alpha
    "bg_primary": QColor(16, 16, 18, 165),       # Main UI background
    "bg_secondary": QColor(28, 28, 32, 230),     # Settings/dialogs
    "bg_elevated": QColor(38, 38, 42, 245),      # Elevated surfaces
    "bg_hover": QColor(255, 255, 255, 8),        # Hover states
    
    # Borders and dividers
    "border_subtle": QColor(255, 255, 255, 20),
    "border_strong": QColor(255, 255, 255, 40),
    
    # Typography
    "text_primary": QColor(255, 255, 255, 250),
    "text_secondary": QColor(255, 255, 255, 180),
    "text_muted": QColor(255, 255, 255, 120),
    
    # State colors
    "idle": QColor(16, 185, 129),          # Emerald
    "recording": QColor(239, 68, 68),      # Red
    "processing": QColor(245, 158, 11),    # Amber
    "success": QColor(34, 197, 94),        # Green
    "error": QColor(220, 38, 38),          # Crimson
}


class TranscriptionHistory:
    """Stores transcription history"""
    
    def __init__(self):
        self.filepath = HISTORY_FILE
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
    
    def add(self, text: str, duration: float, audio_file: str | None = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "duration": round(duration, 2),
            "words": len(text.split()),
            "chars": len(text)
        }
        if audio_file:
            entry["audio_file"] = audio_file
        self.entries.insert(0, entry)
        self.entries = self.entries[:500]
        self._save()
    
    def get_recent(self, n: int = 50) -> list[dict]:
        return self.entries[:n]


class SettingsDialog(QDialog):
    """Professional settings dialog with all app configuration"""
    
    def __init__(self, parent=None, app_ref=None):
        super().__init__(parent)
        self.app_ref = app_ref
        
        # Window setup
        self.setWindowTitle("Cognitive Flow Settings")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(480, 600)
        
        # Center on screen
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        self.move(
            screen_geometry.x() + (screen_geometry.width() - self.width()) // 2,
            screen_geometry.y() + (screen_geometry.height() - self.height()) // 2
        )
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Container with background
        container = QFrame()
        container.setObjectName("settingsContainer")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(32, 32, 24, 32)  # Less right margin for scrollbar
        container_layout.setSpacing(24)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        title = QLabel("Settings")
        title_font = QFont("Segoe UI", 18)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {COLORS['text_primary'].name()};")
        
        close_btn = QPushButton("x")
        close_btn.setFixedSize(32, 32)
        close_btn.clicked.connect(self.close)
        close_btn.setObjectName("closeButton")
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(close_btn)
        
        container_layout.addLayout(header_layout)
        
        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_content.setMaximumWidth(400)  # Constrain content width
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(40)
        scroll_layout.setContentsMargins(0, 8, 24, 8)  # More right margin for scrollbar
        
        # Input Device Selection
        scroll_layout.addWidget(self._create_section_header("Microphone"))
        input_layout = QVBoxLayout()
        input_layout.setSpacing(8)
        
        self.input_combo = NoScrollComboBox()
        self.input_combo.setObjectName("settingsCombo")
        self.input_combo.setFixedHeight(36)
        
        # Populate with available input devices
        self._input_devices = []  # Store (index, name) tuples
        if self.app_ref and hasattr(self.app_ref, 'get_input_devices'):
            self._input_devices = self.app_ref.get_input_devices()
            self.input_combo.addItem("System Default", None)
            for idx, name in self._input_devices:
                self.input_combo.addItem(name, idx)
            
            # Set current selection
            current_idx = getattr(self.app_ref, 'input_device_index', None)
            if current_idx is not None:
                for i in range(self.input_combo.count()):
                    if self.input_combo.itemData(i) == current_idx:
                        self.input_combo.setCurrentIndex(i)
                        break
        
        self.input_combo.currentIndexChanged.connect(self._on_input_device_changed)
        input_layout.addWidget(self.input_combo)
        
        input_desc = QLabel("Select which microphone to use for recording")
        input_desc.setWordWrap(True)
        input_desc.setStyleSheet(f"color: {COLORS['text_muted'].name()}; font-size: 11px;")
        input_layout.addWidget(input_desc)
        
        scroll_layout.addLayout(input_layout)

        # Backend Selection
        scroll_layout.addWidget(self._create_section_header("Transcription Engine"))
        backend_layout = QVBoxLayout()
        backend_layout.setSpacing(8)

        self.backend_combo = NoScrollComboBox()
        self.backend_combo.addItem("Whisper (OpenAI)", "whisper")
        self.backend_combo.addItem("Parakeet (NVIDIA) - Faster", "parakeet")
        self.backend_combo.setObjectName("settingsCombo")
        self.backend_combo.setFixedHeight(36)

        # Set current backend
        current_backend = getattr(self.app_ref, 'backend_type', 'whisper') if self.app_ref else 'whisper'
        for i in range(self.backend_combo.count()):
            if self.backend_combo.itemData(i) == current_backend:
                self.backend_combo.setCurrentIndex(i)
                break

        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        backend_layout.addWidget(self.backend_combo)

        self.backend_desc = QLabel("Whisper is well-tested. Parakeet is ~50x faster with better accuracy.")
        self.backend_desc.setWordWrap(True)
        self.backend_desc.setStyleSheet(f"color: {COLORS['text_muted'].name()}; font-size: 11px;")
        backend_layout.addWidget(self.backend_desc)

        scroll_layout.addLayout(backend_layout)

        # Model Selection
        scroll_layout.addWidget(self._create_section_header("Model"))
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)

        self.model_combo = NoScrollComboBox()
        self.model_combo.setObjectName("settingsCombo")
        self.model_combo.setFixedHeight(36)
        self._populate_model_combo()  # Populate based on current backend
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)

        self.model_desc = QLabel("")
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet(f"color: {COLORS['text_muted'].name()}; font-size: 11px;")
        self._update_model_description()
        model_layout.addWidget(self.model_desc)

        scroll_layout.addLayout(model_layout)
        
        # Output Options Section
        scroll_layout.addWidget(self._create_section_header("Output"))
        output_layout = QVBoxLayout()
        output_layout.setSpacing(8)
        
        self.trailing_space_cb = QCheckBox("Add space after transcription")
        self.trailing_space_cb.setChecked(
            self.app_ref.add_trailing_space if self.app_ref and hasattr(self.app_ref, 'add_trailing_space') else True
        )
        self.trailing_space_cb.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_primary'].name()};
                font-size: 12px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid {COLORS['border_strong'].name()};
                background-color: {COLORS['bg_elevated'].name()};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['idle'].name()};
                border-color: {COLORS['idle'].name()};
            }}
        """)
        self.trailing_space_cb.toggled.connect(self._on_trailing_space_changed)
        output_layout.addWidget(self.trailing_space_cb)
        
        trailing_space_desc = QLabel("Adds a space after each transcription so consecutive recordings don't run together")
        trailing_space_desc.setWordWrap(True)
        trailing_space_desc.setStyleSheet(f"color: {COLORS['text_muted'].name()}; font-size: 11px;")
        output_layout.addWidget(trailing_space_desc)
        
        scroll_layout.addLayout(output_layout)
        
        # Statistics Section
        scroll_layout.addWidget(self._create_section_header("Statistics"))
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(12)
        
        if self.app_ref and hasattr(self.app_ref, 'stats'):
            stats = self.app_ref.stats.stats
            
            stats_grid = QVBoxLayout()
            stats_grid.setSpacing(16)
            
            avg_speed = self.app_ref.stats.get_avg_speed_ratio()
            avg_words = self.app_ref.stats.get_avg_words_per_recording()
            speaking_wpm = self.app_ref.stats.get_speaking_speed_wpm()
            seconds_per_word = self.app_ref.stats.get_seconds_per_word()
            comparison = self.app_ref.stats.get_typing_vs_speaking_comparison()
            
            stats_items = [
                ("Total Recordings", f"{stats.get('total_records', 0):,}"),
                ("Words Transcribed", f"{stats.get('total_words', 0):,}"),
                ("Total Audio Time", f"{stats.get('total_seconds', 0) / 60:.1f} min"),
                ("", ""),
                ("Your Speaking Speed", f"{speaking_wpm:.0f} WPM"),
                ("Your Typing Speed", "30 WPM"),
                ("Seconds per Word", f"{seconds_per_word:.2f}s"),
                ("", ""),
                ("Time Typing Would Take", f"{comparison['typing_time']:.0f} min"),
                ("Time Speaking Took", f"{comparison['speaking_time']:.0f} min"),
                ("Time Saved", self.app_ref.stats.get_time_saved()),
                ("Efficiency Gain", f"{comparison['efficiency_ratio']:.1f}x faster"),
                ("", ""),
                ("Avg Words/Recording", f"{avg_words:.1f}"),
                ("Avg Processing Speed", f"{avg_speed:.2f}x" if avg_speed > 0 else "N/A"),
            ]
            
            for label, value in stats_items:
                if label == "":
                    stats_grid.addSpacing(12)
                    continue
                    
                item_layout = QHBoxLayout()
                item_layout.setSpacing(8)
                
                label_widget = QLabel(label)
                label_widget.setStyleSheet(f"color: {COLORS['text_secondary'].name()}; font-size: 12px;")
                
                value_widget = QLabel(value)
                value_font = QFont("Segoe UI", 12)
                value_font.setWeight(QFont.Weight.DemiBold)
                value_widget.setFont(value_font)
                value_widget.setStyleSheet(f"color: {COLORS['text_primary'].name()};")
                
                item_layout.addWidget(label_widget)
                item_layout.addStretch()
                item_layout.addWidget(value_widget)
                
                stats_grid.addLayout(item_layout)
            
            stats_layout.addLayout(stats_grid)
        
        scroll_layout.addLayout(stats_layout)
        
        # Recent Transcriptions
        scroll_layout.addWidget(self._create_section_header("Recent Transcriptions"))
        
        if self.app_ref and hasattr(self.app_ref, 'ui') and self.app_ref.ui:
            recent = self.app_ref.ui.history.get_recent(5)
            
            for entry in recent:
                entry_widget = self._create_history_entry(entry)
                scroll_layout.addWidget(entry_widget)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        container_layout.addWidget(scroll)
        
        main_layout.addWidget(container)
        self.setLayout(main_layout)
        
        # Styling
        self.setStyleSheet(f"""
            #settingsContainer {{
                background-color: {COLORS['bg_secondary'].name()};
                border-radius: 12px;
                border: 1px solid {COLORS['border_subtle'].name()};
            }}
            
            #closeButton {{
                background-color: transparent;
                border: none;
                color: {COLORS['text_secondary'].name()};
                font-size: 16px;
                border-radius: 6px;
            }}
            
            #closeButton:hover {{
                background-color: {COLORS['bg_hover'].name()};
                color: {COLORS['text_primary'].name()};
            }}
            
            #settingsCombo {{
                background-color: {COLORS['bg_elevated'].name()};
                border: 1px solid {COLORS['border_subtle'].name()};
                border-radius: 6px;
                padding: 8px 12px;
                color: {COLORS['text_primary'].name()};
                font-size: 12px;
            }}
            
            #settingsCombo:hover {{
                border-color: {COLORS['border_strong'].name()};
            }}
            
            #settingsCombo::drop-down {{
                border: none;
                width: 20px;
            }}
            
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            
            QScrollArea > QWidget > QWidget {{
                background-color: transparent;
            }}
            
            QScrollBar:vertical {{
                background-color: rgba(255, 255, 255, 5);
                width: 8px;
                margin: 4px 2px 4px 0px;
                border-radius: 4px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: rgba(255, 255, 255, 40);
                border-radius: 4px;
                min-height: 30px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: rgba(255, 255, 255, 60);
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background-color: transparent;
            }}
        """)
        
        # Shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 120))
        shadow.setOffset(0, 8)
        container.setGraphicsEffect(shadow)
        
        # Fade in animation
        self.setWindowOpacity(0)
        self.fade_in = QPropertyAnimation(self, b"windowOpacity")
        self.fade_in.setDuration(200)
        self.fade_in.setStartValue(0)
        self.fade_in.setEndValue(1)
        self.fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        QTimer.singleShot(10, self.fade_in.start)
    
    def _create_section_header(self, text):
        """Create a section header label"""
        label = QLabel(text)
        font = QFont("Segoe UI", 11)
        font.setWeight(QFont.Weight.DemiBold)
        label.setFont(font)
        label.setStyleSheet(f"color: {COLORS['text_secondary'].name()}; margin-top: 4px;")
        return label
    
    def _create_history_entry(self, entry):
        """Create a history entry widget"""
        widget = QFrame()
        widget.setObjectName("historyEntry")
        widget.setStyleSheet(f"""
            #historyEntry {{
                background-color: rgba({COLORS['bg_elevated'].red()}, {COLORS['bg_elevated'].green()}, {COLORS['bg_elevated'].blue()}, {COLORS['bg_elevated'].alpha()});
                border-radius: 6px;
                padding: 12px;
            }}
            
            #historyEntry:hover {{
                background-color: rgba(60, 60, 64, 255);
            }}
        """)
        
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 14, 16, 14)
        
        # Text preview
        text_preview = entry.get('text', '')[:100]
        if len(entry.get('text', '')) > 100:
            text_preview += "..."
        
        text_label = QLabel(text_preview)
        text_label.setWordWrap(True)
        text_label.setStyleSheet(f"color: {COLORS['text_primary'].name()}; font-size: 12px;")
        layout.addWidget(text_label)
        
        # Meta info
        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(12)
        
        words_label = QLabel(f"{entry.get('words', 0)} words")
        words_label.setStyleSheet(f"color: {COLORS['text_muted'].name()}; font-size: 10px;")
        
        duration_label = QLabel(f"{entry.get('duration', 0)}s")
        duration_label.setStyleSheet(f"color: {COLORS['text_muted'].name()}; font-size: 10px;")
        
        meta_layout.addWidget(words_label)
        meta_layout.addWidget(duration_label)
        meta_layout.addStretch()
        
        layout.addLayout(meta_layout)
        
        # Make clickable to copy
        widget.mousePressEvent = lambda e: self._copy_entry(entry.get('text', ''))
        widget.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        return widget
    
    def _copy_entry(self, text):
        """Copy entry text to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        print(f"[Clipboard] Copied: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    def _get_current_backend(self) -> str:
        """Get current backend type from app or combo."""
        if self.app_ref and hasattr(self.app_ref, 'backend_type'):
            return self.app_ref.backend_type
        return self.backend_combo.currentData() or 'whisper'

    def _populate_model_combo(self):
        """Populate model combo based on current backend."""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        backend = self._get_current_backend()

        if backend == 'parakeet':
            self.model_combo.addItems([
                "nemo-parakeet-tdt-0.6b-v2",
                "nemo-parakeet-tdt-0.6b-v3",
                "nemo-parakeet-tdt-0.6b-v3-int8",
            ])
            current = getattr(self.app_ref, 'parakeet_model', 'nemo-parakeet-tdt-0.6b-v2') if self.app_ref else 'nemo-parakeet-tdt-0.6b-v2'
        else:
            self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
            current = getattr(self.app_ref, 'model_name', 'medium') if self.app_ref else 'medium'

        # Set current selection
        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.model_combo.blockSignals(False)

    def _update_model_description(self):
        """Update model description based on current selection."""
        backend = self._get_current_backend()
        model = self.model_combo.currentText()

        if backend == 'parakeet':
            descriptions = {
                "nemo-parakeet-tdt-0.6b-v2": "English only, fastest, ~2GB VRAM",
                "nemo-parakeet-tdt-0.6b-v3": "25 European languages, slightly larger",
                "nemo-parakeet-tdt-0.6b-v3-int8": "Quantized v3, smaller & faster",
            }
        else:
            descriptions = {
                "tiny": "Fastest but least accurate - good for testing",
                "base": "Fast with reasonable accuracy",
                "small": "Good balance for most users",
                "medium": "Best balance of speed and accuracy (recommended)",
                "large": "Most accurate but slower - requires more RAM"
            }

        self.model_desc.setText(descriptions.get(model, ""))

    def _on_backend_changed(self, index):
        """Handle backend selection change."""
        if not self.app_ref:
            return

        backend = self.backend_combo.itemData(index)
        if backend == self.app_ref.backend_type:
            return

        # Skip if already loading
        if hasattr(self.app_ref, 'model_loading') and self.app_ref.model_loading:
            print(f"[Settings] Model already loading - ignoring backend change")
            self.backend_combo.blockSignals(True)
            for i in range(self.backend_combo.count()):
                if self.backend_combo.itemData(i) == self.app_ref.backend_type:
                    self.backend_combo.setCurrentIndex(i)
                    break
            self.backend_combo.blockSignals(False)
            return

        self.app_ref.backend_type = backend
        if hasattr(self.app_ref, 'save_config'):
            self.app_ref.save_config()

        # Update model combo for new backend
        self._populate_model_combo()
        self._update_model_description()

        # Reload model with new backend
        if hasattr(self.app_ref, 'load_model'):
            model = self.model_combo.currentText()
            print(f"[Settings] Switching to {backend} ({model})...")
            self.app_ref.load_model()

    def _on_model_changed(self, model_name):
        """Handle model selection change - reloads model live"""
        self._update_model_description()

        if not self.app_ref:
            return

        backend = self._get_current_backend()

        # Determine which config property to update
        if backend == 'parakeet':
            if hasattr(self.app_ref, 'parakeet_model') and self.app_ref.parakeet_model == model_name:
                return
            self.app_ref.parakeet_model = model_name
        else:
            if hasattr(self.app_ref, 'model_name') and self.app_ref.model_name == model_name:
                return
            self.app_ref.model_name = model_name

        # Skip if already loading a model (prevents OOM from rapid clicking)
        if hasattr(self.app_ref, 'model_loading') and self.app_ref.model_loading:
            print(f"[Settings] Model already loading - ignoring {model_name}")
            self._populate_model_combo()  # Reset to current
            return

        if hasattr(self.app_ref, 'save_config'):
            self.app_ref.save_config()

        # Reload model in background
        if hasattr(self.app_ref, 'load_model'):
            print(f"[Settings] Loading {model_name}...")
            self.app_ref.load_model()
    
    def _on_trailing_space_changed(self, checked):
        """Handle trailing space toggle"""
        if self.app_ref:
            self.app_ref.add_trailing_space = checked
            if hasattr(self.app_ref, 'save_config'):
                self.app_ref.save_config()
            print(f"[Settings] Trailing space: {'on' if checked else 'off'}")
    
    def _on_input_device_changed(self, index):
        """Handle input device selection change"""
        if self.app_ref:
            device_idx = self.input_combo.itemData(index)
            device_name = self.input_combo.currentText()
            self.app_ref.input_device_index = device_idx
            if hasattr(self.app_ref, 'save_config'):
                self.app_ref.save_config()
            print(f"[Settings] Input device: {device_name}")
    
    def closeEvent(self, event):
        """Fade out on close"""
        self.fade_out = QPropertyAnimation(self, b"windowOpacity")
        self.fade_out.setDuration(150)
        self.fade_out.setStartValue(1)
        self.fade_out.setEndValue(0)
        self.fade_out.setEasingCurve(QEasingCurve.Type.InCubic)
        self.fade_out.finished.connect(lambda: super(SettingsDialog, self).closeEvent(event))
        self.fade_out.start()
        event.ignore()


class FloatingIndicator(QWidget):
    """Floating indicator widget - collapses to dot when idle"""
    
    EXPANDED_WIDTH = 180
    COLLAPSED_WIDTH = 44  # Just enough for the dot with padding
    COLLAPSE_DELAY_MS = 3000  # 3 seconds
    
    def __init__(self, on_click=None, get_last_transcription=None, show_settings=None, retry_last=None):
        super().__init__()
        self.on_click = on_click
        self.get_last_transcription = get_last_transcription
        self.show_settings_callback = show_settings
        self.retry_last_callback = retry_last
        
        # State
        self.state = "loading"
        self.status_text = "Loading..."
        self._is_hovered = False
        self._is_collapsed = False
        self._current_width = self.EXPANDED_WIDTH
        
        # Animatable properties
        self._circle_color = QColor(COLORS["processing"])
        self._text_color = QColor(COLORS["text_secondary"])
        self._hover_opacity = 0.0
        
        # Window setup
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        
        self.setFixedSize(self.EXPANDED_WIDTH, 40)
        
        # Position - store screen info for repositioning
        screen = QGuiApplication.primaryScreen()
        self._screen_geometry = screen.availableGeometry()
        self._margin = 24
        self._base_y = self._screen_geometry.y() + self._screen_geometry.height() - self.height() - self._margin
        self._update_position()
        print(f"[UI] Window positioned at ({self.x()}, {self.y()})")
        
        # Collapse timer
        self._collapse_timer = QTimer(self)
        self._collapse_timer.setSingleShot(True)
        self._collapse_timer.timeout.connect(self._collapse)
        
        # Width animation
        self._width_animation = QPropertyAnimation(self, b"indicator_width")
        self._width_animation.setDuration(250)
        self._width_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Layout
        layout = QHBoxLayout()
        layout.setContentsMargins(40, 10, 14, 10)
        layout.setSpacing(0)
        
        # Status label
        self.status_label = QLabel(self.status_text)
        font = QFont("Segoe UI", 9)
        font.setWeight(QFont.Weight.DemiBold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.2)
        self.status_label.setFont(font)
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Dragging
        self.drag_start = None
        
        # Animations
        self.color_animation = QPropertyAnimation(self, b"circle_color")
        self.text_color_animation = QPropertyAnimation(self, b"text_color")
        self.hover_animation = QPropertyAnimation(self, b"hover_opacity")
        
        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Enable mouse tracking for hover
        self.setMouseTracking(True)
        
        # Entry animation with fallback to ensure visibility
        self.setWindowOpacity(0)
        self.entry_anim = QPropertyAnimation(self, b"windowOpacity")
        self.entry_anim.setDuration(400)
        self.entry_anim.setStartValue(0)
        self.entry_anim.setEndValue(1)
        self.entry_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.entry_anim.finished.connect(lambda: self.setWindowOpacity(1))  # Ensure opacity
        QTimer.singleShot(100, self.entry_anim.start)
        # Fallback: force opacity to 1 after animation should have completed
        QTimer.singleShot(600, lambda: self.setWindowOpacity(1) if self.windowOpacity() < 1 else None)
    
    # Qt Properties
    def get_circle_color(self):
        return self._circle_color
    
    def set_circle_color(self, color):
        self._circle_color = color
        self.update()
    
    circle_color = pyqtProperty(QColor, get_circle_color, set_circle_color)
    
    def get_text_color(self):
        return self._text_color
    
    def set_text_color(self, color):
        self._text_color = color
        self.status_label.setStyleSheet(f"color: {color.name()};")
    
    text_color = pyqtProperty(QColor, get_text_color, set_text_color)
    
    def get_indicator_width(self):
        return self._current_width
    
    def set_indicator_width(self, value):
        self._current_width = int(value)
        self.setFixedWidth(self._current_width)
        self._update_position()
        # Hide text when collapsed
        if self._current_width < self.EXPANDED_WIDTH * 0.6:
            self.status_label.hide()
        else:
            self.status_label.show()
        self.update()
    
    indicator_width = pyqtProperty(int, get_indicator_width, set_indicator_width)
    
    def _update_position(self):
        """Keep indicator anchored to bottom-right"""
        x = self._screen_geometry.x() + self._screen_geometry.width() - self._current_width - self._margin
        self.move(x, self._base_y)

    def refresh_geometry(self):
        """Refresh screen geometry and reposition - call if display config changed"""
        screen = QGuiApplication.primaryScreen()
        self._screen_geometry = screen.availableGeometry()
        self._base_y = self._screen_geometry.y() + self._screen_geometry.height() - self.height() - self._margin
        self._update_position()

    def ensure_visible(self):
        """Force the indicator to be visible and properly positioned"""
        # Refresh screen geometry in case display changed
        self.refresh_geometry()
        # Ensure opacity is 1
        self.setWindowOpacity(1)
        # Show and raise to top
        self.show()
        self.raise_()
        self.activateWindow()
        # Force repaint
        self.update()
        print(f"[UI] ensure_visible: pos=({self.x()}, {self.y()}) opacity={self.windowOpacity()} visible={self.isVisible()}")
    
    def _collapse(self):
        """Animate to collapsed state"""
        if self._is_collapsed or self.state != "idle":
            return
        self._is_collapsed = True
        self._width_animation.stop()
        self._width_animation.setStartValue(self._current_width)
        self._width_animation.setEndValue(self.COLLAPSED_WIDTH)
        self._width_animation.start()
    
    def _expand(self):
        """Animate to expanded state"""
        if not self._is_collapsed:
            return
        self._is_collapsed = False
        self._collapse_timer.stop()
        self._width_animation.stop()
        self._width_animation.setStartValue(self._current_width)
        self._width_animation.setEndValue(self.EXPANDED_WIDTH)
        self._width_animation.start()
    
    def _start_collapse_timer(self):
        """Start timer to collapse after delay"""
        self._collapse_timer.stop()
        self._collapse_timer.start(self.COLLAPSE_DELAY_MS)
    
    def get_hover_opacity(self):
        return self._hover_opacity
    
    def set_hover_opacity(self, value):
        self._hover_opacity = value
        self.update()
    
    hover_opacity = pyqtProperty(float, get_hover_opacity, set_hover_opacity)
    
    def paintEvent(self, event):
        """Render the indicator"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        # Background path
        bg_rect = self.rect()
        path = QPainterPath()
        path.addRoundedRect(bg_rect.x(), bg_rect.y(), bg_rect.width(), bg_rect.height(), 8, 8)
        
        # Drop shadow
        shadow_offset = 3
        shadow_path = QPainterPath()
        shadow_path.addRoundedRect(
            bg_rect.x() + shadow_offset,
            bg_rect.y() + shadow_offset,
            bg_rect.width(),
            bg_rect.height(),
            8, 8
        )
        painter.fillPath(shadow_path, QColor(0, 0, 0, 60))
        
        # Main background
        painter.fillPath(path, COLORS["bg_primary"])
        
        # Hover overlay
        if self._hover_opacity > 0:
            hover_color = QColor(COLORS["bg_hover"])
            hover_color.setAlpha(int(15 * self._hover_opacity))
            painter.fillPath(path, hover_color)
        
        # Border
        pen = QPen(COLORS["border_subtle"])
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Status dot
        dot_x = 22
        dot_y = self.height() // 2
        dot_radius = 5
        
        # Glow when active
        if self.state in ["recording", "processing"]:
            glow_radius = dot_radius + 6
            glow = QRadialGradient(dot_x, dot_y, glow_radius)
            glow_color = QColor(self._circle_color)
            glow_color.setAlpha(50)
            glow.setColorAt(0, glow_color)
            glow_color.setAlpha(0)
            glow.setColorAt(1, glow_color)
            painter.setBrush(glow)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                int(dot_x - glow_radius),
                int(dot_y - glow_radius),
                int(glow_radius * 2),
                int(glow_radius * 2)
            )
        
        # Dot with gradient
        gradient = QRadialGradient(
            dot_x - dot_radius * 0.4,
            dot_y - dot_radius * 0.4,
            dot_radius * 1.6
        )
        lighter = QColor(self._circle_color).lighter(120)
        gradient.setColorAt(0, lighter)
        gradient.setColorAt(1, self._circle_color)
        
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            int(dot_x - dot_radius),
            int(dot_y - dot_radius),
            int(dot_radius * 2),
            int(dot_radius * 2)
        )
    
    def set_state(self, state: str, status_text: str | None = None):
        """Update state with smooth transitions"""
        self.state = state
        
        if status_text:
            self.status_text = status_text
        else:
            status_map = {
                "loading": "Loading...",
                "idle": "Ready",
                "recording": "Recording",
                "processing": "Processing"
            }
            self.status_text = status_map.get(state, "Ready")
        
        self.status_label.setText(self.status_text)
        
        # State colors
        state_config = {
            "loading": (COLORS["processing"], COLORS["text_secondary"]),
            "idle": (COLORS["idle"], COLORS["text_secondary"]),
            "recording": (COLORS["recording"], COLORS["text_primary"]),
            "processing": (COLORS["processing"], COLORS["text_primary"]),
        }
        
        dot_color, text_color = state_config.get(state, state_config["idle"])

        # Force immediate color update (in case animation fails)
        self._circle_color = dot_color
        self._text_color = text_color
        self.status_label.setStyleSheet(f"color: {text_color.name()};")
        self.update()  # Force repaint

        # Smooth transitions (will animate from current to same color, but ensures repaint)
        self.color_animation.stop()
        self.color_animation.setDuration(300)
        self.color_animation.setStartValue(dot_color)
        self.color_animation.setEndValue(dot_color)
        self.color_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.color_animation.start()

        self.text_color_animation.stop()
        self.text_color_animation.setDuration(300)
        self.text_color_animation.setStartValue(text_color)
        self.text_color_animation.setEndValue(text_color)
        self.text_color_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.text_color_animation.start()
        
        # Collapse/expand behavior
        if state in ("recording", "processing", "loading"):
            # Expand immediately when active
            self._collapse_timer.stop()
            self._expand()
        elif state == "idle":
            # Start collapse timer when idle
            self._start_collapse_timer()
    
    def show_context_menu(self, position):
        """Context menu"""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS['bg_secondary'].name()};
                border: 1px solid {COLORS['border_subtle'].name()};
                border-radius: 8px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 8px 24px 8px 12px;
                color: {COLORS['text_primary'].name()};
                border-radius: 5px;
                font-size: 12px;
            }}
            QMenu::item:selected {{
                background-color: rgba(60, 60, 64, 255);
            }}
        """)
        
        copy_action = QAction("Copy Last Transcription", self)
        copy_action.triggered.connect(self.copy_last_transcription)
        menu.addAction(copy_action)

        retry_action = QAction("Retry Last Recording", self)
        retry_action.triggered.connect(self.retry_last_recording)
        menu.addAction(retry_action)

        menu.addSeparator()

        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self.show_settings_callback)
        menu.addAction(settings_action)

        menu.exec(self.mapToGlobal(position))
    
    def copy_last_transcription(self):
        """Copy last transcription"""
        if self.get_last_transcription:
            text = self.get_last_transcription()
            if text:
                clipboard = QApplication.clipboard()
                clipboard.setText(text)
                print(f"[Clipboard] Copied: {text[:50]}{'...' if len(text) > 50 else ''}")

    def retry_last_recording(self):
        """Retry transcription from last saved audio"""
        if self.retry_last_callback:
            self.retry_last_callback()
    
    def enterEvent(self, event):
        self._is_hovered = True
        self.hover_animation.stop()
        self.hover_animation.setDuration(200)
        self.hover_animation.setStartValue(self._hover_opacity)
        self.hover_animation.setEndValue(1.0)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.hover_animation.start()
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        # Expand on hover if collapsed
        if self._is_collapsed:
            self._expand()
    
    def leaveEvent(self, event):
        self._is_hovered = False
        self.hover_animation.stop()
        self.hover_animation.setDuration(200)
        self.hover_animation.setStartValue(self._hover_opacity)
        self.hover_animation.setEndValue(0.0)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.hover_animation.start()
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        # Restart collapse timer when leaving (if idle)
        if self.state == "idle":
            self._start_collapse_timer()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = event.globalPosition().toPoint()
    
    def mouseMoveEvent(self, event):
        if self.drag_start:
            delta = event.globalPosition().toPoint() - self.drag_start
            self.move(self.pos() + delta)
            self.drag_start = event.globalPosition().toPoint()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.on_click and self.drag_start:
                delta = event.globalPosition().toPoint() - self.drag_start
                if abs(delta.x()) < 5 and abs(delta.y()) < 5:
                    self.on_click()
            self.drag_start = None


class CognitiveFlowUI(QObject):
    """UI coordinator"""
    
    state_changed = pyqtSignal(str, str)
    show_settings_signal = pyqtSignal()
    
    def __init__(self, app):
        # Create QApplication FIRST before calling super().__init__()
        # This ensures QObject is created in the correct thread context
        self.qt_app = QApplication.instance()
        if self.qt_app is None:
            self.qt_app = QApplication(sys.argv)
        
        super().__init__()
        self.app = app
        self.indicator = None
        self.history = TranscriptionHistory()
        self.settings_dialog = None
        
        # Use QueuedConnection to ensure UI updates happen on Qt main thread
        self.state_changed.connect(self._update_indicator_state, Qt.ConnectionType.QueuedConnection)
        self.show_settings_signal.connect(self._show_settings_on_main_thread, Qt.ConnectionType.QueuedConnection)
    
    def start(self):
        """Start Qt application"""
        self.indicator = FloatingIndicator(
            on_click=self.app.toggle_recording if hasattr(self.app, 'toggle_recording') else None,
            get_last_transcription=self.get_last_transcription,
            show_settings=self.show_settings,
            retry_last=self.app.retry_last if hasattr(self.app, 'retry_last') else None
        )
        self.indicator.show()
    
    def _update_indicator_state(self, state: str, status_text: str):
        """Thread-safe state update"""
        if self.indicator:
            self.indicator.set_state(state, status_text if status_text else None)
    
    def set_state(self, state: str, status_text: str | None = None):
        """Update state"""
        if status_text is None:
            status_text = ""
        self.state_changed.emit(state, status_text)
        # Force immediate Qt event processing to ensure state change is visible
        if self.qt_app:
            self.qt_app.processEvents()
    
    def get_last_transcription(self):
        """Get last transcription"""
        recent = self.history.get_recent(1)
        if recent:
            return recent[0].get("text", "")
        return None
    
    def add_transcription(self, text: str, duration: float, audio_file: str | None = None):
        """Add to history"""
        self.history.add(text, duration, audio_file)
    
    def show(self):
        """Show the overlay indicator"""
        if self.indicator:
            self.indicator.ensure_visible()
    
    def hide(self):
        """Hide the overlay indicator"""
        if self.indicator:
            self.indicator.hide()
    
    def show_settings(self):
        """Show settings dialog - thread safe"""
        self.show_settings_signal.emit()
    
    def _show_settings_on_main_thread(self):
        """Actually show settings - called on Qt main thread"""
        if not self.settings_dialog:
            self.settings_dialog = SettingsDialog(parent=self.indicator, app_ref=self.app)
        self.settings_dialog.exec()
    
    def destroy(self):
        """Clean up"""
        if self.indicator:
            self.indicator.close()
