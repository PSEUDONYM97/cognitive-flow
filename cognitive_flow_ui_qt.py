"""
Cognitive Flow UI - Maximum Polish Edition
Every detail refined. Professional settings screen. Perfect UX.
"""

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                              QMenu, QDialog, QPushButton, QComboBox, QSpinBox, 
                              QCheckBox, QScrollArea, QFrame, QGraphicsDropShadowEffect)
from PyQt6.QtCore import (Qt, QPropertyAnimation, QEasingCurve, pyqtProperty, 
                          QObject, pyqtSignal, QTimer, QPoint, QSize)
from PyQt6.QtGui import (QPainter, QColor, QRadialGradient, QFont, QAction, 
                         QPainterPath, QLinearGradient, QPen, QCursor)
import sys
import json
from pathlib import Path
from datetime import datetime

# Professional color system
COLORS = {
    # Backgrounds with proper alpha
    "bg_primary": QColor(16, 16, 18, 165),       # Main UI background
    "bg_secondary": QColor(28, 28, 32, 230),     # Settings/dialogs
    "bg_elevated": QColor(38, 38, 42, 245),      # Elevated surfaces
    "bg_hover": QColor(255, 255, 255, 8),        # Hover states - very subtle (was 15)
    
    # Borders and dividers
    "border_subtle": QColor(255, 255, 255, 20),
    "border_strong": QColor(255, 255, 255, 40),
    
    # Typography
    "text_primary": QColor(255, 255, 255, 250),
    "text_secondary": QColor(255, 255, 255, 180),
    "text_muted": QColor(255, 255, 255, 120),
    
    # State colors - vibrant and clear
    "idle": QColor(16, 185, 129),          # Emerald
    "recording": QColor(239, 68, 68),      # Red
    "processing": QColor(245, 158, 11),    # Amber
    "success": QColor(34, 197, 94),        # Green
    "error": QColor(220, 38, 38),          # Crimson
}


class TranscriptionHistory:
    """Stores transcription history"""
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
        from PyQt6.QtGui import QGuiApplication
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
        container_layout.setContentsMargins(40, 36, 40, 36)  # Even more breathing room
        container_layout.setSpacing(32)  # Increased spacing between major sections
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        title = QLabel("Settings")
        title_font = QFont("Segoe UI", 18)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {COLORS['text_primary'].name()};")
        
        close_btn = QPushButton("✕")
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
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(40)  # More space between sections
        scroll_layout.setContentsMargins(0, 8, 16, 8)  # Add padding on sides and top/bottom
        
        # Model Selection
        scroll_layout.addWidget(self._create_section_header("Transcription Model"))
        model_layout = QVBoxLayout()
        model_layout.setSpacing(8)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        
        # Set current model from app
        if self.app_ref and hasattr(self.app_ref, 'model_name'):
            self.model_combo.setCurrentText(self.app_ref.model_name)
        else:
            self.model_combo.setCurrentText("medium")
        
        self.model_combo.setObjectName("settingsCombo")
        self.model_combo.setFixedHeight(36)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        self.model_desc = QLabel("Medium provides the best balance of speed and accuracy")
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet(f"color: {COLORS['text_muted'].name()}; font-size: 11px;")
        model_layout.addWidget(self.model_desc)
        
        scroll_layout.addLayout(model_layout)
        
        # Statistics Section
        scroll_layout.addWidget(self._create_section_header("Statistics"))
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(12)
        
        if self.app_ref and hasattr(self.app_ref, 'stats'):
            stats = self.app_ref.stats.stats
            
            stats_grid = QVBoxLayout()
            stats_grid.setSpacing(16)  # More space between stat rows
            
            stats_items = [
                ("Total Recordings", f"{stats.get('total_records', 0):,}"),
                ("Words Transcribed", f"{stats.get('total_words', 0):,}"),
                ("Time Saved", self.app_ref.stats.get_time_saved()),
                ("Total Audio", f"{stats.get('total_seconds', 0) / 60:.1f} minutes"),
            ]
            
            for label, value in stats_items:
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
            
            QScrollBar:vertical {{
                background-color: transparent;
                width: 8px;
                margin: 0px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {COLORS['border_subtle'].name()};
                border-radius: 4px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {COLORS['border_strong'].name()};
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
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
        layout.setSpacing(10)  # More space between text and meta
        layout.setContentsMargins(16, 14, 16, 14)  # More padding inside history entries
        
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
    
    def _on_model_changed(self, model_name):
        """Handle model selection change"""
        print(f"[Settings] Model changed to: {model_name}")
        
        # Update description based on model
        descriptions = {
            "tiny": "Fastest but least accurate - good for testing",
            "base": "Fast with reasonable accuracy",
            "small": "Good balance for most users",
            "medium": "Best balance of speed and accuracy (recommended)",
            "large": "Most accurate but slower - requires more RAM"
        }
        
        self.model_desc.setText(descriptions.get(model_name, ""))
        
        # Save to app config (requires restart to take effect)
        if self.app_ref:
            if hasattr(self.app_ref, 'model_name'):
                self.app_ref.model_name = model_name
            if hasattr(self.app_ref, 'save_config'):
                self.app_ref.save_config()
                print(f"[Settings] Model saved. Restart app to use {model_name} model.")
    
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


class MaxPolishIndicator(QWidget):
    """Maximum polish floating indicator - every detail perfect"""
    
    def __init__(self, on_click=None, get_last_transcription=None, show_settings=None):
        super().__init__()
        self.on_click = on_click
        self.get_last_transcription = get_last_transcription
        self.show_settings_callback = show_settings
        
        # State
        self.state = "loading"
        self.status_text = "Loading..."
        self._is_hovered = False
        
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
        
        # Sleeker size: 180 x 40 (more compact)
        self.setFixedSize(180, 40)
        
        # Position - use availableGeometry to account for taskbar
        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()  # Excludes taskbar
        margin = 24
        x = screen_geometry.x() + screen_geometry.width() - self.width() - margin
        y = screen_geometry.y() + screen_geometry.height() - self.height() - margin
        self.move(x, y)
        print(f"[UI] Window positioned at ({x}, {y}) on screen {screen_geometry.width()}x{screen_geometry.height()}")
        
        # Layout - account for dot space
        layout = QHBoxLayout()
        layout.setContentsMargins(40, 10, 14, 10)  # Left margin = 40 to make room for dot at x=22
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
        
        # Entry animation - store as instance variable to prevent garbage collection
        self.setWindowOpacity(0)
        self.entry_anim = QPropertyAnimation(self, b"windowOpacity")
        self.entry_anim.setDuration(400)
        self.entry_anim.setStartValue(0)
        self.entry_anim.setEndValue(1)
        self.entry_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        QTimer.singleShot(100, self.entry_anim.start)
    
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
    
    def get_hover_opacity(self):
        return self._hover_opacity
    
    def set_hover_opacity(self, value):
        self._hover_opacity = value
        self.update()
    
    hover_opacity = pyqtProperty(float, get_hover_opacity, set_hover_opacity)
    
    def paintEvent(self, event):
        """Maximum polish rendering"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # Background path
        bg_rect = self.rect().adjusted(0, 0, 0, 0)
        path = QPainterPath()
        path.addRoundedRect(bg_rect.x(), bg_rect.y(), bg_rect.width(), bg_rect.height(), 8, 8)
        
        # Drop shadow - more visible
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
        
        # Status dot - perfectly centered
        dot_x = 22
        dot_y = self.height() // 2
        dot_radius = 5
        
        # Subtle glow when active
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
        
        # Smooth transitions
        self.color_animation.stop()
        self.color_animation.setDuration(300)
        self.color_animation.setStartValue(self._circle_color)
        self.color_animation.setEndValue(dot_color)
        self.color_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.color_animation.start()
        
        self.text_color_animation.stop()
        self.text_color_animation.setDuration(300)
        self.text_color_animation.setStartValue(self._text_color)
        self.text_color_animation.setEndValue(text_color)
        self.text_color_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.text_color_animation.start()
    
    def show_context_menu(self, position):
        """Styled context menu"""
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
    
    def enterEvent(self, event):
        """Hover enter"""
        self._is_hovered = True
        self.hover_animation.stop()
        self.hover_animation.setDuration(200)
        self.hover_animation.setStartValue(self._hover_opacity)
        self.hover_animation.setEndValue(1.0)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.hover_animation.start()
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    
    def leaveEvent(self, event):
        """Hover leave"""
        self._is_hovered = False
        self.hover_animation.stop()
        self.hover_animation.setDuration(200)
        self.hover_animation.setStartValue(self._hover_opacity)
        self.hover_animation.setEndValue(0.0)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.hover_animation.start()
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    
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
    """Maximum polish UI coordinator"""
    
    state_changed = pyqtSignal(str, str)
    
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.qt_app = None
        self.indicator = None
        self.history = TranscriptionHistory()
        self.settings_dialog = None
        
        self.state_changed.connect(self._update_indicator_state)
    
    def start(self):
        """Start Qt application"""
        self.qt_app = QApplication.instance()
        if self.qt_app is None:
            self.qt_app = QApplication(sys.argv)
        
        self.indicator = MaxPolishIndicator(
            on_click=self.app.toggle_recording if hasattr(self.app, 'toggle_recording') else None,
            get_last_transcription=self.get_last_transcription,
            show_settings=self.show_settings
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
    
    def get_last_transcription(self):
        """Get last transcription"""
        recent = self.history.get_recent(1)
        if recent:
            return recent[0].get("text", "")
        return None
    
    def add_transcription(self, text: str, duration: float):
        """Add to history"""
        self.history.add(text, duration)
    
    def show_settings(self):
        """Show settings dialog"""
        if not self.settings_dialog:
            self.settings_dialog = SettingsDialog(parent=self.indicator, app_ref=self.app)
        self.settings_dialog.exec()
    
    def destroy(self):
        """Clean up"""
        if self.indicator:
            self.indicator.close()


if __name__ == "__main__":
    # Test
    class TestApp:
        def __init__(self, ui_ref):
            self.ui = ui_ref
            self.recording = False
            
            # Mock stats
            class MockStats:
                def __init__(self):
                    self.stats = {
                        'total_records': 1514,
                        'total_words': 70779,
                        'total_seconds': 15420,
                    }
                def get_time_saved(self):
                    return "21.6 hours"
            
            self.stats = MockStats()
        
        def toggle_recording(self):
            self.recording = not self.recording
            if self.recording:
                print("[Test] Recording")
                self.ui.set_state("recording")
            else:
                print("[Test] Processing")
                self.ui.set_state("processing")
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(2000, lambda: self.ui.set_state("idle"))
    
    test_app = TestApp(None)
    ui = CognitiveFlowUI(test_app)
    test_app.ui = ui
    ui.start()
    ui.set_state("idle", "Ready")
    
    print("Maximum Polish UI Test")
    print("- Hover for visual feedback")
    print("- Right-click for menu with settings")
    print("- Click to toggle recording")
    
    sys.exit(ui.qt_app.exec())
