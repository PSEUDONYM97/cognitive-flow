# Voice-to-Text Overlay UI/UX Research: What Makes a Great Recording Indicator

**Research Date**: 2026-03-09
**Researcher**: AI Research Specialist
**Sources Analyzed**: 30+
**Research Scope**: Overlay UI patterns, visual design, animation, state indicators for Windows voice dictation apps

---

## Executive Summary

1. **The pill/capsule shape with collapse-to-dot behavior is the dominant pattern** in 2025-2026. Google Gemini Live, SuperWhisper, and MacWhisper all use variants of an expanding pill that collapses to minimal footprint when idle. This is now the de facto standard.

2. **Animated waveforms outperform static level bars** for perceived quality and user engagement - they feel "alive" and signal activity more naturally. But they're expensive to implement well in PyQt6 canvas painting. A hybrid approach (waveform bars + level-responsive glow) is the sweet spot.

3. **Color conventions are more established than you'd think**: Red = recording (universal, rooted in broadcast history), Amber/Orange = processing/thinking, Green = ready/success, Blue = loading. Deviating from this creates cognitive friction.

4. **The current Cognitive Flow indicator is architecturally sound but has specific weaknesses**: The audio level bar is at the bottom of the widget and easy to miss. The dot glow is present but subtle. The collapse is good but the 3-second timer may be too fast.

5. **The biggest gap is the recording state feedback** - a spinning/pulsing animation during recording is now expected by users who've used any modern voice app. The current static red dot doesn't convey "actively capturing" well enough.

---

## Research Overview

**Purpose**: Define what great looks like for the Cognitive Flow floating overlay indicator, grounded in how the best voice-to-text and AI assistant apps handle this problem in 2025-2026.

**Scope**: Visual design patterns, animation techniques, color systems, state transitions, and specific failure modes to avoid.

**Methodology**: Analysis of apps including SuperWhisper, MacWhisper, Wispr Flow, Talon Voice, Windows Voice Access, Discord overlay, Gemini Live, Siri, Google Assistant's design language, and Windows 11 Fluent Design guidelines.

---

## What Existing Apps Do

### SuperWhisper (Mac, now Windows)

The reference implementation everyone copies. Key design decisions:

- **Main window**: Horizontal layout showing real-time waveform. Not an overlay - it's a proper window you invoke.
- **Mini window variant**: Small floating indicator always accessible. Shows waveform during recording, reveals stop button on hover.
- **Color-coded state dot**: Yellow = initializing, Blue = transcribing, Green = done.
- **Waveform visualization**: The waveform "visually represents the audio captured by your microphone" - it's the primary feedback during dictation, not a secondary element.
- **v2.0 (2025)**: Completely overhauled design. More compact, faster feel, Parakeet integration.

**Key lesson**: SuperWhisper treats the waveform as the hero element during recording. It's not a secondary "nice to have" - it's the primary signal that the app is working.

### MacWhisper (Mac only)

- **Global overlay**: Appears via keyboard shortcut. Stays visible during recording.
- **Error states**: Shows errors directly in the overlay (e.g., "blank dictation detected").
- **Dismissal**: Light-dismiss after transcription completes.
- Simpler than SuperWhisper - pill indicator with microphone icon and state text.

### Wispr Flow / WisprFlow

- **Floating bubble UI on Android**: A persistent floating button that can be summoned across apps.
- **Mac**: More integrated with text fields - appears near the cursor.
- Design is minimal - optimized for "out of the way until you need it."

### Talon Voice

- **Talon HUD**: Community-built overlay that adds a proper status bar. Shows current mode (sleep/command/dictation), language indicator, focus indicator.
- **Status bar**: Small, positioned at screen edge or top of active window.
- **Orange-red focus box**: Appears on top-center of focused window as mode indicator.
- New in 2025: Memory-safe renderer for better stability.

### Windows Voice Access (Microsoft)

- **Persistent bar at top of primary screen**: Always visible when Voice Access is active. Not collapsible.
- **Live transcription display**: Shows what you're saying in real time.
- **Processing indicator**: Center of bar shows command processing.
- Tasked for accessibility - heavy, not subtle.

### Discord Game Overlay

Excellent reference for "small, non-distracting, clearly communicates state":

- **Avatar grid**: Shows who's in voice channel, collapses to nothing when no one's speaking.
- **Green border glow on active speaker**: Simple, immediately readable.
- **Strong color for active states**: Red glow = mic muted, green glow = camera active.
- **2025 redesign**: Individual widgets instead of full interface. Action bar with just the controls you need.

### Google Gemini Live (2026 redesign)

The state of the art as of early 2026:

- **Floating pill UI**: Minimizes to a small pill that floats over other apps.
- **Waveform background**: The pill's background shows a live waveform animation of the Gemini Live visualization.
- **Text transcription** appears above the pill controls.
- **Glow animation**: Color glow around the panel signals AI is active.
- **Controls in pill**: Mic mute, end session, screen share - accessible without expanding.
- **Expand on demand**: Tap to get full interface.

This is where the category is heading: a pill that's always present but self-contained, with a live waveform in the background rather than as a separate element.

### Siri / Google Assistant Visual Language

Both have evolved from simple button interactions to immersive ambient indicators:

- **Siri waveform**: Four-color fluid waveform (red, orange, pink, purple). Responds directly to voice amplitude. Became iconic.
- **iOS 17 Dynamic Island**: Microphone activity shows as an orange dot privacy indicator. Recording state morphs the island shape.
- **Dynamic Island states**: Compact pill (background), medium banner (quick controls), large card (full expansion). Transitions: 0.3-0.5 seconds.
- **Google Assistant glow (2019 onward)**: Screen edge glow animation indicates listening. Project Astra (2025): Persistent blue glow around screen perimeter when AI is controlling.
- **Color language**: Google uses a four-color gradient (blue, green, red, yellow) as the brand signature for active AI.

---

## Key Design Findings

### Finding 1: The Pill Collapse Pattern Is Now Universal

**Evidence**: Every major voice app in 2025-2026 uses some variant of:
- Expanded state (pill with content) during active use
- Collapsed state (dot or small pill) when idle
- Smooth animated transition between the two
- Expand on hover or on activity

**Implication**: Cognitive Flow's current collapse-to-dot behavior is correct. The question is execution quality:
- Transition speed (250ms) is good
- The collapsed width (44px) is right
- The expand-on-hover is correct behavior
- The 3-second collapse timer might be too aggressive - consider 5 seconds

### Finding 2: Waveform Beats Dot for Recording State

**Evidence**: SuperWhisper, Gemini Live, Wispr Flow, MacWhisper - they all show a waveform during recording, not a static colored dot. Research shows animated waveforms:
- Are perceived as higher quality
- Make the interface feel "alive and human"
- More clearly signal that audio is being actively captured
- Give users confidence the app is hearing them

**Implication**: The current bottom-bar level indicator is easy to miss and low-information. During recording, the level feedback should be the most prominent visual element.

**The gap in Cognitive Flow**: A thin 3px bar at the bottom of a 40px widget is nearly invisible. Users can't tell if the app is hearing them.

### Finding 3: Color Conventions Are Load-Bearing

**Evidence**:
- Red = recording: Universal, rooted in broadcast history ("on-air" light, VCR record). SuperWhisper deviates (uses blue for transcription) but maintains red's urgency for errors.
- Amber/Orange = working/processing: Google Assistant, SuperWhisper (yellow for initializing), Discord (amber for intermediate states).
- Green = success/ready: Broadcast studios ("it's a wrap"), Discord (speaking indicator), most apps.
- Blue = loading/AI thinking: SuperWhisper (blue for transcription), Gemini (blue glow for AI control), iOS (blue microphone = listening in accessibility tools).

**Implication**: Cognitive Flow's color palette is well-chosen:
- `idle`: Emerald green (10, 185, 129) - correct
- `recording`: Red (239, 68, 68) - universally correct
- `processing`: Amber (245, 158, 11) - correct

The colors aren't the problem. The problem is they're only expressed through an 8-10px dot.

### Finding 4: Glow = Active, No Glow = Passive

**Evidence**: Discord's green glow on speaking, Gemini's panel glow, Siri's screen-edge glow. The pattern is consistent: when something is "hot" (actively doing work), it glows. When passive, it's flat.

**Current Cognitive Flow implementation**: Has a radial glow on the dot during recording/processing (radius 11px, alpha 50). This is good but:
- The glow is subtle because the background is `rgba(16, 16, 18, 165)` which is very dark
- The glow bleeds into the background rather than contrasting against it

**Implication**: The glow needs to be more visible. Options: larger glow radius, higher alpha, or a contrasting ring/border that lights up instead.

### Finding 5: Audio Level Feedback Needs to Be Obvious

**Evidence**: SuperWhisper's waveform is the primary recording indicator. Discord's avatar border lights up clearly when someone speaks. VU meters in professional audio - clearly visible feedback.

**Current Cognitive Flow gap**: The 3px bottom bar (`bar_height = 3`) is too thin to be glanceable. At 24" monitor distance, 3px is nearly invisible. The level bar also only appears during `recording` state and only when `_audio_level > 0`.

**What users need**: "Is this thing hearing me?" needs to be answerable with a glance, not close inspection.

### Finding 6: Processing State Needs Its Own Visual Language

**Evidence**: SuperWhisper uses yellow for initializing, blue for transcription. Discord uses a spinner. Windows Voice Access shows a processing indicator in the center of the bar. The pattern: processing is distinct from recording.

**Current Cognitive Flow**: Processing just changes the dot to amber. No motion. Users don't know if the app is working or frozen.

**What works**: A spinner overlay on the dot, a pulsing animation, or a progress-style sweep animation - anything that conveys "computing is happening."

---

## Deep Analysis

### Overlay Positioning: Where to Live

**Bottom-right corner is the dominant convention** for floating overlays that don't steal focus:
- Discord overlay: Configurable but defaults to bottom-right
- Windows system tray: Bottom-right
- Cognitive Flow: Bottom-right with 24px margin

**Why bottom-right works**:
- Reading flow is top-left to bottom-right, so the bottom-right is "out of the way" of content
- System tray trains users to look there for status indicators
- Away from modal/notification areas (top-right on Windows)

**Alternative: Top-center** (Windows Voice Access, some Talon HUD configs): Works for accessibility-focused apps where the indicator is the primary interface. Too intrusive for background-mode apps like Cognitive Flow.

**Sizing conventions**:
- Minimum useful size: ~32px height (readable text + dot)
- Comfortable: 36-44px height
- Maximum without being a window: ~200px wide
- Cognitive Flow: 40px height, 180px expanded, 44px collapsed - this is right

**Corner radius**: Windows 11 uses rounded corners heavily. 8-12px corner radius is appropriate for a floating pill. Cognitive Flow uses 8px - correct.

### Transparency and Background Materials

**Windows 11 Fluent Design**: Two main surface types:
- **Mica**: Semi-transparent, tints to desktop wallpaper color. For persistent surfaces.
- **Acrylic**: Background blur + noise. For transient/popup surfaces.

**For a voice indicator overlay**, Acrylic is appropriate - it's a transient surface. But real Acrylic (Windows API `DwmSetWindowAttribute` with `DWMWA_SYSTEMBACKDROP_TYPE`) requires WinUI or careful Win32 calls. PyQt6 can approximate with semi-transparent dark background.

**What works without real Acrylic**:
- Dark semi-transparent background (`rgba(16, 16, 18, 165)`) reads fine over most content
- 1px white border at 20-40 alpha provides depth separation
- Drop shadow provides elevation sense

**Cognitive Flow's current implementation**: `bg_primary: rgba(16, 16, 18, 165)` is good. The `border_subtle: rgba(255, 255, 255, 20)` is appropriate but could go to 30 alpha for slightly more definition.

**What NOT to do with transparency**:
- Too transparent (< 100 alpha) makes text unreadable over light backgrounds
- Frosted glass without actual blur just looks muddy
- Background that matches the wallpaper too closely makes the widget invisible

### Animation System

**Research consensus on durations**:
- State transitions: 200-300ms (Cognitive Flow: 250-300ms - correct)
- Hover: 150-200ms (Cognitive Flow: 200ms - correct)
- Collapse/expand: 200-300ms (Cognitive Flow: 250ms - correct)
- Entry/exit: 300-500ms (Cognitive Flow: 400ms - fine)

**Easing recommendations**:
- Entrances: `ease-out` (starts fast, slows down - feels responsive)
- Exits: `ease-in` (speeds up then gone - natural disappearance)
- State changes: `ease-in-out` or `OutCubic`
- Never use `linear` - looks robotic

**Cognitive Flow**: Uses `OutCubic` throughout. This is correct for entrances. For exits (collapse), `InCubic` would be slightly more natural but `OutCubic` is acceptable.

**Spring animations**: Physics-based springs (stiffness/damping/mass) feel more natural than easing curves but are harder to implement in PyQt6's `QPropertyAnimation`. The current curve-based approach is fine.

**Pulsing animations**: For a "recording" state, a continuous pulse (scale or opacity) on the dot communicates "actively capturing" better than a static colored circle. Implementation: `QPropertyAnimation` cycling between opacity 0.6 and 1.0, duration 800-1200ms, `SineCurve` easing.

### The Level Meter Problem

**What the research shows**:

The three options, ranked by effectiveness:

1. **Animated waveform bars** (multiple bars at different heights responding to FFT bins): Most informative, most visually appealing, highest implementation cost. SuperWhisper, Gemini Live.

2. **Single animated bar that grows/shrinks** (current Cognitive Flow bottom bar): Low information density, easy to miss, but low implementation cost. Adequate but underwhelming.

3. **Pulsing dot that responds to amplitude** (dot size or glow radius scales with audio level): Great compromise. Easy to implement via `QPropertyAnimation` driven by `set_audio_level()`. Immediately visible because it uses the already-prominent dot.

**Recommendation**: The dot-glow-to-audio-level approach is the best fit for Cognitive Flow's architecture. Tie the glow radius directly to the audio level rather than (or in addition to) the bottom bar.

**Waveform bars if you want to go there**: 5-7 bars, each independently responding to RMS energy. Can be approximated without FFT by using the overall RMS with slight random variation per bar. Height: 16-24px. Width: 3-4px per bar with 2px gap. Color matches state dot color.

### What Modern Voice Apps Get Right That Most Don't

**1. Hotkey confirmation feedback**: The moment the hotkey is pressed, there should be immediate visual feedback - before recording starts, before audio is captured. The indicator should respond within one frame of the keypress. Cognitive Flow starts recording immediately but the UI update goes through threading - there may be a perceptible lag.

**2. Silence detection visual**: Some apps (notably Whisper Desktop) show three states: voice activity detected, transcribing, stalled. Showing "silence" vs "hearing voice" within the recording state is useful feedback.

**3. Error state clarity**: MacWhisper shows errors directly in the overlay. Cognitive Flow needs to handle this - what does the user see if transcription fails?

**4. Clipboard mode differentiation**: Cognitive Flow has two modes (type vs clipboard). The overlay click triggers clipboard mode. There's no visual distinction between the two modes in the current indicator.

---

## Synthesis and Recommendations

### What's Already Good in Cognitive Flow

- Bottom-right corner positioning: Correct.
- Collapse-to-dot with hover-expand: Correct behavior.
- Color palette (emerald idle, red recording, amber processing): All correct.
- Dark semi-transparent background: Good.
- 8px corner radius: Appropriate.
- Font choice (Segoe UI 9 DemiBold): Correct for Windows.
- Drop shadow: Good for elevation.
- Drag to reposition: Good UX.
- Right-click context menu: Good UX.

### Specific Things to Improve

**Priority 1 - Pulsing animation during recording**: Add a continuous pulsing animation to the dot when in recording state. The dot should breathe (opacity oscillates 0.7 to 1.0, or radius oscillates slightly). This is the single most impactful change. Implementation: drive a looping `QPropertyAnimation` on `circle_color` or a new `_dot_pulse` property when state == "recording".

**Priority 2 - Level-responsive glow**: Instead of (or in addition to) the bottom bar, scale the glow radius with audio level. Currently glow_radius is fixed at `dot_radius + 6`. Scale it: `glow_radius = dot_radius + 4 + (level * 12)`. This makes "is it hearing me" immediately visible.

**Priority 3 - Processing animation**: Amber dot + amber text isn't enough. Add a spinner or sweep animation to the dot during processing. Options:
- Rotating arc drawn in `paintEvent` around the dot
- Pulsing at faster frequency than recording (400ms vs 1000ms)
- Brief opacity flash on state transition

**Priority 4 - Waveform bars** (optional, significant effort): Replace or augment the bottom bar with 5-7 waveform bars next to the dot during recording. These respond to audio level with slight per-bar variation. This moves Cognitive Flow from "functional" to "polished."

**Priority 5 - Mode distinction**: When in clipboard mode (triggered by clicking the indicator), show a different icon or tint the dot differently (e.g., purple/violet instead of red). Small but meaningful.

### Color Refinements

The palette is solid. Consider:
- `idle` dot: The current emerald is a bit bright green. A slightly cooler teal (`rgb(6, 182, 212)`) reads more "technology" and less "success/money."
- Border alpha: Increase from 20 to 28-32 for slightly better edge definition.
- Glow alpha: Increase from 50 to 70-80 for more visible glow.

### Size and Position Refinements

- Height: 40px is fine. Could go to 36px for slightly more minimal feel.
- Collapsed width: 44px is right. Don't go smaller than 36px.
- Margin from edge: 24px is good. Bottom margin might benefit from being slightly larger (28-32px) to clear the taskbar on high-DPI displays.
- Dot radius: Current 5px is fine. During recording, don't scale the dot itself (confusing) - use glow instead.

### What NOT to Do

**1. Don't animate the position/size of the whole widget based on audio level.** Only the internal visual (dot glow, waveform bars) should respond to audio. The widget bounds should stay stable.

**2. Don't use red for anything other than recording.** Red means danger in Windows UI. Using it for error states (which Cognitive Flow does with `error: rgb(220, 38, 38)`) is fine, but don't accidentally show it as idle.

**3. Don't make the overlay clickthrough when recording/processing.** During active states, the user may want to interact with it (cancel, clipboard mode). Clickthrough should only be an option when truly idle.

**4. Don't collapse during recording or processing.** Cognitive Flow correctly prevents this. Never animate away from the user during active states.

**5. Don't use opacity below 0.6 for the background.** At lower opacity, text becomes unreadable over bright backgrounds (common with white browser windows). Current `165/255 = 0.65` opacity is near the minimum.

**6. Don't use more than 3 colors in the state system.** Adding a fourth state color (e.g., purple for clipboard mode) should be done carefully. More than 4 unique indicator states creates cognitive overhead.

**7. Don't put important feedback at the edge of the widget** (like the current 3px bottom bar). Users look at the center of the indicator (the dot) for state information.

**8. Don't skip the hover state expansion.** It's a small detail but critical for discoverability - collapsed dot expands to show what it is when you approach it.

**9. Don't show tooltips for the main recording workflow.** Tooltips are for secondary actions (context menu items). The main state should be self-evident from the visual.

**10. Don't fight Windows z-order without `WindowStaysOnTopHint`.** Cognitive Flow already has this. The 30-second heartbeat to recover from compositor hiding is good practice.

---

## Decision Framework: Waveform vs Level Bar vs Pulsing Dot

```
Do you have time to implement FFT/frequency analysis?
  YES -> Use multi-bar waveform (5-7 bars, respond to frequency bins)
  NO  ->
    Is audio-level feedback the primary concern?
      YES -> Pulsing dot + level-responsive glow (best single-element solution)
      NO  -> Static colored dot (adequate, but feels unpolished)
    Do you want a visual bar at all?
      YES -> Keep bottom bar but increase height to 6px and make it respond faster
      NO  -> Remove it, invest in dot glow instead
```

**For Cognitive Flow**: Pulsing dot + level-responsive glow during recording. Add waveform bars if/when you want to invest in it.

---

## Practical Implementation Guide

### Pulsing Dot During Recording

In `__init__` of `FloatingIndicator`, add:
```python
self._pulse_opacity = 1.0
self._pulse_animation = QPropertyAnimation(self, b"pulse_opacity")
self._pulse_animation.setDuration(900)
self._pulse_animation.setStartValue(0.65)
self._pulse_animation.setEndValue(1.0)
self._pulse_animation.setEasingCurve(QEasingCurve.Type.SineCurve)
self._pulse_animation.setLoopCount(-1)  # infinite
```

In `set_state()`:
```python
if state == "recording":
    self._pulse_animation.start()
else:
    self._pulse_animation.stop()
    self._pulse_opacity = 1.0
```

In `paintEvent()`, multiply the dot alpha by `self._pulse_opacity`.

### Level-Responsive Glow

Replace the fixed `glow_radius = dot_radius + 6` with:
```python
glow_radius = dot_radius + 4 + int(self._audio_level * 14)
glow_alpha = 40 + int(self._audio_level * 40)  # 40-80 alpha
```

### Processing Spinner Arc

In `paintEvent()`, when `self.state == "processing"`:
```python
# Draw rotating arc around the dot
arc_pen = QPen(self._circle_color)
arc_pen.setWidth(2)
painter.setPen(arc_pen)
painter.setBrush(Qt.BrushStyle.NoBrush)
painter.drawArc(
    int(dot_x - dot_radius - 4),
    int(dot_y - dot_radius - 4),
    (dot_radius + 4) * 2,
    (dot_radius + 4) * 2,
    self._spin_angle * 16,  # Qt uses 1/16 degree units
    120 * 16  # 120 degree arc
)
```

Drive `_spin_angle` with a `QTimer` incrementing by 6 degrees per 16ms (~60fps), reset to 0 at 360.

---

## Common Pitfalls to Avoid

1. **Invisible indicator**: Test against white backgrounds (Word, browser). Dark-only testing misses this.
2. **Lag between hotkey and visual response**: Profile the threading path. If `set_state("recording")` is called 100ms after keypress, users notice.
3. **Animation timer drift on sleep/wake**: Cognitive Flow's 30s heartbeat handles visibility recovery but spinning animations need to be reset on wake too.
4. **DPI scaling bugs**: Test at 125%, 150%, 175% scaling. The 24px margin may put the indicator off-screen on some DPI + taskbar configurations.
5. **Taskbar overlap**: On Windows 11, the taskbar is 48px by default. The 24px bottom margin + 40px height = 64px from bottom edge, which is fine, but verify on a live system.
6. **Right-click context menu appearing behind the indicator**: Qt menus can sometimes render behind a `WindowStaysOnTopHint` window's own context. Test on actual Windows.

---

## Further Research Opportunities

- **Accessibility audit**: Screen reader behavior with the indicator. `Tool` window flag may make it invisible to accessibility tools.
- **Multi-monitor behavior**: Cognitive Flow reads `primaryScreen()` but voice work often happens on the secondary monitor.
- **Dark/light mode switching**: The dark-only palette is fine for most use but might need a light-mode fallback.
- **User testing**: What collapse delay feels right? 3 seconds may be too fast. 5-6 seconds is more comfortable based on how long it takes to look at what was transcribed.

---

## References

1. [SuperWhisper Recording Window Documentation](https://superwhisper.com/docs/get-started/interface-rec-window)
2. [SuperWhisper v2.0 Announcement](https://alternativeto.net/news/2025/7/superwhisper-v2-0-presents-new-design-faster-parakeet-model-and-lower-latency/)
3. [Talon HUD GitHub](https://github.com/chaosparrot/talon_hud)
4. [Discord Game Overlay 101](https://support.discord.com/hc/en-us/articles/217659737-Game-Overlay-101)
5. [Discord Player Q1 2025 Release](https://discord.com/blog/player-release-q12025)
6. [Google Gemini Glow Floating Overlay](https://9to5google.com/2024/08/23/gemini-glow-floating-overlay/)
7. [Google Gemini Live Floating Pill UI](https://www.androidheadlines.com/2026/02/google-gemini-live-floating-pill-redesign-android.html)
8. [Windows Voice Access Getting Started](https://support.microsoft.com/en-us/topic/get-started-with-voice-access-bd2aa2dc-46c2-486c-93ae-3d75f7d053a4)
9. [Acrylic Material - Windows Apps](https://learn.microsoft.com/en-us/windows/apps/design/style/acrylic)
10. [Mica Material - Windows Apps](https://learn.microsoft.com/en-us/windows/apps/design/style/mica)
11. [Widget Design Fundamentals - Windows Apps](https://learn.microsoft.com/en-us/windows/apps/design/widgets/widgets-design-fundamentals)
12. [MacWhisper Global Feature](https://macwhisper.helpscoutdocs.com/article/16-global)
13. [Wispr Flow Features](https://wisprflow.ai/features)
14. [Glassmorphism UI Design Trend](https://designsbydaveo.com/what-is-glassmorphism-ui-design-trend-for-2024/)
15. [Voice User Interface Design Best Practices 2025](https://lollypop.design/blog/2025/august/voice-user-interface-design-best-practices/)
16. [Why is the Record Icon Always Red - UX Stack Exchange](https://stackprinter.com/export?linktohome=true&printer=false&question=41434&service=ux.stackexchange)
17. [NN/G Animation Duration](https://www.nngroup.com/articles/animation-duration/)
18. [ElevenLabs Live Waveform Component](https://ui.elevenlabs.io/docs/components/live-waveform)
19. [Apple Dynamic Island Wave UX Lesson](https://medium.com/@shubhamdeepgupta/apples-dynamic-island-wave-a-small-detail-with-a-big-ux-design-lesson-665fefa13025)
20. [Siri Wave JS Library](https://github.com/kopiro/siriwave)
21. [Voice User Interface Design Guide 2026](https://fuselabcreative.com/voice-user-interface-design-guide-2026/)
22. [Nerd Dictation - No Visual Feedback Analysis](https://news.ycombinator.com/item?id=29972579)
23. [Color Psychology in UI UX 2025](https://mockflow.com/blog/color-psychology-in-ui-design)
24. [Microsoft Fluent Design System](https://fluent2.microsoft.design/material)
25. [Voxtype Status Indicator Design](https://voxtype.io/)
26. [winWhisper App Design](https://www.winwhisper.app/)
27. [Wispr Flow Floating Bubble UI Android](https://hothardware.com/news/wispr-flow-ai-dictation-app-android)
28. [Web Animation Best Practices](https://gist.github.com/uxderrick/07b81ca63932865ef1a7dc94fbe07838)
29. [NN/G Indicators vs Notifications](https://www.nngroup.com/articles/indicators-validations-notifications/)
30. [Smashing Magazine Notification Design Guidelines](https://www.smashingmagazine.com/2025/07/design-guidelines-better-notifications-ux/)

---

**Research Confidence Level**: High - multiple authoritative sources, convergent findings across competing products, grounded in analysis of the actual Cognitive Flow codebase.
