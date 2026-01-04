# Cognitive Flow - Project Instructions

## Versioning Rules

**ALWAYS update version and changelog for every feature/fix:**

1. Bump version in `cognitive_flow/__init__.py`
2. Update changelog in `cognitive_flow/app.py` (in the `main()` function banner)
3. Commit message should reference the version

**Version format:** `MAJOR.MINOR.PATCH`
- PATCH: Bug fixes, small tweaks
- MINOR: New features, settings, UI changes
- MAJOR: Breaking changes, major rewrites

## Project Structure

```
cognitive_flow/
    __init__.py      # Version number here
    __main__.py      # python -m cognitive_flow
    app.py           # Main app, changelog in main()
    ui.py            # PyQt6 UI, settings dialog
    logger.py        # Logging with timing support
    paths.py         # App data paths
```

## Key Files

- **Config:** `%APPDATA%\CognitiveFlow\config.json`
- **Stats:** `%APPDATA%\CognitiveFlow\statistics.json`
- **Debug log:** `%APPDATA%\CognitiveFlow\debug_transcriptions.log` (only in --debug mode)

## Running

```cmd
cognitive-flow          # Background mode (detached)
cognitive-flow --debug  # Foreground with verbose output + file logging
```

## Current Config Options

- `model_name`: Whisper model (tiny/base/small/medium/large)
- `add_trailing_space`: Add space after each transcription
- `input_device_index`: PyAudio device index (null = system default)

## Git Workflow

- Create feature branch for changes
- Merge to main when complete
- Push after merge
