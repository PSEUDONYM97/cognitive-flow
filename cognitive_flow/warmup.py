"""
Cognitive Flow Warmup - Pre-loads ctranslate2 DLLs at Windows login.

This runs silently at startup to cache the DLLs in memory,
so Cognitive Flow starts fast when you actually need it.

Run manually: pythonw -m cognitive_flow.warmup
Install task: python -m cognitive_flow.warmup --install
Remove task:  python -m cognitive_flow.warmup --uninstall
"""

import sys
import subprocess


def warmup():
    """Import ctranslate2 to trigger DLL caching."""
    try:
        import ctranslate2  # noqa: F401 - This is the slow import we're warming
    except ImportError:
        pass  # Not installed, nothing to warm


def install_startup_task():
    """Install Cognitive Flow to run at Windows login."""
    import os
    from pathlib import Path
    
    # Get Startup folder
    startup_dir = Path(os.environ.get("APPDATA")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    shortcut_path = startup_dir / "CognitiveFlow.bat"
    
    pythonw = sys.executable.replace("python.exe", "pythonw.exe")
    if not os.path.exists(pythonw):
        pythonw = sys.executable
    
    # Run the full app in background mode at startup
    bat_content = f'@echo off\nstart "" /b "{pythonw}" -m cognitive_flow\n'
    
    try:
        shortcut_path.write_text(bat_content)
        print(f"Startup script installed: {shortcut_path}")
        print("Cognitive Flow will start automatically at Windows login")
        return True
    except Exception as e:
        print(f"Failed to install: {e}")
        return False


def uninstall_startup_task():
    """Remove the startup script."""
    import os
    from pathlib import Path
    
    startup_dir = Path(os.environ.get("APPDATA")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    shortcut_path = startup_dir / "CognitiveFlow.bat"
    
    try:
        if shortcut_path.exists():
            shortcut_path.unlink()
            print(f"Startup script removed: {shortcut_path}")
        else:
            print("Startup script not found")
        return True
    except Exception as e:
        print(f"Failed to remove: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cognitive Flow Warmup")
    parser.add_argument("--install", action="store_true", help="Install startup task")
    parser.add_argument("--uninstall", action="store_true", help="Remove startup task")
    args = parser.parse_args()
    
    if args.install:
        install_startup_task()
    elif args.uninstall:
        uninstall_startup_task()
    else:
        # Default: just do the warmup silently
        warmup()


if __name__ == "__main__":
    main()
