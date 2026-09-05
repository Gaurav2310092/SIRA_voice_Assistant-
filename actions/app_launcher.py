"""Spawns local desktop apps without blocking SIRA's main voice loop."""

import subprocess

from config.settings import APP_PATHS


def open_app(app_name: str):
    if not app_name:
        return "I didn't catch which app to open."

    app_key = app_name.lower().strip()
    executable = APP_PATHS.get(app_key)

    if not executable:
        return f"I don't know how to open {app_name}."

    try:
        # Popen is non-blocking — SIRA keeps listening while the app launches
        subprocess.Popen(executable, shell=False)
        return f"Opening {app_name}."
    except FileNotFoundError:
        return f"Couldn't find {app_name} on this system."
    except Exception as e:
        return f"Failed to open {app_name}: {e}"
