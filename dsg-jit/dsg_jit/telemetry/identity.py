from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_install_id: Optional[str] = None
_session_id: Optional[str] = None


def _get_config_dir() -> Path:
    """Get the platform-appropriate config directory for DSG-JIT."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            base = Path(appdata)
        else:
            base = Path.home() / "AppData" / "Roaming"
    else:
        # Linux and others: XDG_CONFIG_HOME or ~/.config
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg:
            base = Path(xdg)
        else:
            base = Path.home() / ".config"
    return base / "dsgjit"


def _get_telemetry_file() -> Path:
    """Get the path to the telemetry identity file."""
    return _get_config_dir() / "telemetry.json"


def get_install_id() -> str:
    """Get or create the persistent install ID.

    The install ID is a UUIDv4 that uniquely identifies this installation.
    It is stored persistently and reused across sessions.

    :return: The install ID as a string.
    """
    global _install_id
    if _install_id is not None:
        return _install_id

    telemetry_file = _get_telemetry_file()

    # Try to load existing install ID
    if telemetry_file.exists():
        try:
            with open(telemetry_file, "r") as f:
                data = json.load(f)
                _install_id = data.get("install_id")
                if _install_id:
                    return _install_id
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    # Generate new install ID
    _install_id = str(uuid.uuid4())

    # Persist it
    try:
        telemetry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "install_id": _install_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "telemetry_level": "standard",
        }
        with open(telemetry_file, "w") as f:
            json.dump(data, f, indent=2)
    except (IOError, OSError):
        # Fail silently - telemetry should never break user code
        pass

    return _install_id


def get_session_id() -> str:
    """Get or create the per-process session ID.

    The session ID is a UUIDv4 generated once per Python process.
    It is not persisted.

    :return: The session ID as a string.
    """
    global _session_id
    if _session_id is None:
        _session_id = str(uuid.uuid4())
    return _session_id


def reset_session_id() -> None:
    """Reset the session ID (mainly for testing)."""
    global _session_id
    _session_id = None
