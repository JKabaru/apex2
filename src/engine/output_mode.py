_current_mode = "focus"


def set_mode(mode: str) -> None:
    global _current_mode
    m = mode.lower()
    if m in ("focus", "verbose"):
        _current_mode = m


def get_mode() -> str:
    return _current_mode


def is_verbose() -> bool:
    return _current_mode == "verbose"


def is_focus() -> bool:
    return _current_mode == "focus"
