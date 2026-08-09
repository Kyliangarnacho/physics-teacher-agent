"""Pure helpers for stable in-page conversation navigation."""


def message_anchor(message_id: object) -> str:
    """Return an injection-safe, stable DOM anchor for one stored message."""
    safe_id = "".join(
        char
        for char in str(message_id)
        if char.isalnum() or char in "-_"
    )
    return f"message-{safe_id}" if safe_id else ""
