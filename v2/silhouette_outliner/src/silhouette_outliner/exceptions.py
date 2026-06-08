from __future__ import annotations


class CollectCancelled(Exception):
    """Raised when a collect run is cancelled cooperatively."""
