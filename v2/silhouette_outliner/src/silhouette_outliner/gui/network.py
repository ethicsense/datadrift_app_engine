from __future__ import annotations

import socket
from typing import Iterable

_DEFAULT_HOSTS: tuple[tuple[str, int], ...] = (
    ("www.musinsa.com", 443),
    ("1.1.1.1", 53),
)


def check_online(
    hosts: Iterable[tuple[str, int]] | None = None,
    timeout: float = 3.0,
) -> bool:
    """Return True if at least one host accepts a TCP connection."""
    targets = tuple(hosts) if hosts is not None else _DEFAULT_HOSTS
    for host, port in targets:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False
