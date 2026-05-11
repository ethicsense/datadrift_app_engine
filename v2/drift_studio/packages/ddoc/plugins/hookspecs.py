"""
Compatibility shim for editable installs.

This module makes `ddoc.plugins.hookspecs` importable when `ddoc` is resolved
as a namespace package rooted at `packages/ddoc`.
"""

from ddoc.ddoc.plugins.hookspecs import *  # noqa: F401,F403

