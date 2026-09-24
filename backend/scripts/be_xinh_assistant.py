"""Stable entrypoint for Bé Xinh voice chat.

The implementation remains in ``halinh_assistant.py`` for backward
compatibility with older launchers and deployments.
"""

from __future__ import annotations

from halinh_assistant import main


if __name__ == "__main__":
    main()
