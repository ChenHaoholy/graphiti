"""Compatibility wrapper for the simplified graph build command.

Prefer:
  python -m src.build
"""

import asyncio

try:
    from .build import main
except ImportError:
    from build import main


if __name__ == "__main__":
    asyncio.run(main())
