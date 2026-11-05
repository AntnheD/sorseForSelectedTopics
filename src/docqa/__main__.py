"""Allow ``python -m docqa ...``."""

import sys

from .cli import main

sys.exit(main())
