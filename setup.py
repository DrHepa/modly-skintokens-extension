from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skintokens_ext.setup_runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
