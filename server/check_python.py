"""Is this interpreter able to run NetSim3D?

Exit codes
  0  yes, a tested version
  1  free-threaded build: reject. No PyTorch wheels exist for cp3xxt, and
     most packages fall back to compiling from source on Windows
  2  too old
  3  newer than what has been tested: usable, but wheels may not exist yet

  python check_python.py --report   also prints a one-line description
"""

import sys
import sysconfig

TESTED_MIN = (3, 9)
TESTED_MAX = (3, 14)

free = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
ver = sys.version_info[:2]

if free:
    code = 1
elif ver < TESTED_MIN:
    code = 2
elif ver > TESTED_MAX:
    code = 3
else:
    code = 0

if "--report" in sys.argv:
    tag = {0: "supported", 1: "FREE-THREADED, not usable",
           2: "too old", 3: "newer than tested"}[code]
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}."
          f"{sys.version_info.micro} ({tag})")
    print(f"  at {sys.executable}")

raise SystemExit(code)
