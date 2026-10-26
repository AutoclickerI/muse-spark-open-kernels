"""Normal h200_lab sender with a P13-local 14k receipt reservation.

13744 packet bytes + 256 marker bytes + 2000 error-suffix bytes fit the judge's
16k tail. No shared tool, timing, sample, or kernel change.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'tools'))
import h200_lab
h200_lab.LAB_ONLY_LIMIT=14000
h200_lab.main()
