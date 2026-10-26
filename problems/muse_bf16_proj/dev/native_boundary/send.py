"""Use h200_lab's normal sender with this lab's checked 14k output reservation.

The judge retains16k; 256marker +2000suffix bytes remain beyond13744packet.
This process-local planning constant changes no candidate, measurement or shared tool.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'tools'))
import h200_lab
h200_lab.LAB_ONLY_LIMIT=14000
h200_lab.main()
