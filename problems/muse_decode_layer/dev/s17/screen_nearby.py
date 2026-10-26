"""All 65 review cases plus official cases on the exact S17 composition."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'s16'))
sp=importlib.util.spec_from_file_location('s17_screen',H/'screen.py');screen=importlib.util.module_from_spec(sp);sp.loader.exec_module(screen)
import screen_nearby as near
near.H=H
if __name__=='__main__':near.main()
