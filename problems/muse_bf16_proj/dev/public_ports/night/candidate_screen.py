"""Official normal/forced screen of the exact parent+new carrier portfolio patch."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('ng_candidate_standins',H/'local_standins.py');local=importlib.util.module_from_spec(sp);sys.modules[sp.name]=local;sp.loader.exec_module(local)
def install(m,forced=False):local.install(m,forced);return []
local.prior.s19.prior.s19.old.install=install;local.prior.s19.prior.s19.old.main()
