"""S21 layer screens on GPU0 with the explicit S20 Hopper stand-ins."""
import importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('s21_prior_screen',H.parent/'s20/screen.py');prior=importlib.util.module_from_spec(sp);sp.loader.exec_module(prior)
install=prior.install
if __name__=='__main__':prior.s19.old.install=install;prior.s19.old.main()
