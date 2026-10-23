import sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("c", sys.argv[1]); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
t = c._Timer()
print("base", t.base)
c._probes(t)
c._deepgemm(t)
