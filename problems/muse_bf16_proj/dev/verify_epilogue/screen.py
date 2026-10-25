"""local_eval with an equivalent, memory-efficient cold READ of its full 1 GiB.

uint8.sum materializes an 8 GiB int64 temporary. Reading the same buffer
through an int64 view preserves the cold-cache step and avoids that temporary.
All tests, fixtures, graph timing, reference checks and static checks remain
tools/local_eval.py's. Optional --forced installs the standard Hopper stand-ins.
"""
import importlib.util
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'tools'))
import local_eval


def replay_us(self,graph):
    self.flush_buf.view(local_eval.torch.int64).sum()
    self.start.record();graph.replay();self.end.record();self.end.synchronize()
    return self.start.elapsed_time(self.end)*1000.


def main():
    local_eval.Timer.replay_us=replay_us
    forced='--forced' in sys.argv
    if forced:
        sys.argv.remove('--forced')
        spec=importlib.util.spec_from_file_location('p7_force',REPO/'problems/muse_decode_layer/dev/best7/force.py')
        force=importlib.util.module_from_spec(spec);spec.loader.exec_module(force)
        original=local_eval.load_module
        def loader(name,path):
            m=original(name,path)
            if name=='submission':force.install(m,False,False)
            return m
        local_eval.load_module=loader
    local_eval.main()


if __name__=='__main__':main()
