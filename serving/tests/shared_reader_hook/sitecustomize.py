"""Force only the Hopper bank binder; execute real reader arithmetic on sm120."""
import importlib.abc
import importlib.machinery
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
runpy.run_path(str(ROOT/'serving/harness/pyhook/sitecustomize.py'))
TARGET='sglang.srt.layers.quantization.muse_spark_nvfp4_moe'


class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return None
    def exec_module(self,module):
        self.original.exec_module(module)
        cls=module.MuseSparkNvFp4FusedMoEMethod
        original=cls._process_weights_for_marlin
        def prepare(self,layer):
            result=original(self,layer)
            assert layer.num_local_experts==32 and layer.hidden_size==layer.intermediate_size_per_partition==4096
            from sglang.srt.layers.muse_spark_expert_bank import prepare_shared_bank
            prepare_shared_bank(layer)
            print('SHARED_READER actual NVFP4 arithmetic; local_start',layer.local_expert_offset,flush=True)
            return result
        cls._process_weights_for_marlin=prepare


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname==TARGET:
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            spec.loader=Loader(spec.loader);return spec


sys.meta_path.insert(0,Finder())
