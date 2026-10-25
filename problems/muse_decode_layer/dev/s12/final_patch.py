"""Frozen S12 handoff on exact S11 final (acda6c62), no wide banks."""
import hashlib
import importlib.util
from pathlib import Path

def patch(source):
    spec=importlib.util.spec_from_file_location('s12_final_builder',Path(__file__).with_name('build.py'))
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    assert hashlib.sha256(source.encode()).hexdigest()==b.BASE_SHA
    result,_=b.compose(['wide'])
    assert hashlib.sha256(result.encode()).hexdigest()=='240248b2c27c890bf5055a77e87d2fc9c4429b747a04d714e51d5f67e936bb3e'
    return result
