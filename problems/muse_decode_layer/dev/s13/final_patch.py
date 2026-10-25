"""Frozen S13 final on exact S12 final (240248b2): attention D1 + PERMW256, no cpall."""
import hashlib
import importlib.util
from pathlib import Path


def patch(source):
    spec=importlib.util.spec_from_file_location('s13_final_builder',Path(__file__).with_name('build.py'))
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    assert hashlib.sha256(source.encode()).hexdigest()==b.BASE_SHA
    result,_=b.compose(['cpall'])
    assert hashlib.sha256(result.encode()).hexdigest()=='958f1258908f44834ec6cf0c33af00ff6c08ea62b2f5cc483a36f0a91f39d0b6'
    return result
