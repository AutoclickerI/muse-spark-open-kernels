"""Asserted aggregate S13 handoff onto S12 final."""
import hashlib
import importlib.util
from pathlib import Path

def patch(source):
    spec=importlib.util.spec_from_file_location('s13_builder',Path(__file__).with_name('build.py'))
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    assert hashlib.sha256(source.encode()).hexdigest()==b.BASE_SHA
    return b.compose()[0]
