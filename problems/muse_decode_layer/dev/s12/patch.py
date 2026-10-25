"""Aggregate repository handoff: patch(exact S11 final source) -> current pinned S12 source."""
import hashlib
import importlib.util
from pathlib import Path


def patch(source):
    spec = importlib.util.spec_from_file_location('s12_build', Path(__file__).with_name('build.py'))
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert hashlib.sha256(source.encode()).hexdigest() == builder.BASE_SHA
    result, _ = builder.compose()
    return result
