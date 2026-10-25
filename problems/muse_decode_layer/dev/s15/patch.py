"""Experimental S15 handoff on exact S14; promotion awaits paired reset receipt."""
import hashlib
import importlib.util
from pathlib import Path

BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
CANDIDATE_SHA='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'


def patch(source):
    assert hashlib.sha256(source.encode()).hexdigest()==BASE_SHA
    spec=importlib.util.spec_from_file_location('s15_builder',Path(__file__).with_name('build.py'))
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    result,_=builder.compose()
    assert hashlib.sha256(result.encode()).hexdigest()==CANDIDATE_SHA
    return result
