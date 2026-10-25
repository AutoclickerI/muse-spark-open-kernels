"""Frozen S14 handoff: exact S13 final 958f1258 -> S14 final f1c77ce6."""
import hashlib
import importlib.util
from pathlib import Path

BASE_SHA = '958f1258908f44834ec6cf0c33af00ff6c08ea62b2f5cc483a36f0a91f39d0b6'
FINAL_SHA = 'f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'


def patch(source):
    assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA
    spec = importlib.util.spec_from_file_location('s14_builder', Path(__file__).with_name('build.py'))
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    result, _ = builder.compose(without=('qkv3', 'o'))
    assert hashlib.sha256(result.encode()).hexdigest() == FINAL_SHA
    return result
