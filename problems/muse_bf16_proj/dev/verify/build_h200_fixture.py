"""Build the local-only whole-layer test for every measured H200 Lt recipe."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def build():
    spec=importlib.util.spec_from_file_location('fixture',HERE/'build_fixture.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    original=builder.build().read_text().split('# Local integration fixture:',1)[0]
    path=ROOT/'runs/screen/verify_proj/integration_h200_fixture.py'
    path.write_text(original+(HERE/'h200_fixture_tail.py').read_text())
    return path


if __name__=='__main__':print(build())
