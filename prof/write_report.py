"""Generate a current H200 profile report directly from saved receipts and runtime profiles."""
import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import profile_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write Markdown here; otherwise print it")
    args = parser.parse_args()
    data = [profile_report.build(problem) for problem in profile_report.PROBLEMS]
    output = StringIO()
    with redirect_stdout(output):
        profile_report.markdown(data)
    if args.out:
        args.out.write_text(output.getvalue())
    else:
        print(output.getvalue(), end="")


if __name__ == "__main__":
    main()
