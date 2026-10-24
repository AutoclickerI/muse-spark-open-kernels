"""Regenerate the complete source-only patch and retain reproducible checks."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TREE = ROOT / 'serving/sglang'
BASE = 'a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f'


def git(*args):
    return subprocess.check_output(['git', '-C', str(TREE), *args])


def sections(data):
    return {part.split(b'\n', 1)[0].split(b' b/', 1)[1].decode():
            b'diff --git ' + part for part in data.split(b'diff --git ')[1:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--tested-tree', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    args = parser.parse_args()
    old = sections(args.previous.read_bytes())
    untracked = git('ls-files', '--others', '--exclude-standard', '--', 'python/sglang').decode().splitlines()
    for path in untracked:
        assert Path(path).suffix in {'.py', '.cu', '.cuh', '.h', '.hpp', '.c', '.cc', '.cpp', '.json'}, path
    if untracked:
        subprocess.run(['git', '-C', str(TREE), 'add', '-N', '--', *untracked], check=True)
    data = git('diff', '--no-ext-diff', '--no-renames', BASE, '--', 'python/sglang')
    new = sections(data)
    assert not old.keys() - new.keys(), ('lost accepted paths', old.keys() - new.keys())
    comment_only_paths = []
    for path in new:
        live, tested = (TREE / path).read_bytes(), (args.tested_tree / path).read_bytes()
        if live != tested:
            assert path.endswith('.py') and (
                ast.dump(ast.parse(live)) == ast.dump(ast.parse(tested))
            ), ('live code differs from tested snapshot', path)
            comment_only_paths.append(path)
    assert len(data) < 720000, len(data)
    assert not any(s in data for s in [b'GIT binary patch', b'old mode ', b'new mode ', b'rename from ', b'rename to '])
    longest = 0
    for path, part in new.items():
        assert path.startswith('python/sglang/') and '/model_loader/' not in path, path
        assert Path(path).suffix in {'.py', '.cu', '.cuh', '.h', '.hpp', '.c', '.cc', '.cpp', '.json'}, path
        if path.endswith('.py'):
            ast.parse((TREE / path).read_text(), filename=path)
        for line in part.splitlines():
            if line.startswith(b'+') and not line.startswith(b'+++'):
                longest = max(longest, len(line[1:].decode()))
                assert len(line[1:].decode()) <= 400 and len(line[1:]) <= 400, (path, len(line)-1)
    subprocess.run(['git', '-C', str(TREE), 'diff', '--check', BASE], check=True)
    assert data == git('diff', '--no-ext-diff', '--no-renames', BASE, '--', 'python/sglang'), 'source changed while generating'
    patch = ROOT / 'serving/patch.diff'
    temp = patch.with_suffix('.diff.tmp')
    temp.write_bytes(data)
    subprocess.run(['git', '-C', '/data2/jihoon/muse_kernel/sglang-stock', 'apply', '--check', str(temp)], check=True)
    worker = Path('/data2/jihoon/muse_kernel/harness/results/host-evidence/worker-hunks.diff')
    subprocess.run(['git', '-C', str(TREE), 'apply', '--reverse', '--check', str(worker)], check=True)
    os.replace(temp, patch)
    result = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), files=len(new),
                  longest_added_line=longest, old_paths_preserved=True,
                  source_matches_tested_tree=str(args.tested_tree),
                  comment_only_paths=comment_only_paths,
                  incoming_moe_lane_diff_preserved=True,
                  worker_hunks_preserved=True, clean_base_apply=True, source_ast_whitespace=True,
                  unchanged_accepted_paths=sorted(p for p in old if new[p] == old[p]),
                  changed_accepted_paths=sorted(p for p in old if new[p] != old[p]),
                  new_paths=sorted(new.keys() - old.keys()))
    args.evidence.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
