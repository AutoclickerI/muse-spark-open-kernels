#!/bin/sh
# Refresh upstream/ (llms.txt, the terms, the public API's JSON) and each problem's official
# files (okc init's output), so anything the organizers change shows up in `git diff`.
# Never touches a problem's submission.py once it exists. No login needed.
#
#   tools/sync_upstream.sh
set -eu
cd "$(dirname "$0")/.."
site=https://openkernelchallenge.dev
api=$site/api/v1
okc=${OKC:-$(command -v okc || echo "$HOME/.okc/bin/okc")}
json() { curl -fsS "$1" | python3 -I -m json.tool --sort-keys; }

mkdir -p upstream/api/problems
curl -fsS "$site/llms.txt" -o upstream/llms.txt
curl -fsS "$site/terms.pdf" -o upstream/terms.pdf
for p in problems serving rules; do json "$api/$p" > "upstream/api/$p.json"; done

names=$(python3 -I -c 'import json; print(" ".join(p["name"] for p in json.load(open("upstream/api/problems.json"))["problems"]))')
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for n in $names; do
  json "$api/problems/$n" > "upstream/api/problems/$n.json"
  "$okc" init "$n" "$tmp/$n" > /dev/null
  mkdir -p "problems/$n"
  for f in "$tmp/$n"/*; do
    b=$(basename "$f")
    [ "$b" = submission.py ] && [ -e "problems/$n/submission.py" ] && continue
    cp "$f" "problems/$n/$b"
  done
done
git status --short upstream problems
