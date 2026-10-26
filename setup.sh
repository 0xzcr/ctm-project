#!/usr/bin/env bash
# Fetch the upstream CTM code at the pinned commit, and install the Python packages this project needs.
# Every result from this project is computed against exactly this upstream commit.
set -euo pipefail

cd "$(dirname "$0")"

PINNED_COMMIT=4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466
UPSTREAM=third_party/continuous-thought-machines

if [ ! -d "$UPSTREAM" ]; then
  git clone https://github.com/SakanaAI/continuous-thought-machines "$UPSTREAM"
fi

git -C "$UPSTREAM" fetch --quiet origin "$PINNED_COMMIT" || true
git -C "$UPSTREAM" checkout --quiet "$PINNED_COMMIT"

pip install -r requirements.txt

echo "upstream CTM pinned at $(git -C "$UPSTREAM" rev-parse HEAD)"
