#!/usr/bin/env bash
# Publish one configuration index and rebuild wheels/index.html from latest main.
#
# generate_config_index.sh has already written
# wheels/cu${CUDA_SHORT}-torch${TORCH_SHORT}/index.html. Committing a root
# index generated from a stale checkout conflicts with the next waiting
# update (both touch wheels/index.html) and leaves main failing the check
# that wheels/index.html matches generate_root_index.py. Each attempt
# resets to the remote branch, reapplies this configuration's page, and
# regenerates the root page before committing.
set -euo pipefail

: "${CUDA_VERSION:?CUDA_VERSION is required}"
: "${PYTORCH_VERSION:?PYTORCH_VERSION is required}"
: "${CUDA_SHORT:?CUDA_SHORT is required}"
: "${TORCH_SHORT:?TORCH_SHORT is required}"

index="wheels/cu${CUDA_SHORT}-torch${TORCH_SHORT}/index.html"
if [ ! -f "$index" ]; then
    echo "Missing $index. Run generate_config_index.sh first." >&2
    exit 1
fi

saved=$(mktemp)
cp "$index" "$saved"
cleanup() {
    rm -f "$saved"
}
trap cleanup EXIT

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

remote="${INDEX_GIT_REMOTE:-origin}"
branch="${INDEX_GIT_BRANCH:-main}"

for attempt in 1 2 3 4 5; do
    git fetch "$remote" "$branch"
    # Discard the local page so a failed push can retry from the new tip.
    git reset --hard "$remote/$branch"
    mkdir -p "$(dirname "$index")"
    cp "$saved" "$index"
    python3 generate_root_index.py
    git add "$index" wheels/index.html
    if git diff --cached --quiet; then
        echo "No index changes"
        exit 0
    fi
    git commit -m "Update index: CUDA ${CUDA_VERSION} + PyTorch ${PYTORCH_VERSION}"
    if git push "$remote" "HEAD:${branch}"; then
        echo "Pushed index update"
        exit 0
    fi
    echo "Push attempt ${attempt} failed; retrying" >&2
    sleep $((attempt * 2))
done

echo "Failed to push index update" >&2
exit 1
