#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# After a merge-train aggregate PR is merged, closes every original PR it
# cherry-picked, pointing authors at the aggregate PR
# (.github/workflows/merge-train-close.yml).
#
# Note (documented, not engineered around): the aggregate PR merges squash-only,
# so the final commit on master is attributed to whoever merged it -- per-author
# git attribution is only preserved up to the cherry-pick stage. Maintainers
# merging an individual batch that should keep author attribution can use a
# non-squash merge for that one PR instead.
set -euo pipefail

AGGREGATE_PR_NUMBER="$1"
BODY="$2"
DRY_RUN="${DRY_RUN:-false}"

run() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

numbers=$(printf '%s' "$BODY" | grep -oE '^- #[0-9]+' | grep -oE '[0-9]+' || true)

if [ -z "$numbers" ]; then
    echo "no '- #<number>' entries found in aggregate PR #$AGGREGATE_PR_NUMBER body, nothing to close"
    exit 0
fi

for n in $numbers; do
    echo "closing original PR #$n, superseded by #$AGGREGATE_PR_NUMBER"
    run gh pr close "$n" --repo "$GITHUB_REPOSITORY" --comment \
        "Merged via bulk batch in #$AGGREGATE_PR_NUMBER."
done
