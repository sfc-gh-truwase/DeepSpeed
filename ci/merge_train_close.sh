#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# After a merge-train aggregate PR is merged, closes every original PR it
# cherry-picked, pointing authors at the aggregate PR
# (.github/workflows/merge-train-close.yml).
#
# Each original PR is closed, never merged directly -- its change already
# landed on master as part of the aggregate PR's squash commit. `git
# cherry-pick` preserves each commit's original author, so GitHub
# automatically adds a Co-authored-by trailer to that squash commit for every
# distinct author among the batched PRs (see merge_train_batch.sh); the
# close comment below additionally points each author at the aggregate PR
# where they're credited.
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
    echo "closing original PR #$n (without merging it), superseded by #$AGGREGATE_PR_NUMBER"
    run gh pr close "$n" --repo "$GITHUB_REPOSITORY" --comment \
        "Closing this PR without merging it: its change already landed on master via the merge-train batch #$AGGREGATE_PR_NUMBER, which has just merged. You're credited as a contributor on that PR."
done
