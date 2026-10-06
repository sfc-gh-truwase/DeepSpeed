#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Scans open PRs and auto-labels merge-train / roadmap-pr candidates
# (.github/workflows/merge-train-triage.yml).
#
# Requires GH_TOKEN and GITHUB_REPOSITORY in the environment, and
# ci/merge_train_eligibility.py's PyYAML dependency already installed.
#
# A failed gh call must fail the script, not silently skip a PR.
set -euo pipefail

CONFIG=".github/merge-train-config.yml"
MERGE_TRAIN_LABEL="merge-train"
ROADMAP_LABEL="roadmap-pr"
DRY_RUN="${DRY_RUN:-false}"

run() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

# Roadmap-item linkage: if the PR body names a Roadmap-Item that is itself
# labeled `roadmap`, flag it roadmap-pr and skip merge-train consideration --
# roadmap work gets individual priority review, not batching.
check_roadmap_link() {
    local pr_number="$1" body="$2"
    local item
    item=$(printf '%s' "$body" | grep -oE 'Roadmap-Item:\s*#[0-9]+' | grep -oE '[0-9]+' | head -1 || true)
    [ -z "$item" ] && return 1

    if gh issue view "$item" --repo "$GITHUB_REPOSITORY" --json labels \
            --jq '.labels[].name' 2>/dev/null | grep -qx roadmap; then
        echo "PR #$pr_number links roadmap item #$item -> labeling $ROADMAP_LABEL"
        run gh pr edit "$pr_number" --repo "$GITHUB_REPOSITORY" --add-label "$ROADMAP_LABEL"
        return 0
    fi
    echo "PR #$pr_number references #$item but it is not labeled '$ROADMAP_LABEL', ignoring"
    return 1
}

checks_green() {
    local pr_number="$1"
    local bad
    bad=$(gh pr checks "$pr_number" --repo "$GITHUB_REPOSITORY" --required \
            --json bucket --jq '[.[] | select(.bucket != "pass")] | length' 2>/dev/null || echo 1)
    [ "$bad" = "0" ]
}

process_pr() {
    local pr_number="$1"
    local pr_json
    pr_json=$(gh pr view "$pr_number" --repo "$GITHUB_REPOSITORY" \
        --json isDraft,mergeable,additions,deletions,changedFiles,files,labels,body,createdAt)
    echo "$pr_json" > "/tmp/pr-${pr_number}.json"

    local body
    body=$(printf '%s' "$pr_json" | jq -r '.body // ""')
    local has_merge_train
    has_merge_train=$(printf '%s' "$pr_json" | jq -r \
        --arg l "$MERGE_TRAIN_LABEL" '[.labels[].name] | any(. == $l)')

    if check_roadmap_link "$pr_number" "$body"; then
        if [ "$has_merge_train" = "true" ]; then
            echo "PR #$pr_number is now roadmap-linked -> removing $MERGE_TRAIN_LABEL"
            run gh pr edit "$pr_number" --repo "$GITHUB_REPOSITORY" --remove-label "$MERGE_TRAIN_LABEL"
        fi
        return
    fi

    local verdict
    verdict=$(python3 ci/merge_train_eligibility.py "/tmp/pr-${pr_number}.json" "$CONFIG")
    if [ "$verdict" != "ELIGIBLE" ]; then
        if [ "$has_merge_train" = "true" ]; then
            echo "PR #$pr_number no longer eligible ($verdict) -> removing $MERGE_TRAIN_LABEL"
            run gh pr edit "$pr_number" --repo "$GITHUB_REPOSITORY" --remove-label "$MERGE_TRAIN_LABEL"
            run gh pr comment "$pr_number" --repo "$GITHUB_REPOSITORY" --body \
                "This PR no longer qualifies for merge-train auto-batching (${verdict#INELIGIBLE: }) and has been un-flagged."
        fi
        return
    fi

    if ! checks_green "$pr_number"; then
        echo "PR #$pr_number is diff-eligible but required checks aren't all green yet, skipping for now"
        return
    fi

    if [ "$has_merge_train" != "true" ]; then
        echo "PR #$pr_number is eligible and green -> labeling $MERGE_TRAIN_LABEL"
        run gh pr edit "$pr_number" --repo "$GITHUB_REPOSITORY" --add-label "$MERGE_TRAIN_LABEL"
        run gh pr comment "$pr_number" --repo "$GITHUB_REPOSITORY" --body \
            "This PR was automatically flagged for inclusion in the next merge-train batch based on its small, non-GPU diff and passing CI. If it shouldn't be bulk-merged, remove the \`$MERGE_TRAIN_LABEL\` label or add \`no-merge-train\`."
    fi
}

main() {
    local numbers
    numbers=$(gh pr list --repo "$GITHUB_REPOSITORY" --state open --limit 500 --json number --jq '.[].number')
    for n in $numbers; do
        process_pr "$n"
    done
}

main
