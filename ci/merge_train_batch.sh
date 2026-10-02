#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Cherry-picks every merge-train-labeled PR onto a fresh integration branch
# and opens one aggregate PR against master for bulk review+merge
# (.github/workflows/merge-train.yml).
#
# Requires GH_TOKEN and GITHUB_REPOSITORY in the environment, and a checkout
# of the repo (origin/master fetched) in the current working directory with
# `contents: write` / `pull-requests: write` permissions on the token.
#
# A failed gh call must fail the script; a failed cherry-pick for one PR must
# not abort the batch for the others.
set -euo pipefail

MERGE_TRAIN_LABEL="merge-train"
BULK_PR_LABEL="bulk-merge-pr"
DRY_RUN="${DRY_RUN:-false}"
BRANCH="merge-train/$(date -u +%Y%m%dT%H%M%SZ)"

run() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

main() {
    git fetch origin master
    git checkout -B "$BRANCH" origin/master

    local candidates
    candidates=$(gh pr list --repo "$GITHUB_REPOSITORY" --label "$MERGE_TRAIN_LABEL" \
        --state open --json number,title --jq '.[] | "\(.number)\t\(.title)"')

    if [ -z "$candidates" ]; then
        echo "no PRs currently labeled $MERGE_TRAIN_LABEL, nothing to batch"
        return 0
    fi

    local included=()
    local excluded=()

    while IFS=$'\t' read -r number title; do
        [ -z "$number" ] && continue

        # Re-verify right before batching: the label may be hours/days stale.
        local mergeable
        mergeable=$(gh pr view "$number" --repo "$GITHUB_REPOSITORY" --json mergeable --jq '.mergeable')
        if [ "$mergeable" = "CONFLICTING" ]; then
            echo "PR #$number ($title): no longer mergeable, excluding"
            excluded+=("$number: not mergeable with current master")
            run gh pr edit "$number" --repo "$GITHUB_REPOSITORY" --remove-label "$MERGE_TRAIN_LABEL"
            run gh pr comment "$number" --repo "$GITHUB_REPOSITORY" --body \
                "Excluded from this merge-train batch: no longer mergeable against master. Please rebase; it will be re-flagged automatically once eligible again."
            continue
        fi

        git fetch origin "pull/${number}/head:pr-${number}"
        local base
        base=$(git merge-base origin/master "pr-${number}")

        if git cherry-pick -x "${base}..pr-${number}"; then
            echo "PR #$number ($title): cherry-picked cleanly"
            included+=("#$number $title")
        else
            echo "PR #$number ($title): cherry-pick conflict, excluding"
            git cherry-pick --abort || true
            excluded+=("$number: cherry-pick conflict")
            run gh pr edit "$number" --repo "$GITHUB_REPOSITORY" --remove-label "$MERGE_TRAIN_LABEL"
            run gh pr comment "$number" --repo "$GITHUB_REPOSITORY" --body \
                "Excluded from this merge-train batch: cherry-picking onto current master produced a conflict. Please rebase; it will be re-flagged automatically once eligible again."
        fi
        git branch -D "pr-${number}" >/dev/null 2>&1 || true
    done <<< "$candidates"

    if [ "${#included[@]}" -eq 0 ]; then
        echo "no PRs cherry-picked cleanly, nothing to open"
        return 0
    fi

    {
        echo "Automated merge-train batch: bulk review/merge of small, low-risk PRs."
        echo
        echo "Included:"
        printf -- '- %s\n' "${included[@]}"
        if [ "${#excluded[@]}" -gt 0 ]; then
            echo
            echo "Excluded this run (left on merge-train for next time, or already un-flagged):"
            printf -- '- %s\n' "${excluded[@]}"
        fi
    } > /tmp/merge-train-pr-body.txt

    run git push origin "$BRANCH"
    run gh pr create --repo "$GITHUB_REPOSITORY" \
        --base master --head "$BRANCH" \
        --title "merge-train: batch of ${#included[@]} small PR(s)" \
        --body-file /tmp/merge-train-pr-body.txt \
        --label "$BULK_PR_LABEL" --label will-review
}

main
