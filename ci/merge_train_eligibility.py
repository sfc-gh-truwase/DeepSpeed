#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Decide whether a single PR is eligible for merge-train auto-batching
(.github/workflows/merge-train-triage.yml).

Usage: merge_train_eligibility.py PR_JSON_FILE CONFIG_FILE

PR_JSON_FILE: output of
  gh pr view <n> --json isDraft,mergeable,additions,deletions,changedFiles,files,labels
CONFIG_FILE: .github/merge-train-config.yml

The required-checks-green condition is checked by the caller (it needs a
separate `gh pr checks` call per PR); this script only judges the diff shape,
since that's the part worth keeping in one place and unit-testable in
isolation. Always exits 0 -- a PR that doesn't qualify is normal input, not a
script error -- and prints exactly one of:
  ELIGIBLE
  INELIGIBLE: <human-readable reason>
"""
import fnmatch
import json
import sys

try:
    import yaml
except ImportError:
    print("INELIGIBLE: pyyaml not available")
    sys.exit(0)


def decide(pr, cfg):
    if pr.get("isDraft"):
        return "INELIGIBLE: draft PR"
    if pr.get("mergeable") == "CONFLICTING":
        return "INELIGIBLE: merge conflicts with base"

    changed_lines = pr.get("additions", 0) + pr.get("deletions", 0)
    changed_files = pr.get("changedFiles", 0)
    if changed_lines > cfg["max_changed_lines"]:
        return f"INELIGIBLE: {changed_lines} changed lines > max {cfg['max_changed_lines']}"
    if changed_files > cfg["max_changed_files"]:
        return f"INELIGIBLE: {changed_files} changed files > max {cfg['max_changed_files']}"

    labels = {l["name"] for l in pr.get("labels", [])}
    excluded = sorted(set(cfg.get("excluded_labels", [])) & labels)
    if excluded:
        return f"INELIGIBLE: carries excluded label(s) {excluded}"

    gpu_globs = cfg.get("gpu_sensitive_paths", [])
    touched = [f["path"] for f in pr.get("files", [])]
    hits = [p for p in touched if any(fnmatch.fnmatch(p, g) for g in gpu_globs)]
    if hits:
        return f"INELIGIBLE: touches GPU-sensitive path(s) {hits[:3]}"

    return "ELIGIBLE"


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} PR_JSON_FILE CONFIG_FILE", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        pr = json.load(f)
    with open(sys.argv[2]) as f:
        cfg = yaml.safe_load(f)
    print(decide(pr, cfg))


if __name__ == "__main__":
    main()
