# GitHub Publishing Notes

The publishable branch should contain only source code, configs, notebooks,
tests, lightweight documentation, `.gitkeep` placeholders, and small manifest
indexes such as `docs/runs_logbook.md`.

Do not publish branches whose reachable history contains datasets,
checkpoints, exported models, generated figures, TensorBoard logs, run archives,
or Colab transfer bundles. Removing these files from the latest commit is not
enough if they still exist in parent commits.

Before pushing a final branch, inspect reachable large objects for that branch:

```bash
git rev-list --objects <branch> |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  sort -k3 -n |
  tail -20
```

If large historic artifacts appear, publish from a clean snapshot branch instead
of a branch based on the old history. The clean snapshot should keep local data
and run products on disk, but exclude them from Git through `.gitignore`.
