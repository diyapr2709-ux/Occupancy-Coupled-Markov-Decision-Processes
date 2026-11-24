# Code behind reports launched from uncommitted code

A report launched from uncommitted code records only a hash of that code (`git.code_diff_sha256`, computed by `git_state()` in `experiments/train_agent.py`). Where that code never reached a commit, the patch here rebuilds it:

```
git checkout 7eb042f403d6
git apply provenance/<patch>
```

`git_state()` on the result returns the hash the report records.

| Patch | Reports it rebuilds | `code_diff_sha256` |
| --- | --- | --- |
| `train_20261123_190717_20498.patch` | `runs/train/train_20261123_190717_20498.json`, `runs/train/train_20261123_190750_20538.json` | `cc07f4716fdd…` |
| `train_20261123_194806_29133.patch` | `runs/train/train_20261123_194806_29133.json` | `dc11d99b168f…` |

Each patch was rebuilt by replaying the recorded code edits onto `7eb042f`. Applied to a fresh checkout of that commit, each reproduces its report's hash exactly.

The other reports launched on Modal predate the hash, so their code cannot be checked this way: `train_20261123_184321_12020`, `train_20261123_184953_15052` and `phase_20261123_190013_15752`.
