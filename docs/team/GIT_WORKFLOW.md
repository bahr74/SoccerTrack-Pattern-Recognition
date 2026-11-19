# Git Collaboration Workflow (Windows, PowerShell in VS Code)

**Repository:** https://github.com/bahr74/SoccerTrack-Pattern-Recognition · **Integration branch:** `main`
**Owner of this document:** Member 4. Ownership of every file: [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §2.

Open the integrated terminal in VS Code (**Terminal → New Terminal**) and make sure it is **PowerShell**. All commands below are run from the repository root.

---

## 1. Branch model

| Branch | Owner | Purpose |
|---|---|---|
| `main` | everyone, via pull requests only | Always passes `pytest`; the data machine runs reported experiments from it |
| `feature/scratch-pipeline` | Member 1 | Data + Pipeline A |
| `feature/yolo-pipeline` | Member 2 | Pipeline B |
| `feature/evaluation` | Member 3 | Contracts + evaluation |
| `feature/visualization-docs` | Member 4 | Config, visualisation, docs, integration |

Short-lived extra branches are allowed for urgent fixes: `fix/<short-name>`, created from `main` and merged by pull request.

---

## 2. One-time setup for a new team member

```powershell
# 1. Identity (use your own name and the e-mail linked to your GitHub account)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global core.autocrlf true          # Windows line endings in the working copy, LF in the repository
git config --global pull.rebase false           # merge on pull (simplest for students)
git config --global init.defaultBranch main

# 2. Clone into a folder on a drive with a few GB free (D:\ is an example; C: is nearly full on the lead machine)
cd D:\
git clone https://github.com/bahr74/SoccerTrack-Pattern-Recognition.git pattern_prj
cd pattern_prj

# 3. Environment and local data paths: follow docs/team/SETUP_AND_DATA.md
#    (creates .venv and your own git-ignored configs/data_sources.yaml)

# 4. Install the local guard that blocks direct pushes to main (Section 9)
```

The repository owner must first add every member as a collaborator on GitHub (**Settings → Collaborators**).

### 2.1 Initial skeleton (one time, before feature work)
1. The prepared skeleton is reviewed by the team (and ChatGPT).
2. After approval, **one person** (the repository owner) commits it on a branch `setup/skeleton`, opens a pull request, and merges it into `main`.
3. Only then do members create their feature branches **from the updated `main`**.

---

## 3. Create your feature branch (after the skeleton is merged)

```powershell
git switch main
git pull origin main
git switch -c feature/scratch-pipeline          # use YOUR branch name from Section 1
git push -u origin feature/scratch-pipeline      # first push sets the upstream
```

Later sessions:
```powershell
git switch feature/scratch-pipeline
git pull                                          # gets your own branch's remote changes
```

---

## 4. Daily cycle: change → test → inspect → commit → push

```powershell
git status                                        # what changed?
git diff                                          # review your edits

pytest                                            # MUST pass before every push (fast, no data needed)

git add src/soccertrack/detector_yolo.py tests/test_detector_yolo.py     # add files BY NAME
git diff --cached --name-only                     # inspect exactly what will be committed
git commit -m "yolo: tile generation and cross-tile NMS with unit tests"
git push
```

**Rules:**
- Add files by name. Avoid `git add .` and `git add -A` unless you have just checked `git status`.
- Commit messages: `<area>: <what and why>` (`scratch:`, `yolo:`, `eval:`, `viz:`, `docs:`, `config:`).
- Commit small, coherent steps (one milestone or sub-step per commit).

---

## 5. Pull requests

1. Push your branch, then on GitHub: **Pull requests → New pull request**; base `main`, compare `feature/<yours>`.
2. Use a title naming the milestone, e.g. `Member 2 · M1: tiling and merging`.
3. Description checklist (copy it):
   ```
   - [ ] Only files I own are changed (INTEGRATION_PLAN.md §2)
   - [ ] pytest passes locally (paste the summary line)
   - [ ] The skips of this milestone are replaced by real tests (`pytest -rs` shows none left for it)
   - [ ] Outputs validated with contracts.validate_* (if this PR produces artefacts)
   - [ ] No data, weights, outputs, .npz, notebook outputs or configs/data_sources.yaml staged
   - [ ] No credentials or absolute machine paths in committed files
   - [ ] Contract or interface changes: none / approved in issue #…
   ```
4. **Review:** at least one other member reviews. Foundation PRs (`config.py`, `contracts.py`, `data_io` core) are reviewed by all four.
5. **Merge:** use **Squash and merge** for feature milestones. The author merges after approval and a passing local `pytest` on the updated branch.

---

## 6. Getting approved updates from `main` into your branch

Do this at least once a day and always before opening a pull request:

```powershell
git switch feature/scratch-pipeline
git fetch origin
git merge origin/main                              # bring the latest main into your branch
pytest                                             # re-test after merging
git push
```

---

## 7. Merge conflicts

Because each file has one owner, conflicts should be rare. When `git merge origin/main` reports a conflict:

```powershell
git status                                         # lists "both modified" files
code path\to\file.py                               # VS Code shows Accept Current / Incoming / Both
# edit until correct, remove all <<<<<<< ======= >>>>>>> markers
git add path\to\file.py
git commit                                         # completes the merge
pytest
git push
```

- If the conflicting file is **not yours**, do not choose for the owner: run `git merge --abort` and ask the owner (or Member 4 for shared docs).
- Never resolve a notebook (`.ipynb`) conflict by hand. Only Member 4 edits notebooks.

---

## 8. Shared configuration and contract changes

| You want to change | Do this |
|---|---|
| Your own config (`scratch.yaml` / `yolo.yaml` / `evaluation.yaml`) | Change it in your branch; mention it in the PR |
| Another member's config or code | Open a GitHub issue (or team message) with the exact change; the owner applies it in their branch |
| `configs/project.yaml` split values, `SHARED_CONTRACTS.md`, `contracts.py` schemas | Issue + approval from **all four** members; Member 3 (contracts) or Member 4 (project.yaml) applies it with updated fixtures/tests in one PR |
| `requirements.txt` (new dependency) | Ask the team first; Member 4 adds it after approval |

---

## 9. Protecting `main`

**On GitHub (repository owner):** **Settings → Branches → Add branch protection rule** (or **Rules → Rulesets**) for `main`:
- require a pull request before merging, with 1 approval;
- block force pushes and deletions.

On a private repository with a free plan, these rules may be unavailable. Then the local guard below is mandatory for everyone.

**Local guard (every member, once per clone):** a `pre-push` hook that refuses pushes to `main`:

```powershell
$hook = "#!/bin/sh`nwhile read local_ref local_sha remote_ref remote_sha; do`n  if [ `"`$remote_ref`" = `"refs/heads/main`" ]; then`n    echo `"Direct pushes to main are blocked. Open a pull request.`" >&2`n    exit 1`n  fi`ndone`nexit 0`n"
[IO.File]::WriteAllText("$PWD\.git\hooks\pre-push", $hook)
Get-Content .git\hooks\pre-push                    # check the content
```

Git for Windows runs hooks with its bundled shell, so no extra installation is needed. Test it safely with `git push --dry-run origin HEAD:main`, which should be refused.

Also: never run `git push --force` on `main`, and never commit while `main` is checked out. If you did, move the commit to your branch:
```powershell
git switch -c feature/<yours>-rescue              # keeps the commit on a new branch
git switch main
git reset --hard origin/main                       # ONLY after the commit is safe on the rescue branch
```

---

## 10. What must never be committed

| Never commit | Why / where it lives |
|---|---|
| `data/`, any `*.mp4`, `*.avi`, `*.mov` | Datasets stay on F: or in git-ignored `data/` |
| `*.pt`, `*.pth`, `*.onnx`, `*.ckpt`, `*.joblib`, `*.npz`, `*.npy` | Weights, models, patch sets |
| `outputs/`, `runs/` | Generated artefacts (exchange via the data machine) |
| `configs/data_sources.yaml` | Machine-specific absolute paths |
| `.venv/`, `__pycache__/`, `.pytest_cache/`, `pip_cache/`, `hf_cache/`, `tmp/` | Environment and caches |
| `.env`, `.hf_token`, any token or password | Credentials (log in with `huggingface-cli login`, which stores the token outside the repository) |
| Notebook outputs | Clear them first: `jupyter nbconvert --clear-output --inplace notebooks\phase2_baseline.ipynb` |
| Files under `docs/phase1/` | Frozen Phase 1 materials |

### 10.1 Verify `.gitignore` and the staged files before every commit

```powershell
git check-ignore -v configs/data_sources.yaml outputs/x.csv data/raw/x.mp4    # each must print a matching rule
git status --short                                                             # nothing unexpected (?? = untracked)
git diff --cached --name-only                                                  # exactly the files you intend
git diff --cached --name-only | ForEach-Object { if ((Test-Path $_) -and (Get-Item $_).Length -gt 2MB) { "TOO LARGE: $_" } }
git diff --cached -- src scripts configs tests notebooks | Select-String -Pattern "hf_[A-Za-z0-9]{20,}|F:\\|F:/"   # must print nothing
```
The last command looks for tokens and absolute F: paths in code, configs, tests and notebooks. Documentation under `docs/` may name the lead machine's paths. It must never contain tokens.

If something wrong is staged: `git restore --staged <file>`.

---

## 11. Quick reference

| Task | Command |
|---|---|
| Which branch am I on? | `git branch --show-current` |
| Switch branch | `git switch <branch>` |
| Update `main` locally | `git switch main; git pull origin main` |
| Bring `main` into my branch | `git fetch origin; git merge origin/main` |
| Discard my unstaged edits to a file | `git restore <file>` |
| Unstage a file | `git restore --staged <file>` |
| Undo the last local (unpushed) commit, keep changes | `git reset --soft HEAD~1` |
| See history | `git log --oneline --graph -15` |
