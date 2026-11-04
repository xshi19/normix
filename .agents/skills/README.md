# Skills

`.agents/skills/` is the only skill discovery directory in this repo.
Codex and Cursor both scan this directory.
Do not add a second copy of a normix skill.
Read each `SKILL.md` for when to use that skill.

## Vendored shared skills

These directories are vendored from `xshi19/xshi-skills`.
`install.sh --copy --project` wrote them at the commit below.
Change a vendored skill only by a pull request to that repo.
Do not edit the copies in this repository.

Source repository: https://github.com/xshi19/xshi-skills

Commit: `e1442246b89b13eeee62a8e84fedcc0d41ede361`

The upstream license is MIT.
A later `install.sh` run leaves an existing non-symlink directory untouched.
That run does not write this README.

### Math writing

- `notation-economy`

### Decision records

- `decision-records`

### Science stack

- `principle-code-mirrors-the-math`
- `principle-closed-form-over-general-case`
- `principle-one-source-of-truth-derive-the-rest`
- `principle-explicit-backend-no-silent-placement`
- `principle-immutable-values-for-math-objects`
- `principle-fail-loud-on-invalid-domain`
- `principle-choose-the-numerically-honest-formulation`
- `benchmarking`
- `mutation-testing`
- `numerical-review`

## Normix-owned skills

These directories are maintained in this repository.
`install.sh` does not own them.

- `agent-maintenance`
- `architect`
- `arena`
- `docs-publish`
- `figure-it-out`
- `git-conventions`
- `how`
- `interrogate`
- `no-comments`
- `normix-formal-verify`
- `running-tests`
- `show-me`
- `tdd`
- `unslop`
- `why`
