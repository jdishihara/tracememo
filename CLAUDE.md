# tracememo

Read `SPEC.md` first: it is the project specification and defines scope, architecture,
and the milestone order. `DECISIONS.md` records choices made where the spec was ambiguous.

## Working rules

- Build in the milestone order in `SPEC.md` section 8 and stop for review after each milestone.
- Ask before adding dependencies not listed in `SPEC.md` section 3.
- No real NASA data is ever committed. `data/` is git-ignored except `data/synthetic/`.
- Every public function has a docstring and type hints. Prefer small modules and pure functions.
- LLM calls go through an interface with a fake implementation; no test hits the real API.

## Commands

```
.venv/bin/pip install -e ".[dev]"      # setup (uv is not installed on this machine)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest
.venv/bin/tracememo synth drone --out data/synthetic/drone --seed 0
.venv/bin/tracememo build --config examples/drone_memo/project.yaml
```
