@AGENTS.md

## Claude Code notes

- The development machine is Windows 11. The Bash tool is Git Bash; PowerShell is also
  available. Use `uv run …` from the repo root (the uv binary is on PATH), never a bare
  system `python`, which is a different interpreter without the project dependencies.
- Prefer `uv run --frozen` while `uv.lock` has uncommitted changes.
- Full-suite and full-CNS commands run for minutes. Run them in the background and write
  their output to a file in your scratchpad, not to `outputs/`.
- Console output on this machine uses a cp1254 code page; scripts that print non-ASCII
  characters (for example `→` in docstrings) can raise `UnicodeEncodeError` unless you set
  `PYTHONIOENCODING=utf-8`.
- Start each task by reading `docs/PROJECT_STATE.md`, as AGENTS.md section 1 requires.
