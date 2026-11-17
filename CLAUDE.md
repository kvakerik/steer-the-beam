# Steer the Beam: project context for Claude Code

Background from the hackathon (originally written for Cline):

@.clinerules/01-challenge.md
@.clinerules/02-project.md
@.clinerules/03-working-rules.md

## Current state (after Junction 2026), overrides anything older above
- The hackathon is over; the project was submitted solo by Erik (particle physicist). Repo: github.com/kvakerik/steer-the-beam (remote `mine` or `origin`; the old remote is a friend's repo, don't push there).
- Final settings: budget **60** shots, medium randomisation, θmax 0.45 mrad, noise ν 0.25.
- Reference = honest **BayesAgent(n_particles=100_000, roughening=0.2, seed=env_seed + 1_000_003)**. An earlier version leaked the answer through a shared seed; never share the env seed with the agent again. The sidecar draws a secret salt at startup.
- Results (Harbor on Modal): reference **0.50 ± 0.11**, Gemini 3.8 Flash **0.26 ± 0.11** (15 runs), stop-early 0.005, do-nothing 0.000. Don't change `beamsim/`, `agents/`, `mcp_server/`, `tasks/` or the scoring unless explicitly asked.
- The Gemini API key is no longer available. The website replays recorded Gemini runs from `runs/harbor/opencode/`; nothing needs the key.
- Website: `webapp/` (FastAPI + one-page `static/index.html`), deployed on Modal: `https://kvakerik--tt20-beam-steering-webapp-fastapi-app.us-east.modal.run`. Backup of the submitted version: `webapp/backup_v1/`. New design work goes to a separate Modal app (`steer-the-beam-v2`) and branch `website-v2`, keeping the original URL unchanged. Keep `min_containers=0` so idle apps cost nothing.
- Audience of the website: non-physicists. Plain language ("magnet", "sensor", "centring %"), technical details only in the collapsible "Details for physicists".

## When you finish a task
End with a short **review for Erik to paste into his other Claude chat**, in one block:
- what you changed (files, in plain words)
- what you tested and how (commands, what you saw)
- what doesn't work yet or looks off, and any decisions you made
- the URLs (local / Modal) and the commit or branch
- 2–3 open questions or next options

## Machine
- Windows + PowerShell, Python venv in `.venv` (`.venv\Scripts\Activate.ps1`); harbor and modal are installed there; Modal login is saved.
- Run the site locally: `python -m webapp.server` → http://localhost:8000. Deploy: `modal deploy webapp/<app file>.py`.
- Always `git --no-pager …`; never run commands that wait for input. Shell scripts keep LF line endings.
