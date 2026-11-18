# TT20 Beam Steering — an RL environment where frontier agents must do real inference

An agent operates a simulated CERN proton transfer line (TT20-like, SPS → North Area).
Every corrector magnet has an unknown misalignment. The agent sets 22 corrector magnets and
sees only noisy split-foil monitor readings. It has to **thread the beam through the whole
900 m line and centre it**, within a limited number of shots.

The physics and the reference algorithm follow Adam Štafa's CERN report *"Algorithms for
Steering the CERN TT20 Transfer Line"* (2026), see `docs/tt20_report.md`.

## Live Interactive Demo

Try the live interactive demo, deployed publicly on Modal:

- 👉 **v2 (redesigned for non-physicists):** [https://kvakerik--steer-the-beam-v2-fastapi-app.us-east.modal.run](https://kvakerik--steer-the-beam-v2-fastapi-app.us-east.modal.run)
- **v1 (version submitted at Junction 2026):** [https://kvakerik--tt20-beam-steering-webapp-fastapi-app.us-east.modal.run](https://kvakerik--tt20-beam-steering-webapp-fastapi-app.us-east.modal.run)

- **Play:** drag the 12 up–down magnets on the plot (or use the sliders) to thread the proton beam through the centre of every sensor. The **AI view** toggle hides the beam and shows only the noisy sensor readings the agent gets; a blind attempt in AI view is the fair comparison with the agents.
- **Watch the algorithm:** replay of recorded honest Bayes runs (seeds 0–4): the reference `BayesAgent` (100,000 particles, roughening 0.2, independent agent seed), recorded with `python -m webapp.record_bayes` and replayed through the same `BeamlineEnv(seed)`, which reproduces each run exactly.
- **Watch Gemini:** replay of the recorded OpenCode + Gemini 3.8 Flash benchmark episodes (two valid runs per puzzle, replayed in turn) with the reasoning transcript.
- Both Watch tabs have 1×/3×/10× speed and Pause/Resume. The plots show the true beam (god view), which the agents never see.

The website shows **centring %**, the true mean centring over all 21 sensors: **Algorithm 67.4 %, Gemini 29.4 %** (averages over the 5 website puzzles, using every valid recorded run). It is shown alongside the official benchmark score from 0 to 1 used in the evaluation below (**0.50 vs 0.26**), which is computed from the same centring.

Run locally:
```bash
python -m webapp.server  # serves at http://localhost:8000
```

## Why this is hard (and fair)

- **Partial observability.** Each monitor reports only a foil ratio and an intensity level,
  plus noise. Once the beam hits the pipe, every downstream monitor (in *both* planes) reads
  background only, so the agent is blind beyond the loss point.
- **Inference, not lookup.** The agent gets the full nominal model: response matrices, beam
  widths, apertures, noise level and the physics equations. Success requires combining many
  noisy shots into a belief about 22 unknown offsets and choosing informative next settings.
- **Model mismatch.** The real machine differs from the nominal model (β-beating from initial
  optics mismatch, ±2 % corrector gains, emittance and background spread, after Table 1 of the
  report). The agent is told this.
- **Limited budget.** 60 shots per episode. Random search gets nowhere in 22 dimensions.

Nothing is hidden that a good operator would need: a Bayesian particle filter using
*exactly the same tools* reaches much higher scores (below).

## Environment

| | |
|---|---|
| Goal | Thread and centre the beam: mean intensity-normalised symmetry Ŝ over all 21 monitors (solved at ≥ 0.95) |
| Actions (MCP tools) | `get_machine_info()`, `set_correctors(h[10], v[12])` (1 shot), `get_history()`, `submit()` |
| Observation | per monitor: foil ratio r, level ℓ (noisy, eq. 4 of the report) |
| Hidden state | true offsets Δθ, perturbed optics, gains. These exist only in the `beamline` sidecar container |
| Budget | 60 shots, medium randomisation, θmax = 0.45 mrad, noise ν = 0.25 |

## Reward

```
S0       = true mean Ŝ at the starting setting (θ = 0)
Sf       = true mean Ŝ at the final setting (noise-free, on the real perturbed machine)
progress = clip((Sf − S0) / (0.95 − S0), 0, 1)
reward   = 0.8 · progress + 0.2 · solved · (1 − shots_used / budget)
```

- Doing nothing → exactly 0. Stopping early → close to 0.
- The efficiency bonus applies only to solved episodes, so quitting early can't earn it.
- The score is computed by the simulator from the true state, never from anything the agent reports.

### Reward integrity

- **Zero Simulator Access in Agent Container**: The simulator, true offsets, perturbed optics, and score token live exclusively in the `beamline` sidecar container. The agent container contains only Python and NumPy with no simulator code.
- **Seed Leak Discovery & Fix**: In an earlier development version of the reference, an identical RNG stream (`agent_seed = env_seed`) leaked the true offsets into particle 0. Coupled with an artificially small roughening (0.02), this yielded an inflated score (~0.70), while an honest independent seed scored ~0.05 due to particle depletion. We resolved this by decoupling the agent RNG seed (`env_seed + 1_000_003`), tuning roughening to `0.2` with 100k particles (honest score: 0.504 on Modal, 0.469 locally), and verifying against an independently written scripted threading operator baseline (`agents/threading.py`, scoring 0.392).
- **Anti-Exploit In-Memory Startup Salt**: To prevent agents from guessing hidden offsets from the nominal seed in the task name, the sidecar server generates a fresh cryptographically random 128-bit secret salt (`secrets.token_hex(16)`) at process startup. The salt is **kept strictly in process memory** and is never written to disk, Dockerfiles, environment variables, logs, or responses. The simulator derives its effective seed from `hash(seed, salt)`. Each trial is therefore a **fresh random puzzle from the same distribution**.
- **Token-Protected Verifier Endpoint**: `/score` requires a per-task token that only the verifier container holds (mounted after the agent finishes).
- **Server-Side Guardrails**: Budget (60 shots), clipping to ±θmax, NaN/Inf rejection, and freeze-after-submit are enforced server-side.

## Results

Medium randomisation, budget 60, tasks `beam-steering-s000 … s004`, evaluated with Harbor on Modal cloud:

| Agent | Task s000 | Task s001 | Task s002 | Task s003 | Task s004 | Overall (Mean ± Std) | Solved % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference: Bayes particle filter (`oracle`, r=0.2)** | 0.392 | 0.394 | 0.657 | 0.452 | 0.624 | **0.504 ± 0.114** | 0.0% |
| **Gemini 3.8 Flash via OpenCode (3 runs/task)\*** | 0.133 ± 0.016 | 0.198 ± 0.016 | 0.310 ± 0.037 | 0.173 ± 0.021 | 0.407 ± 0.011 | **0.244 ± 0.103** | 0.0% |
| **Stop-early (1–2 shots, submit)** | 0.001 | 0.026 | 0.000 | 0.000 | 0.000 | **0.005 ± 0.010** | 0.0% |
| **Do-nothing (`nop`)** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000 ± 0.000** | 0.0% |
| **Random search (local baseline)** | 0.020 | 0.015 | 0.025 | 0.018 | 0.022 | **0.020 ± 0.004** | 0.0% |

*\*Note: Earlier Gemini pilot runs scored a mean of 0.259 (~0.26). Each trial is now a fresh random puzzle from the same medium randomization distribution.*

---

### Local Benchmark Comparison (Seeds 0–9)

Evaluated locally with strictly decoupled agent RNG seed (`seed + 1_000_003`), 60-shot budget, and medium domain randomization:

| Seed | Bayes (roughening = 0.2) | Bayes (roughening = 0.5) | Threading (`agents/threading.py`) | Gemini (`opencode`) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.4308 | 0.1586 | 0.1813 | — |
| **1** | 0.6273 | 0.1809 | 0.4790 | — |
| **2** | 0.3851 | 0.0007 | 0.3954 | — |
| **3** | 0.7439 | 0.1585 | 0.3699 | — |
| **4** | 0.6460 | 0.3916 | 0.4800 | — |
| **5** | 0.3551 | 0.1602 | 0.3978 | — |
| **6** | 0.3891 | 0.3849 | 0.0272 | — |
| **7** | 0.3944 | 0.2040 | 0.2656 | — |
| **8** | 0.3611 | 0.0501 | 0.7501 | — |
| **9** | 0.3539 | 0.2716 | 0.5700 | — |
| **Mean** | **0.4687** | **0.1961** | **0.3916** | **0.259 (~0.26)** |

---

### Where the frontier agent fails

Gemini 3.8 Flash uses bash and Python scripts to invert the nominal response matrix, successfully threading the first 4–6 monitors ($S_f \approx 0.28\text{–}0.36$). However, once downstream losses or optical mismatch occur, it resorts to local single-corrector tweaks rather than maintaining a joint multi-shot belief across all 22 offsets. It exhausts its 60-shot budget before threading the full line.

In contrast:
- **BayesAgent** updates a joint belief over all 22 offsets from every shot, including "beam lost" monitors which carry negative constraint information.
- **ThreadingAgent** progressively scans and centers monitor by monitor, locking upstream correctors and refining with regularized multi-monitor least squares.
Both reference approaches decisively outperform the LLM agent.

## Reproduce

Requirements: Python 3.11+, a [Modal](https://modal.com) account (`modal setup`), and a
Gemini API key.

```bash
pip install -r requirements.txt harbor modal
export GEMINI_API_KEY=...            # Windows: $env:GEMINI_API_KEY="..."
./scripts/run_eval.sh                # Windows: scripts\run_eval.ps1
```

This runs the oracle, 3× OpenCode + Gemini, nop and stop-early over all tasks, then prints the
results table. Single runs:

```bash
harbor run -p tasks/beam-steering-s000 -a oracle -e modal
harbor run -p tasks/beam-steering-s000 -a opencode -m gemini/gemini-3.8-flash -e modal
```

The tasks are standard Harbor multi-container tasks and also run locally with `-e docker`.
Fast local checks without Docker: `pytest -q`, `python scripts/local_eval.py`.

## Repository layout

```
beamsim/        simulator (lattice, physics eq. 1–7, randomisation, scoring, agent-facing env_api)
agents/         bayes.py (reference particle filter), baselines.py (random, stop-early)
scripts/        gen_tasks.py, local_eval.py, pilot_llm.py, run_eval.sh/.ps1
tasks/          generated Harbor tasks beam-steering-sNNN
docs/           tt20_report.md (physics source), PLAN.md
```

## Simplifications vs. the real TT20

A synthetic TT20-like lattice is used because the MAD-X optics weren't available. Other
simplifications: Gaussian beam, rectangular apertures, binary loss, uniform noise (as in the
report), and no quadrupole gradient errors.

## Credits

Physics model and Bayesian reference after Adam Štafa (CERN, supervisors A. Menor de Oñate,
M. Schenk). Built for the AfterQuery track at Junction 2026.