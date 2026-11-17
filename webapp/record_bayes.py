"""Precompute the reference-algorithm runs shown in the website's "Watch the algorithm" tab.

Runs the honest reference BayesAgent (100k particles, roughening 0.2, agent seed = env seed
+ 1_000_003, budget 60) on BeamlineEnv(seed), exactly as the live demo does, and stores the
corrector settings of every shot. The website replays them through a fresh BeamlineEnv(seed),
which reproduces the run exactly (checked below). Precomputing lets the speed buttons and
pause work: live computation takes 0.3-1 s per shot, which would cap the replay speed.

Usage:  python -m webapp.record_bayes            # seeds 0-4 (the website's puzzles)
        python -m webapp.record_bayes 0 3        # selected seeds
"""

from __future__ import annotations

import json
import sys
import threading
import time

from beamsim.env_api import BeamlineEnv, evaluate_env
from webapp.server import BAYES_DIR, BAYES_PARAMS, BAYES_SEED_OFFSET, ShotFeed, compute_bayes_run


def record(seed: int) -> dict:
    feed = ShotFeed()
    t0 = time.time()
    env = compute_bayes_run(seed, feed, threading.Event())
    elapsed = time.time() - t0
    assert env is not None
    live = evaluate_env(env)
    shots = feed.snapshot()

    replay = BeamlineEnv(seed=seed, budget=60)
    for s in shots:
        replay.set_correctors(s["theta_h"], s["theta_v"])
    replay.submit()
    rep = evaluate_env(replay)
    if abs(rep.reward - live.reward) > 1e-12 or abs(rep.sf - live.sf) > 1e-12:
        raise RuntimeError(f"seed {seed}: replay differs from live run ({rep.reward} vs {live.reward})")

    return {
        "seed": seed,
        "agent": "BayesAgent",
        "params": {**BAYES_PARAMS, "agent_seed": seed + BAYES_SEED_OFFSET, "budget": 60},
        "reward": float(live.reward),
        "sf": float(live.sf),
        "solved": bool(live.solved),
        "is_lost": bool(live.is_lost),
        "loss_s": live.loss_s,
        "shots_used": len(shots),
        "elapsed_seconds": round(elapsed, 1),
        "shots": shots,
    }


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]] or list(range(5))
    BAYES_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        data = record(seed)
        out = BAYES_DIR / f"seed_{seed:03d}.json"
        out.write_text(json.dumps(data), encoding="utf-8")
        print(f"seed {seed}: reward {data['reward']:.3f}, {data['shots_used']} shots, "
              f"{data['elapsed_seconds']} s, replay identical -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
