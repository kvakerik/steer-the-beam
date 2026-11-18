"""FastAPI Backend for TT20 Beam Steering Live Interactive Demo."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from beamsim.env_api import BeamlineEnv, evaluate_env, get_env_true_trajectory
from agents.bayes import BayesAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = REPO_ROOT / "runs" / "harbor" / "opencode"

app = FastAPI(title="TT20 Beam Steering Interactive Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.api_route("/", methods=["GET", "HEAD"])   # HEAD too: some link-preview crawlers probe first
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return JSONResponse({"status": "ok", "message": "Demo webapp running"})


@app.api_route("/og.png", methods=["GET", "HEAD"])
async def get_og_image():
    """1200x630 link-preview image (Open Graph / Twitter card)."""
    if not (STATIC_DIR / "og.png").exists():
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    return FileResponse(
        STATIC_DIR / "og.png",
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/replays")
async def list_replays():
    """List available Gemini replay runs per seed."""
    results: dict[int, list[dict[str, Any]]] = {s: [] for s in range(5)}
    if RUNS_DIR.exists():
        for f in sorted(RUNS_DIR.glob("beam-steering-s*.json")):
            try:
                name = f.stem
                parts = name.split("_")
                seed_str = parts[0].replace("beam-steering-s", "")
                seed = int(seed_str)
                trial = int(parts[1]) if len(parts) > 1 else 1
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                shots_count = len(data.get("shots", []))
                reward = data.get("reward")
                solved = data.get("solved", False)
                if seed in results:
                    results[seed].append({
                        "file": f.name,
                        "trial": trial,
                        "shots_used": shots_count,
                        "reward": reward,
                        "solved": solved,
                    })
            except Exception:
                continue
    return JSONResponse(results)


@app.get("/api/machine_info")
async def get_nominal_machine_info():
    """Return nominal machine geometry and lattice parameters."""
    env = BeamlineEnv(seed=0)
    info = env.get_machine_info()
    return JSONResponse(info)

PREALIGNED_H = {
    0: [0.000123, -0.000208, -0.000408, -0.000434, 0.000285, 0.000369, 9.4e-05, 0.000203, 4e-05, 0.000402],
    1: [1.1e-05, 0.000399, -0.000318, 0.000414, -0.000166, -6.8e-05, 0.000298, -8.1e-05, 4.4e-05, -0.000423],
    2: [-0.000214, -0.000183, 0.000285, -0.000386, 8.7e-05, 0.000201, -0.000283, -0.000394, -0.000201, 0.000143],
    3: [-0.000358, -0.000249, 0.000269, 7.5e-05, -0.000369, -6e-05, -2e-05, -0.000308, 0.000215, -0.000325],
    4: [0.000404, 1e-05, 0.000415, -0.000372, 0.0001, -0.000111, 0.000275, -0.000292, 0.000345, 3.9e-05],
}


def true_path_at_checkpoints(env: BeamlineEnv) -> tuple[list[float], list[float]]:
    """True beam position (m) at every aperture checkpoint, for drawing only.

    Same formula as BeamlineSim._evaluate (x_ap = R_ap @ (Δθ − g·θ), report eq. 1);
    read-only, so the simulator itself is untouched. Website god-view only.
    """
    sim = env._sim
    kick_h = sim.delta_theta_h - sim.model.h.corrector_gains * sim.current_theta_h
    kick_v = sim.delta_theta_v - sim.model.v.corrector_gains * sim.current_theta_v
    x_ap_h = sim.model.h.r_matrix_ap @ kick_h
    x_ap_v = sim.model.v.r_matrix_ap @ kick_v
    return [float(x) for x in x_ap_h], [float(x) for x in x_ap_v]


def format_state_payload(
    env: BeamlineEnv,
    mode: str,
    seed: int,
    shot: int,
    theta_h: list[float],
    theta_v: list[float],
    obs_h: list[dict[str, Any]],
    obs_v: list[dict[str, Any]],
    done: bool,
    agent_text: str = "",
    agent_stats: Optional[dict[str, Any]] = None,
    sandbox: bool = False,
) -> dict[str, Any]:
    """Build unified WebSocket state payload."""
    score_details = evaluate_env(env)
    traj = get_env_true_trajectory(env)
    info = env.get_machine_info()
    x_ap_h, x_ap_v = true_path_at_checkpoints(env)

    budget_val = None if sandbox else 60
    shots_rem = None if sandbox else max(0, 60 - env._sim.shots_used)
    is_done = False if sandbox else (done or env._submitted or (env._sim.shots_used >= 60))

    return {
        "type": "state",
        "mode": mode,
        "seed": seed,
        "shot": shot,
        "budget": budget_val,
        "shots_used": env._sim.shots_used,
        "shots_remaining": shots_rem,
        "done": is_done,
        "sandbox": sandbox,
        "theta_h": [float(x) for x in theta_h],
        "theta_v": [float(x) for x in theta_v],
        "obs_h": obs_h,
        "obs_v": obs_v,
        "score": {
            "reward": float(score_details.reward),
            "s0": float(score_details.s0),
            "sf": float(score_details.sf),
            "centring_v": float(sum(traj.get("s_hat_v", [])) / len(traj.get("s_hat_v", []))) * 100 if traj.get("s_hat_v") else 0.0,
            "centring_all": float(traj.get("s_mean", 0.0)) * 100,
            "progress": float(score_details.progress),
            "solved": bool(score_details.solved),
            "is_lost": bool(score_details.is_lost),
            "loss_s": score_details.loss_s,
        },
        "god_view": {
            "x_sem_h": [float(x) for x in traj["x_sem_h"]],
            "x_sem_v": [float(x) for x in traj["x_sem_v"]],
            "is_lost": bool(traj["is_lost"]),
            "loss_s": traj["loss_s"],
            "first_lost_sem_h": traj["first_lost_sem_h"],
            "first_lost_sem_v": traj["first_lost_sem_v"],
            "s_mean": float(traj["s_mean"]),
            "distance_d": float(traj["distance_d"]),
            "x_ap_h": x_ap_h,
            "x_ap_v": x_ap_v,
            "s_hat_h": [float(x) for x in traj["s_hat_h"]],
            "s_hat_v": [float(x) for x in traj["s_hat_v"]],
        },
        "machine_info": {
            "s_sem_h": info["h"]["s_sem"],
            "sigma_sem_h": info["h"]["sigma_sem"],
            "foil_length_h": info["h"]["foil_length"],
            "s_sem_v": info["v"]["s_sem"],
            "sigma_sem_v": info["v"]["sigma_sem"],
            "foil_length_v": info["v"]["foil_length"],
            "s_corr_h": info["h"]["s_corr"],
            "s_corr_v": info["v"]["s_corr"],
            "s_ap": info["checkpoints"]["s_ap"],
            "a_h_ap": info["checkpoints"]["a_h_ap"],
            "a_v_ap": info["checkpoints"]["a_v_ap"],
            "theta_max": float(info["theta_max"]),
        },
        "agent_text": agent_text,
        "agent_stats": agent_stats or {},
    }


class SessionManager:
    """Manages state for an active WebSocket client connection."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.mode = "human"
        self.seed = 0
        self.sandbox = False
        self.env: Optional[BeamlineEnv] = None
        self.speed = 1.0
        self.cancel_event = threading.Event()
        self.paused = threading.Event()   # set = Watch playback paused
        self.active_thread: Optional[threading.Thread] = None
        self.loop = asyncio.get_event_loop()

    def reset_env(self, seed: int, budget: int = 60):
        self.seed = seed
        self.cancel_active_task()
        self.env = BeamlineEnv(seed=seed, budget=budget)
        return self.env

    def cancel_active_task(self):
        self.cancel_event.set()
        if self.active_thread and self.active_thread.is_alive():
            self.active_thread.join(timeout=1.0)
        self.cancel_event = threading.Event()
        self.active_thread = None

    async def send_state(self, payload: dict[str, Any]):
        try:
            await self.ws.send_text(json.dumps(payload))
        except Exception:
            pass

    def send_state_threadsafe(self, payload: dict[str, Any]):
        try:
            asyncio.run_coroutine_threadsafe(self.send_state(payload), self.loop)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Watch tabs: one playback mechanism for both the algorithm and the Gemini replay
# ---------------------------------------------------------------------------
BAYES_DIR = Path(__file__).resolve().parent / "data" / "bayes"
# Same honest reference as the benchmark: independent agent seed, roughening 0.2, 100k particles.
BAYES_PARAMS = {"n_particles": 100_000, "chunk_size": 25_000, "roughening": 0.2}
BAYES_SEED_OFFSET = 1_000_003

_bayes_cache: dict[int, list[dict[str, list[float]]]] = {}
_bayes_cache_lock = threading.Lock()


def _load_bayes_recordings() -> None:
    """Load precomputed reference runs (made by `python -m webapp.record_bayes`)."""
    if not BAYES_DIR.exists():
        return
    for f in sorted(BAYES_DIR.glob("seed_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            _bayes_cache[int(data["seed"])] = data["shots"]
        except Exception:
            continue


_load_bayes_recordings()


class ShotFeed:
    """Corrector settings for one playback: a finished list, or filled live by a producer thread."""

    def __init__(self, shots: Optional[list[dict[str, Any]]] = None):
        self._shots: list[dict[str, Any]] = list(shots) if shots is not None else []
        self._finished = shots is not None
        self._cond = threading.Condition()

    def append(self, shot: dict[str, Any]) -> None:
        with self._cond:
            self._shots.append(shot)
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self._finished = True
            self._cond.notify_all()

    def wait_for(self, idx: int, cancel: threading.Event) -> Optional[dict[str, Any]]:
        """Return shot `idx`, waiting for the producer if needed; None when there is no such shot."""
        with self._cond:
            while idx >= len(self._shots) and not self._finished:
                if cancel.is_set():
                    return None
                self._cond.wait(timeout=0.1)
            return self._shots[idx] if idx < len(self._shots) else None

    def __len__(self) -> int:
        with self._cond:
            return len(self._shots)

    @property
    def finished(self) -> bool:
        with self._cond:
            return self._finished

    def snapshot(self) -> list[dict[str, Any]]:
        with self._cond:
            return list(self._shots)


def compute_bayes_run(seed: int, feed: ShotFeed, cancel: threading.Event) -> Optional[BeamlineEnv]:
    """Run the honest reference BayesAgent on BeamlineEnv(seed); every shot is appended to `feed`.

    Returns the finished env, or None if cancelled. Replaying the recorded settings through a
    fresh BeamlineEnv(seed) reproduces the run exactly (same seed, same shot sequence).
    """

    class RecordingEnv(BeamlineEnv):
        def set_correctors(self, h, v):
            if cancel.is_set():
                raise InterruptedError("cancelled")
            res = super().set_correctors(h, v)
            feed.append({
                "theta_h": self._sim.current_theta_h.tolist(),
                "theta_v": self._sim.current_theta_v.tolist(),
            })
            return res

    env = RecordingEnv(seed=seed, budget=60)
    agent = BayesAgent(**BAYES_PARAMS, seed=seed + BAYES_SEED_OFFSET)
    try:
        agent.solve(env)
    except InterruptedError:
        feed.finish()
        return None
    feed.finish()
    return env


GEMINI_TRIAL = 2   # default recorded Gemini run (the trial-1 files are empty)


def gemini_runs(seed: int) -> list[tuple[int, dict[str, Any]]]:
    """All valid recorded OpenCode + Gemini runs for `seed` as (trial, data), sorted by trial."""
    runs: list[tuple[int, dict[str, Any]]] = []
    for f in RUNS_DIR.glob(f"beam-steering-s{seed:03d}_*.json"):
        try:
            trial = int(f.stem.split("_")[-1])
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("shots"):
            runs.append((trial, data))
    return sorted(runs, key=lambda r: r[0])


def load_gemini_run(seed: int, trial: int = GEMINI_TRIAL) -> Optional[dict[str, Any]]:
    """Recorded Gemini run for `seed`: the requested trial if valid, else the first valid one."""
    runs = gemini_runs(seed)
    for t, data in runs:
        if t == trial:
            return data
    return runs[0][1] if runs else None


def gemini_sections(blocks: list[str], n_shots: int) -> list[dict[str, Any]]:
    """Assign each reasoning block of a recorded Gemini run to a shot number.

    The OpenCode log has no shot markers (Gemini fired most shots from its own scripts), so the
    blocks are spread evenly over the run's shots: block b (0-based) of B belongs to shot
    ceil((b + 1) * n / B). The client shows the blocks up to the current shot.
    """
    n_blocks = len(blocks)
    n = max(1, n_shots)
    return [
        {"shot": max(1, min(n, -(-(b + 1) * n // n_blocks))), "text": text}
        for b, text in enumerate(blocks)
    ]


def _replay_centring(seed: int, shots: Optional[list[dict[str, Any]]]) -> Optional[float]:
    """Final centring % of replaying `shots` on BeamlineEnv(seed), exactly as the Watch tab does.

    Centring % = 100 x the true mean normalised symmetry over all 21 sensors (S_f of the reward),
    the same number the website's CENTRING badge shows.
    """
    if not shots:
        return None
    env = BeamlineEnv(seed=seed, budget=60)
    for s in shots[:60]:
        env.set_correctors(s.get("theta_h", [0.0] * 10), s.get("theta_v", [0.0] * 12))
    env.submit()
    return float(evaluate_env(env).sf) * 100.0


_puzzle_scores: Optional[dict[str, Any]] = None


@app.get("/api/puzzle_scores")
async def get_puzzle_scores():
    """Final centring % of the recorded runs per website puzzle (seeds 0-4).

    Algorithm: one recorded run per puzzle. Gemini: every valid recorded run; "gemini" is the mean
    of that puzzle's runs, and "gemini_runs" lists each run (the Watch tab replays them in turn).
    The averages are over the 5 puzzles.
    """
    global _puzzle_scores
    if _puzzle_scores is None:
        puzzles: dict[int, dict[str, Any]] = {}
        for seed in range(5):
            with _bayes_cache_lock:
                bayes_shots = _bayes_cache.get(seed)
            runs = [{"trial": t, "centring": _replay_centring(seed, d["shots"])} for t, d in gemini_runs(seed)]
            vals = [r["centring"] for r in runs if r["centring"] is not None]
            puzzles[seed] = {
                "bayes": _replay_centring(seed, bayes_shots),
                "gemini": sum(vals) / len(vals) if vals else None,
                "gemini_runs": runs,
            }
        average: dict[str, Optional[float]] = {}
        for agent in ("bayes", "gemini"):
            vals = [p[agent] for p in puzzles.values() if p[agent] is not None]
            average[agent] = sum(vals) / len(vals) if vals else None
        _puzzle_scores = {"metric": "centring_percent", "puzzles": puzzles, "average": average}
    return JSONResponse(_puzzle_scores)


def _bayes_producer(seed: int, feed: ShotFeed, cancel: threading.Event) -> None:
    """Fallback when no recording exists: compute live, then cache for the next viewer."""
    try:
        env = compute_bayes_run(seed, feed, cancel)
    except Exception:
        feed.finish()
        return
    if env is not None:
        with _bayes_cache_lock:
            _bayes_cache[seed] = feed.snapshot()


def _wait_while_paused(session: "SessionManager", cancel: threading.Event) -> bool:
    while session.paused.is_set() and not cancel.is_set():
        cancel.wait(0.05)
    return not cancel.is_set()


def _paced_delay(session: "SessionManager", cancel: threading.Event) -> bool:
    """Wait 1/speed seconds of *unpaused* time; speed changes apply immediately."""
    waited = 0.0
    while not cancel.is_set():
        target = 1.0 / max(0.1, session.speed)
        if waited >= target:
            return True
        step = min(0.02, target - waited)
        time.sleep(step)
        if not session.paused.is_set():
            waited += step
    return False


def run_playback(
    session: "SessionManager",
    env: BeamlineEnv,
    mode: str,
    feed: ShotFeed,
    cancel: threading.Event,
    text_blocks: Optional[list[str]] = None,
    final_text: str = "",
) -> None:
    """Step the displayed machine through `feed` at the session's speed; honours pause/resume."""
    idx = 0
    n_planned = len(feed) if feed.finished else 60
    while not cancel.is_set():
        if not _wait_while_paused(session, cancel):
            return
        shot = feed.wait_for(idx, cancel)
        if shot is None:
            break
        if session.paused.is_set():   # paused while waiting for the producer: don't step yet
            continue
        obs = env.set_correctors(shot["theta_h"], shot["theta_v"])
        idx += 1

        text_slice = ""
        if text_blocks:
            target = max(1, int(idx * len(text_blocks) / max(1, n_planned)))
            text_slice = "\n\n---\n\n".join(text_blocks[:target])

        payload = format_state_payload(
            env=env, mode=mode, seed=session.seed, shot=env._sim.shots_used,
            theta_h=list(shot["theta_h"]), theta_v=list(shot["theta_v"]),
            obs_h=obs["h"], obs_v=obs["v"], done=False, agent_text=text_slice,
        )
        payload["done"] = False
        session.send_state_threadsafe(payload)

        if env._sim.shots_used >= 60 or (feed.finished and idx >= len(feed)):
            break
        if not _paced_delay(session, cancel):
            return

    if cancel.is_set():
        return
    if not env._submitted:
        env.submit()
    last = env._sim.last_result
    fin = format_state_payload(
        env=env, mode=mode, seed=session.seed, shot=env._sim.shots_used,
        theta_h=env._sim.current_theta_h.tolist(), theta_v=env._sim.current_theta_v.tolist(),
        obs_h=[{"ratio": o.ratio, "level": o.level} for o in last.obs_h],
        obs_v=[{"ratio": o.ratio, "level": o.level} for o in last.obs_v],
        done=True, agent_text=final_text,
    )
    session.send_state_threadsafe(fin)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SessionManager(websocket)

    env = session.reset_env(seed=0, budget=100000)
    session.mode = "play"
    session.sandbox = True
    th_h = PREALIGNED_H.get(0, [0.0] * 10)
    obs = env.set_correctors(th_h, [0.0] * 12)
    init_payload = format_state_payload(
        env=env,
        mode="play",
        seed=0,
        shot=0,
        theta_h=th_h,
        theta_v=[0.0] * 12,
        obs_h=obs["h"],
        obs_v=obs["v"],
        done=False,
        sandbox=True,
    )
    await session.send_state(init_payload)

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                msg = json.loads(raw_data)
            except Exception:
                continue

            msg_type = msg.get("type")

            if msg_type == "init":
                session.cancel_active_task()
                session.mode = msg.get("mode", "play")
                session.seed = int(msg.get("seed", 0))
                session.speed = float(msg.get("speed", 3.0))

                if session.mode == "play":
                    session.sandbox = True
                    budget = 100000
                    env = session.reset_env(session.seed, budget=budget)
                    th_h = PREALIGNED_H.get(session.seed, [0.0] * 10)
                    th_v = [0.0] * 12
                    obs = env.set_correctors(th_h, th_v)
                    obs_h = obs["h"]
                    obs_v = obs["v"]
                else:
                    session.sandbox = bool(msg.get("sandbox", False))
                    budget = 100000 if session.sandbox else 60
                    env = session.reset_env(session.seed, budget=budget)
                    th_h = [0.0] * 10
                    th_v = [0.0] * 12
                    obs_h = [{"ratio": o.ratio, "level": o.level} for o in env._sim.initial_result.obs_h]
                    obs_v = [{"ratio": o.ratio, "level": o.level} for o in env._sim.initial_result.obs_v]

                payload = format_state_payload(
                    env=env,
                    mode=session.mode,
                    seed=session.seed,
                    shot=0,
                    theta_h=th_h,
                    theta_v=th_v,
                    obs_h=obs_h,
                    obs_v=obs_v,
                    done=False,
                    sandbox=session.sandbox,
                )
                await session.send_state(payload)

            elif msg_type == "fire":
                if session.env is None:
                    continue
                if not session.sandbox and (session.env._submitted or session.env._sim.shots_used >= 60):
                    continue
                if session.mode == "play":
                    th_h = PREALIGNED_H.get(session.seed, [0.0] * 10)
                    th_v = msg.get("theta_v", [0.0] * 12)
                else:
                    th_h = msg.get("theta_h", [0.0] * 10)
                    th_v = msg.get("theta_v", [0.0] * 12)
                try:
                    obs = session.env.set_correctors(th_h, th_v)
                except Exception as e:
                    await session.ws.send_text(json.dumps({"type": "error", "message": str(e)}))
                    continue

                is_done = False if session.sandbox else (session.env._sim.shots_used >= 60)
                payload = format_state_payload(
                    env=session.env,
                    mode=session.mode,
                    seed=session.seed,
                    shot=session.env._sim.shots_used,
                    theta_h=th_h,
                    theta_v=th_v,
                    obs_h=obs["h"],
                    obs_v=obs["v"],
                    done=is_done,
                    sandbox=session.sandbox,
                )
                await session.send_state(payload)

            elif msg_type == "submit":
                if session.env is not None and not session.env._submitted:
                    session.env.submit()
                last_res = session.env._sim.last_result
                payload = format_state_payload(
                    env=session.env,
                    mode=session.mode,
                    seed=session.seed,
                    shot=session.env._sim.shots_used,
                    theta_h=session.env._sim.current_theta_h.tolist(),
                    theta_v=session.env._sim.current_theta_v.tolist(),
                    obs_h=[{"ratio": o.ratio, "level": o.level} for o in last_res.obs_h],
                    obs_v=[{"ratio": o.ratio, "level": o.level} for o in last_res.obs_v],
                    done=True,
                )
                await session.send_state(payload)


            elif msg_type == "set_speed":
                session.speed = max(0.1, float(msg.get("speed", 3.0)))

            elif msg_type in ("pause", "stop"):
                session.paused.set()

            elif msg_type == "resume":
                if "speed" in msg:
                    session.speed = max(0.1, float(msg.get("speed", session.speed)))
                session.paused.clear()

            elif msg_type in ("start_bayes", "start_gemini"):
                is_bayes = msg_type == "start_bayes"
                session.mode = "bayes" if is_bayes else "gemini"
                session.seed = int(msg.get("seed", session.seed))
                session.speed = max(0.1, float(msg.get("speed", 3.0)))
                env = session.reset_env(session.seed)   # also cancels any running playback
                session.paused.clear()
                cancel = session.cancel_event

                text_blocks: list[str] = []
                final_text = ""
                if is_bayes:
                    with _bayes_cache_lock:
                        recorded = _bayes_cache.get(session.seed)
                    if recorded is not None:
                        feed = ShotFeed(recorded)
                    else:
                        feed = ShotFeed()
                        threading.Thread(
                            target=_bayes_producer, args=(session.seed, feed, cancel), daemon=True,
                        ).start()
                else:
                    run_data = load_gemini_run(session.seed, int(msg.get("trial", GEMINI_TRIAL)))
                    if run_data is None:
                        await session.ws.send_text(json.dumps({
                            "type": "error",
                            "message": f"No replay file found for seed {session.seed}",
                        }))
                        continue
                    feed = ShotFeed([
                        {"theta_h": s.get("theta_h", [0.0] * 10), "theta_v": s.get("theta_v", [0.0] * 12)}
                        for s in run_data.get("shots", [])
                    ])
                    text_blocks = [b.strip() for b in run_data.get("agent_text", "").split("\n\n\n\n") if b.strip()]

                init_p = format_state_payload(
                    env=env,
                    mode=session.mode,
                    seed=session.seed,
                    shot=0,
                    theta_h=[0.0] * 10,
                    theta_v=[0.0] * 12,
                    obs_h=[{"ratio": o.ratio, "level": o.level} for o in env._sim.initial_result.obs_h],
                    obs_v=[{"ratio": o.ratio, "level": o.level} for o in env._sim.initial_result.obs_v],
                    done=False,
                )
                if not is_bayes:
                    # The whole transcript is sent once, as per-shot sections; the client reveals them
                    # as the replay advances (no growing transcript in every shot's message).
                    init_p["agent_sections"] = gemini_sections(text_blocks, len(feed))
                await session.send_state(init_p)

                th = threading.Thread(
                    target=run_playback,
                    args=(session, env, session.mode, feed, cancel, None, ""),
                    daemon=True,
                )
                session.active_thread = th
                th.start()

    except WebSocketDisconnect:
        session.cancel_active_task()
    except Exception:
        session.cancel_active_task()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.server:app", host="0.0.0.0", port=8000, reload=False)

