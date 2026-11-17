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


@app.get("/")
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

            elif msg_type == "start_bayes":
                session.cancel_active_task()
                session.mode = "bayes"
                session.seed = int(msg.get("seed", session.seed))
                session.speed = max(0.1, float(msg.get("speed", 3.0)))
                env = session.reset_env(session.seed)

                init_p = format_state_payload(
                    env=env,
                    mode="bayes",
                    seed=session.seed,
                    shot=0,
                    theta_h=[0.0] * 10,
                    theta_v=[0.0] * 12,
                    obs_h=[{"ratio": o.ratio, "level": o.level} for o in env._sim.initial_result.obs_h],
                    obs_v=[{"ratio": o.ratio, "level": o.level} for o in env._sim.initial_result.obs_v],
                    done=False,
                )
                await session.send_state(init_p)

                def run_bayes():
                    class LiveStreamingEnv(BeamlineEnv):
                        def set_correctors(self, h, v):
                            if session.cancel_event.is_set():
                                raise InterruptedError("Cancelled")
                            t0 = time.time()
                            res = super().set_correctors(h, v)
                            shot_num = self._sim.shots_used
                            p = format_state_payload(
                                env=self,
                                mode="bayes",
                                seed=session.seed,
                                shot=shot_num,
                                theta_h=list(h),
                                theta_v=list(v),
                                obs_h=res["h"],
                                obs_v=res["v"],
                                done=(shot_num >= 60),
                            )
                            session.send_state_threadsafe(p)

                            cur_speed = getattr(session, "speed", 1.0)
                            target_dt = 1.0 / max(0.1, cur_speed)
                            elapsed = time.time() - t0
                            delay = max(0.01, target_dt - elapsed)
                            time.sleep(delay)
                            return res

                        def submit(self):
                            res = super().submit()
                            last = self._sim.last_result
                            p = format_state_payload(
                                env=self,
                                mode="bayes",
                                seed=session.seed,
                                shot=self._sim.shots_used,
                                theta_h=self._sim.current_theta_h.tolist(),
                                theta_v=self._sim.current_theta_v.tolist(),
                                obs_h=[{"ratio": o.ratio, "level": o.level} for o in last.obs_h],
                                obs_v=[{"ratio": o.ratio, "level": o.level} for o in last.obs_v],
                                done=True,
                            )
                            session.send_state_threadsafe(p)
                            return res

                    live_env = LiveStreamingEnv(seed=session.seed, budget=60)
                    session.env = live_env

                    agent = BayesAgent(
                        n_particles=100_000,
                        chunk_size=25_000,
                        roughening=0.2,
                        seed=session.seed + 1_000_003,
                    )

                    try:
                        agent.solve(live_env)
                    except InterruptedError:
                        pass
                    except Exception as ex:
                        session.send_state_threadsafe({"type": "error", "message": f"Bayes error: {ex}"})

                th = threading.Thread(target=run_bayes, daemon=True)
                session.active_thread = th
                th.start()

            elif msg_type == "start_gemini":
                session.cancel_active_task()
                session.mode = "gemini"
                session.seed = int(msg.get("seed", session.seed))
                session.speed = max(0.1, float(msg.get("speed", 3.0)))
                trial_num = int(msg.get("trial", 2))
                env = session.reset_env(session.seed)

                run_file = RUNS_DIR / f"beam-steering-s{session.seed:03d}_{trial_num}.json"
                if not run_file.exists():
                    fallback_files = list(RUNS_DIR.glob(f"beam-steering-s{session.seed:03d}_*.json"))
                    for fb in fallback_files:
                        if fb != run_file:
                            run_file = fb
                            break

                if not run_file.exists():
                    await session.ws.send_text(json.dumps({
                        "type": "error",
                        "message": f"No replay file found for seed {session.seed}",
                    }))
                    continue

                with open(run_file, "r", encoding="utf-8") as rf:
                    run_data = json.load(rf)

                shots_data = run_data.get("shots", [])
                agent_full_text = run_data.get("agent_text", "")
                text_blocks = [b.strip() for b in agent_full_text.split("\n\n\n\n") if b.strip()]

                init_p = format_state_payload(
                    env=env,
                    mode="gemini",
                    seed=session.seed,
                    shot=0,
                    theta_h=[0.0] * 10,
                    theta_v=[0.0] * 12,
                    obs_h=[{"ratio": o.ratio, "level": o.level} for o in env._sim.initial_result.obs_h],
                    obs_v=[{"ratio": o.ratio, "level": o.level} for o in env._sim.initial_result.obs_v],
                    done=False,
                    agent_text=text_blocks[0] if text_blocks else "",
                )
                await session.send_state(init_p)

                def run_gemini():
                    n_shots = len(shots_data)
                    for idx, s in enumerate(shots_data):
                        if session.cancel_event.is_set():
                            break
                        th_h = s.get("theta_h", [0.0] * 10)
                        th_v = s.get("theta_v", [0.0] * 12)

                        obs = env.set_correctors(th_h, th_v)
                        is_last = (idx == n_shots - 1) or (env._sim.shots_used >= 60)

                        text_slice = ""
                        if text_blocks:
                            target_count = max(1, int((idx + 1) * len(text_blocks) / n_shots))
                            text_slice = "\n\n---\n\n".join(text_blocks[:target_count])

                        p = format_state_payload(
                            env=env,
                            mode="gemini",
                            seed=session.seed,
                            shot=env._sim.shots_used,
                            theta_h=th_h,
                            theta_v=th_v,
                            obs_h=obs["h"],
                            obs_v=obs["v"],
                            done=is_last,
                            agent_text=text_slice,
                        )
                        session.send_state_threadsafe(p)

                        if is_last:
                            break

                        cur_speed = getattr(session, "speed", 2.0)
                        delay = 1.0 / max(0.1, cur_speed)
                        time.sleep(delay)

                    if not env._submitted:
                        env.submit()

                    last = env._sim.last_result
                    fin_p = format_state_payload(
                        env=env,
                        mode="gemini",
                        seed=session.seed,
                        shot=env._sim.shots_used,
                        theta_h=env._sim.current_theta_h.tolist(),
                        theta_v=env._sim.current_theta_v.tolist(),
                        obs_h=[{"ratio": o.ratio, "level": o.level} for o in last.obs_h],
                        obs_v=[{"ratio": o.ratio, "level": o.level} for o in last.obs_v],
                        done=True,
                        agent_text=agent_full_text,
                    )
                    session.send_state_threadsafe(fin_p)

                th = threading.Thread(target=run_gemini, daemon=True)
                session.active_thread = th
                th.start()

    except WebSocketDisconnect:
        session.cancel_active_task()
    except Exception:
        session.cancel_active_task()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.server:app", host="0.0.0.0", port=8000, reload=False)

