"""Align Gemini's recorded reasoning with its shots, for the website's "Watch Gemini" tab.

The OpenCode log of every Harbor trial (jobs/<job>/<trial>/agent/opencode.txt) is an ordered event
stream: reasoning/text parts interleaved with tool calls. Each shot is one completed
`beamline_set_correctors` MCP call, so every reasoning step can be assigned to the shot it precedes:
text before shot N's call -> shot N; text after the last call -> "after the last shot".

The script checks that the completed calls match the recorded shots of the run file one-to-one
(same count, same corrector values) and skips a run otherwise (the website then falls back to an
even split). Output: webapp/data/gemini_sections/<trial_name>.json

Usage:  python -m webapp.align_gemini
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs" / "harbor" / "opencode"
JOBS_DIR = REPO_ROOT / "jobs"
OUT_DIR = Path(__file__).resolve().parent / "data" / "gemini_sections"
SHOT_TOOL = "beamline_set_correctors"
THETA_MAX = 0.45e-3   # the server clips commanded angles to +-theta_max


def _events(opencode_txt: Path) -> list[dict]:
    lines = opencode_txt.read_text(encoding="utf-8", errors="replace").splitlines()
    return [json.loads(line) for line in lines if line.strip().startswith("{")]


def align_run(run: dict) -> dict | None:
    """Return {"n_shots", "sections": [{"shot", "after", "text"}]} or None if the log doesn't match."""
    log = JOBS_DIR / run["job_id"] / run["trial_name"] / "agent" / "opencode.txt"
    if not log.exists():
        return None
    shots = run.get("shots", [])
    sections: list[dict] = []
    k = 0   # completed shot calls seen so far
    for ev in _events(log):
        kind = ev.get("type")
        part = ev.get("part") or {}
        if kind in ("reasoning", "text") and part.get("text", "").strip():
            text = part["text"].strip()
            if kind == "reasoning":
                text = "[Reasoning]: " + text
            after = k >= len(shots)
            sections.append({"shot": len(shots) + 1 if after else k + 1, "after": after, "text": text})
        elif kind == "tool_use" and part.get("tool") == SHOT_TOOL and (part.get("state") or {}).get("status") == "completed":
            if k >= len(shots):
                return None   # more calls than recorded shots
            inp = part["state"].get("input") or {}
            h = np.clip(np.asarray(inp.get("h", []), float), -THETA_MAX, THETA_MAX)
            v = np.clip(np.asarray(inp.get("v", []), float), -THETA_MAX, THETA_MAX)
            rec = shots[k]
            if h.shape != (10,) or v.shape != (12,) or not (
                np.allclose(h, rec["theta_h"], atol=1e-12) and np.allclose(v, rec["theta_v"], atol=1e-12)
            ):
                return None   # call does not match the recorded shot
            k += 1
    if k != len(shots):
        return None
    return {"source": "opencode.txt event order", "n_shots": len(shots), "sections": sections}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(RUNS_DIR.glob("beam-steering-s*_*.json")):
        run = json.loads(f.read_text(encoding="utf-8"))
        if not run.get("shots"):
            continue
        aligned = align_run(run)
        if aligned is None:
            print(f"{f.name}: no reliable alignment, website keeps the even split")
            continue
        out = OUT_DIR / f"{run['trial_name']}.json"
        out.write_text(json.dumps(aligned, ensure_ascii=False), encoding="utf-8")
        n_after = sum(1 for s in aligned["sections"] if s["after"])
        print(f"{f.name}: {len(aligned['sections'])} reasoning steps over {aligned['n_shots']} shots "
              f"({n_after} after the last shot) -> {out.name}")


if __name__ == "__main__":
    main()
