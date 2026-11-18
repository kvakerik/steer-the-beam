"""Modal deployment of the v2 website (separate app, so the original URL stays unchanged)."""

from pathlib import Path
import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("steer-the-beam-v2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "numpy",
        "scipy",
    )
    .add_local_dir(str(REPO_ROOT / "beamsim"), remote_path="/root/beamsim")
    .add_local_dir(str(REPO_ROOT / "agents"), remote_path="/root/agents")
    .add_local_dir(str(REPO_ROOT / "webapp"), remote_path="/root/webapp")
    .add_local_dir(str(REPO_ROOT / "runs" / "harbor" / "opencode"), remote_path="/root/runs/harbor/opencode")
)


@app.function(
    image=image,
    cpu=2.0,
    memory=2048,
    min_containers=0,
    timeout=600,
)
@modal.asgi_app(label="steer-the-beam")   # URL: https://kvakerik--steer-the-beam.modal.run
def fastapi_app():
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from webapp.server import app as web_app
    return web_app
