"""Local chat workspace for trusted, already reconstructed SMPL-X motions."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from chat_commands import CommandError, EditPlan, parse_command

ROOT = Path(__file__).resolve().parent
WELCOME = "Tell me how to retime your motion. Try: Set speed to 1.5x. I edit the 3D motion and render both versions from the same camera."


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=500)


def write_json(path, data):
    temp = path.with_suffix(".next.json")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def motion_info(path):
    import numpy as np
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    frames = len(saved["smpl_params_global"]["transl"])
    fps = float(saved.get("fps", 30))
    if not np.isfinite(fps) or not 1 <= fps <= 240 or frames < 2 or frames / fps > 30:
        raise ValueError(
            "Use a motion with at least two frames, 1-240 FPS, and at most 30 seconds."
        )
    return dict(frames=frames, fps=fps, seconds=frames / fps, span=(frames - 1) / fps)


def subprocess_render(source, body_model, destination, plan):
    write_json(destination / "request.json", plan)
    with (destination / "render.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "chat_render.py"),
                str(source),
                str(body_model),
                str(destination),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    if completed.returncode:
        raise RuntimeError(
            "Rendering failed. Check the local render.log and OpenGL / body model setup. Your previous edit is unchanged."
        )
    return json.loads((destination / "edit.json").read_text(encoding="utf-8"))


def create_app(
    motion_path=None, body_model=None, output_root=None, renderer=None, source_info=None
):
    source = Path(
        motion_path or os.getenv("WALKEDIT_MOTION", ROOT / "outputs/walk-original.pt")
    ).resolve()
    model = Path(
        body_model or os.getenv("WALKEDIT_BODY_MODEL", ROOT / "data/SMPLX_NEUTRAL.npz")
    ).resolve()
    output = Path(output_root or ROOT / "outputs/chat").resolve()
    output.mkdir(parents=True, exist_ok=True)
    pool, lock, active = ThreadPoolExecutor(max_workers=1), RLock(), set()
    render = renderer or subprocess_render
    info = source_info
    setup_error = None
    if not source.is_file() or not model.is_file():
        setup_error = "Motion or SMPL-X body model missing. Follow the Chat workspace setup in the README."
    elif info is None:
        try:
            info = motion_info(source)
        except Exception as error:
            setup_error = "Could not read the trusted motion: " + str(error)
    source_key = None
    if setup_error is None:
        digest = hashlib.sha256()
        for asset in (source, model):
            with asset.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
        source_key = digest.hexdigest()

    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="WalkEdit Chat", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

    def session_path(session):
        if not re.fullmatch("[a-f0-9]{32}", session):
            raise HTTPException(404, "Session not found.")
        path = output / session
        if not (path / "session.json").is_file():
            raise HTTPException(404, "Session not found.")
        return path

    def read(session):
        data = json.loads(
            (session_path(session) / "session.json").read_text(encoding="utf-8")
        )
        if data.get("source_key") != source_key:
            raise HTTPException(
                409,
                "This session belongs to another motion or body model. Start a new session.",
            )
        if data["state"] == "rendering" and session not in active:
            data.update(
                state="failed",
                message="The server restarted during rendering. Send the command again.",
            )
            write_json(session_path(session) / "session.json", data)
        return data

    def public(data):
        current = data["versions"][data["cursor"]]
        return {
            key: data[key] for key in ("id", "state", "message", "messages")
        } | dict(
            plan=current["plan"],
            result=current.get("result"),
            revision=current.get("revision"),
            can_undo=data["cursor"] > 0,
            source=info,
        )

    def append(data, role, text):
        data["messages"].append(dict(role=role, text=text))

    def perform(session, revision, plan, undo_target):
        destination = session_path(session) / revision
        try:
            result = render(source, model, destination, plan.to_dict())
            with lock:
                data = read(session)
                version = dict(plan=plan.to_dict(), result=result, revision=revision)
                if undo_target is None:
                    data["versions"] = data["versions"][: data["cursor"] + 1] + [
                        version
                    ]
                    data["cursor"] += 1
                else:
                    data["versions"][undo_target] = version
                    data["cursor"] = undo_target
                data.update(state="ready", message="Render ready")
                append(
                    data,
                    "assistant",
                    f"Rendered {plan.describe()}. Edited motion: {result['edited_seconds']:.2f}s ({result['edited_frames']} frames at {result['output_fps']:g} FPS). The comparison holds the shorter motion's final pose.",
                )
                write_json(session_path(session) / "session.json", data)
        except Exception as error:
            with lock:
                data = read(session)
                data.update(state="failed", message=str(error))
                append(
                    data,
                    "assistant",
                    "The render did not finish. Your previous successful edit is unchanged. "
                    + str(error),
                )
                write_json(session_path(session) / "session.json", data)
        finally:
            with lock:
                active.discard(session)

    @app.get("/")
    def home():
        return FileResponse(ROOT / "web/index.html")

    @app.get("/api/health")
    def health():
        reference = (
            source == (ROOT / "outputs/walk-original.pt").resolve()
            and (ROOT / "docs/assets/smplx-comparison.mp4").is_file()
        )
        return dict(
            ready=setup_error is None,
            error=setup_error,
            source=info,
            parser="local English commands",
            source_name=source.name,
            reference_available=reference,
        )

    @app.get("/reference.mp4")
    def reference():
        if not health()["reference_available"]:
            raise HTTPException(404, "Reference preview not available for this source.")
        return FileResponse(ROOT / "docs/assets/smplx-comparison.mp4")

    @app.post("/api/sessions", status_code=201)
    def new_session():
        if setup_error:
            raise HTTPException(409, setup_error)
        session = uuid4().hex
        path = output / session
        path.mkdir()
        data = dict(
            id=session,
            source_key=source_key,
            state="ready",
            message="Ready to edit",
            cursor=0,
            versions=[dict(plan=EditPlan().to_dict())],
            messages=[dict(role="assistant", text=WELCOME)],
        )
        write_json(path / "session.json", data)
        return public(data)

    @app.get("/api/sessions/{session}")
    def status(session: str):
        with lock:
            return public(read(session))

    @app.post("/api/sessions/{session}/messages")
    def message(session: str, request: Message):
        with lock:
            data = read(session)
            if active:
                raise HTTPException(
                    409,
                    "A motion is rendering. Wait for it to finish before sending another edit.",
                )
            plan = EditPlan(**data["versions"][data["cursor"]]["plan"])
            append(data, "user", request.text)
            try:
                command = parse_command(request.text, plan, info["span"])
            except CommandError as error:
                append(data, "assistant", str(error))
                data.update(state="ready", message="Command needs clarification")
                write_json(session_path(session) / "session.json", data)
                return public(data)
            undo_target = None
            if command.action == "undo":
                if data["cursor"] == 0:
                    append(data, "assistant", "You are already at the original timing.")
                    write_json(session_path(session) / "session.json", data)
                    return public(data)
                undo_target = data["cursor"] - 1
                target = data["versions"][undo_target]
                if target.get("result"):
                    data.update(
                        cursor=undo_target,
                        state="ready",
                        message="Previous edit restored",
                    )
                    append(
                        data,
                        "assistant",
                        "Restored " + EditPlan(**target["plan"]).describe() + ".",
                    )
                    write_json(session_path(session) / "session.json", data)
                    return public(data)
                plan = EditPlan(**target["plan"])
            else:
                plan = command.plan
            revision = uuid4().hex
            (session_path(session) / revision).mkdir()
            data.update(
                state="rendering",
                message="Retiming SMPL-X and rendering a new comparison",
            )
            append(data, "assistant", "Rendering " + plan.describe() + ".")
            write_json(session_path(session) / "session.json", data)
            active.add(session)
            pool.submit(perform, session, revision, plan, undo_target)
            return public(data)

    @app.get("/api/sessions/{session}/renders/{revision}/{filename}")
    def artifact(session: str, revision: str, filename: str, download: bool = False):
        path = session_path(session)
        if not re.fullmatch("[a-f0-9]{32}", revision) or filename not in {
            "comparison.mp4",
            "edited.pt",
            "edit.json",
        }:
            raise HTTPException(404, "Artifact not found.")
        file = path / revision / filename
        if not file.is_file():
            raise HTTPException(404, "Artifact not ready.")
        return FileResponse(file, filename=filename if download else None)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the local WalkEdit chat workspace."
    )
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--body-model", type=Path)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(args.motion, args.body_model), host="127.0.0.1", port=args.port
    )
