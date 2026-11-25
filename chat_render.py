"""Trusted rendering worker. Chat text is never executed as code or a shell command."""

import argparse
import hashlib
import json
from pathlib import Path


def render_edit(source, body_model, destination, plan):
    import smplx
    import torch

    from chat_commands import EditPlan
    from render_smplx import load_vertices, render_comparison
    from retime import retime_gvhmr, source_times

    torch.set_num_threads(4)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plan = EditPlan(**plan)
    retime_gvhmr(
        Path(source),
        destination / "edited.pt",
        plan.speed,
        plan.start,
        plan.end,
        plan.transition,
    )
    model = smplx.SMPLX(str(body_model), num_betas=10, use_pca=False)
    original, fps = load_vertices(Path(source), model)
    edited, edited_fps = load_vertices(destination / "edited.pt", model)
    if fps != edited_fps:
        # The renderer compares at one output clock, so resample the reference too.
        retime_gvhmr(Path(source), destination / "reference.pt", 1, None, None)
        original, fps = load_vertices(destination / "reference.pt", model)
    label = f"Retimed / {plan.speed:g}x" + (
        " / interval" if plan.start is not None else ""
    )
    render_comparison(
        original,
        edited,
        fps,
        model.faces,
        destination / "comparison.mp4",
        plan.speed,
        edited_label=label,
    )
    saved = torch.load(source, map_location="cpu", weights_only=True)
    frames = len(saved["smpl_params_global"]["transl"])
    source_fps = float(saved.get("fps", 30))
    times = source_times(
        frames, source_fps, 30, plan.speed, plan.start, plan.end, plan.transition
    )
    result = dict(
        plan=plan.to_dict(),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        source_frames=frames,
        source_fps=source_fps,
        source_seconds=frames / source_fps,
        source_span=(frames - 1) / source_fps,
        edited_frames=len(edited),
        edited_seconds=len(edited) / fps,
        output_fps=fps,
        source_times=times.tolist(),
        note="The shorter motion holds its final pose while the longer motion finishes. Comparison includes a 0.4-second end hold.",
    )
    (destination / "edit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("body_model", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.destination / "request.json").read_text(encoding="utf-8"))
    render_edit(args.source, args.body_model, args.destination, plan)
