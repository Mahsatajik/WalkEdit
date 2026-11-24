import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def source_times(frames, source_fps, output_fps, speed, start=None, end=None, transition=0):
    """Map the output frame clock to source time, in seconds."""
    if frames < 2 or min(source_fps, output_fps, speed) <= 0:
        raise ValueError("Need at least two frames and positive frame rates and speed.")
    if not np.isfinite([source_fps, output_fps, speed]).all():
        raise ValueError("Frame rates and speed must be finite.")
    duration = (frames - 1) / source_fps
    start = 0.0 if start is None else start
    end = duration if end is None else end
    if not 0 <= start < end <= duration:
        raise ValueError(f"Expected 0 <= start < end <= {duration:.3f} seconds.")
    if not np.isfinite(transition) or transition < 0 or transition > (end - start) / 2:
        raise ValueError("Transition must be finite, nonnegative, and at most half the edited interval.")

    if transition:
        # Integrate dt_output = dt_source / speed(source) and invert the clock.
        source = np.linspace(0, duration, max(1001, int(duration * max(source_fps, output_fps) * 20) + 1))
        ramp = np.minimum((source - start) / transition, (end - source) / transition)
        ramp = np.clip(ramp, 0, 1)
        blend = ramp * ramp * ramp * (10 + ramp * (-15 + 6 * ramp))
        inverse_speed = 1 / (1 + (speed - 1) * blend)
        deltas = np.diff(source) * (inverse_speed[:-1] + inverse_speed[1:]) / 2
        clock = np.concatenate([[0], np.cumsum(deltas)])
        output = np.arange(int(np.floor(clock[-1] * output_fps + 1e-8)) + 1) / output_fps
        return np.interp(output, clock, source)

    segment_duration = (end - start) / speed
    output_duration = start + segment_duration + duration - end
    output = np.arange(int(np.floor(output_duration * output_fps + 1e-8)) + 1) / output_fps
    before = np.minimum(output, start)
    inside = np.clip(output - start, 0, segment_duration) * speed
    after = np.maximum(output - start - segment_duration, 0)
    return np.minimum(before + inside + after, duration)


def interpolate(values, clock, sample_times):
    columns = values.reshape(len(clock), -1)
    sampled = np.column_stack([np.interp(sample_times, clock, column) for column in columns.T])
    return sampled.reshape(len(sample_times), *values.shape[1:])


def retime_gvhmr(source, output, speed, start, end, transition=0):
    import torch

    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Use a different path for the edited motion.")
    saved = torch.load(source, map_location="cpu", weights_only=True)
    world = saved["smpl_params_global"]
    translation = world["transl"].numpy()
    source_fps = float(saved.get("fps", 30))
    output_fps = 30
    times = source_times(len(translation), source_fps, output_fps, speed, start, end, transition)
    clock = np.arange(len(translation)) / source_fps
    edited = {"transl": torch.as_tensor(interpolate(translation, clock, times), dtype=world["transl"].dtype)}

    for key in ("global_orient", "body_pose"):
        tensor = world[key]
        rotations = tensor.numpy().reshape(len(clock), -1, 3)
        sampled = [
            Slerp(clock, Rotation.from_rotvec(rotvecs))(times).as_rotvec()
            for rotvecs in rotations.transpose(1, 0, 2)
        ]
        values = np.stack(sampled, axis=1).reshape(len(times), *tensor.shape[1:])
        edited[key] = torch.as_tensor(values, dtype=tensor.dtype)

    betas = world["betas"]
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
    if len(betas) == 1:
        edited["betas"] = betas.expand(len(times), -1).clone()
    elif len(betas) == len(clock):
        edited["betas"] = torch.as_tensor(interpolate(betas.numpy(), clock, times), dtype=betas.dtype)
    else:
        raise ValueError("Expected one shape vector or one per input frame.")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"smpl_params_global": edited, "fps": output_fps}, output)
    print(f"{len(clock)} input frames -> {len(times)} output frames at {output_fps} fps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retime world-space SMPL-X motion.")
    parser.add_argument("source", type=Path, help="A .pt with smpl_params_global, such as hmr4d_results.pt")
    parser.add_argument("output", type=Path, help="Output .pt with edited world motion")
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--start", type=float, help="Start of edit in source seconds")
    parser.add_argument("--end", type=float, help="End of edit in source seconds")
    parser.add_argument("--transition", type=float, default=0, help="Smooth ramp length at each interval edge, in source seconds")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a different path for the edited motion.")
    retime_gvhmr(args.source, args.output, args.speed, args.start, args.end, args.transition)
