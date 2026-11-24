import argparse
from pathlib import Path
from zipfile import ZipFile

import imageio_ffmpeg
import numpy as np
from bvh import Bvh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

from retime import source_times


FPS = 30
WIDTH, HEIGHT = 960, 448


def load_clip(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".bvh")]
            if len(members) != 1:
                raise ValueError("Expected a ZIP archive with one BVH clip.")
            raw = archive.read(members[0]).decode("utf-8-sig")
    else:
        raw = path.read_text(encoding="utf-8-sig")
    clip = Bvh("\n".join(line for line in raw.splitlines() if line.strip()) + "\n")
    return clip, np.arange(clip.nframes) * clip.frame_time


def joint_positions(clip, clock, times):
    names = clip.get_joints_names()
    parents = [clip.joint_parent_index(name) for name in names]
    channels = np.asarray(clip.frames, dtype=float)
    positions = np.zeros((len(times), len(names), 3))
    world_rotations = []
    for index, (name, parent) in enumerate(zip(names, parents)):
        local_channels = clip.joint_channels(name)
        first = clip.get_joint_channels_index(name)
        local_position = np.tile(clip.joint_offset(name), (len(times), 1))
        for axis, label in enumerate("XYZ"):
            channel = label + "position"
            if channel in local_channels:
                local_position[:, axis] += np.interp(times, clock, channels[:, first + local_channels.index(channel)])

        rotation_channels = [i for i, channel in enumerate(local_channels) if channel.endswith("rotation")]
        order = "".join(local_channels[i][0].upper() for i in rotation_channels)
        if order:
            local = Rotation.from_euler(order, channels[:, first + np.array(rotation_channels)], degrees=True)
            local = Slerp(clock, local)(times)
        else:
            local = Rotation.identity(len(times))
        if parent < 0:
            positions[:, index] = local_position
            world_rotations.append(local)
        else:
            positions[:, index] = positions[:, parent] + world_rotations[parent].apply(local_position)
            world_rotations.append(world_rotations[parent] * local)
    return positions, parents


def render(output, original, edited, parents, speed, title, segment):
    # Use one camera and scale for both panels so distance and timing are comparable.
    ground = min(original[..., 1].min(), edited[..., 1].min())
    direction = original[-1, 0, [0, 2]] - original[0, 0, [0, 2]]
    heading = np.arctan2(direction[1], direction[0]) + np.deg2rad(20)
    right = np.array([np.cos(heading), 0, np.sin(heading)])
    up = np.array([-0.2 * np.sin(heading), 1, 0.2 * np.cos(heading)])
    projection = np.column_stack((right, up))
    views = [poses @ projection for poses in (original, edited)]
    all_points = np.concatenate(views).reshape(-1, 2)
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    scale = min(400 / max(1e-8, high[0] - low[0]), 250 / max(1e-8, high[1] - low[1]))

    def screen(projected, panel):
        xy = (projected - (low + high) / 2) * [scale, -scale]
        return xy + [240 + 480 * panel, 242]

    font = ImageFont.load_default(size=21)
    small = ImageFont.load_default(size=14)
    panels = [
        (f"Original  /  {len(original) / FPS:.2f} s", "#b8c9db", original, views[0]),
        (f"Retimed {speed:g}x  /  {len(edited) / FPS:.2f} s", "#6ce5bd", edited, views[1]),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output), (WIDTH, HEIGHT), fps=FPS, codec="libx264",
        macro_block_size=1, output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame_index in range(max(len(original), len(edited)) + FPS // 2):
            image = Image.new("RGB", (WIDTH, HEIGHT), "#0c1624")
            draw = ImageDraw.Draw(image)
            draw.text((24, 16), title, fill="#c7d5e4", font=small)
            for panel, (label, color, poses, projected) in enumerate(panels):
                offset = 480 * panel
                draw.rounded_rectangle((12 + offset, 45, 468 + offset, 428), radius=16, fill="#152337")
                draw.text((28 + offset, 61), label, fill="#f2f5fa", font=font)
                if panel == 1 and segment:
                    draw.text((28 + offset, 89), segment, fill="#c7d5e4", font=small)
                frame = min(frame_index, len(poses) - 1)
                points = screen(projected[frame], panel)
                trail = poses[:, 0].copy()
                trail[:, 1] = ground
                path = [tuple(point) for point in screen(trail @ projection, panel)]
                draw.line(path, fill="#496378", width=2)
                if frame > 0:
                    draw.line(path[:frame + 1], fill=color, width=3)
                for joint, parent in enumerate(parents):
                    if parent >= 0:
                        draw.line((tuple(points[parent]), tuple(points[joint])), fill=color, width=4)
                for x, y in points:
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#f2f5fa")
                draw.text((28 + offset, 398), f"{frame / FPS:.2f} s", fill=color, font=small)
            writer.send(image.tobytes())
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare original and retimed BVH motion in a fixed camera.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--transition", type=float, default=0, help="Smooth ramp length in source seconds")
    parser.add_argument("--clip", nargs=2, type=float, metavar=("FROM", "TO"), help="Crop the input BVH in seconds before retiming.")
    parser.add_argument("--title", default="3D WALK  /  BVH MOTION CAPTURE")
    args = parser.parse_args()
    clip, clock = load_clip(args.source)
    first, last = 0, len(clock) - 1
    if args.clip:
        if not 0 <= args.clip[0] < args.clip[1] <= clock[-1]:
            parser.error(f"Clip bounds must be ordered and within 0 to {clock[-1]:.3f} seconds.")
        first, last = np.rint(np.array(args.clip) / clip.frame_time).astype(int)
    frame_count = last - first + 1
    source_fps = 1 / clip.frame_time
    original_time = source_times(frame_count, source_fps, FPS, 1) + clock[first]
    edited_time = source_times(frame_count, source_fps, FPS, args.speed, args.start, args.end, args.transition) + clock[first]
    original, parents = joint_positions(clip, clock, original_time)
    edited, _ = joint_positions(clip, clock, edited_time)
    segment = None
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else clock[last] - clock[first]
        segment = f"Edit interval  {start:.2f} - {end:.2f} s"
    render(args.output, original, edited, parents, args.speed, args.title, segment)
    print(f"{len(original)} original frames -> {len(edited)} edited frames at {FPS} fps")
