import argparse
from pathlib import Path

import imageio_ffmpeg
import moderngl
import numpy as np
import smplx
import torch
from PIL import Image, ImageDraw, ImageFont


PANEL = (600, 400)
CANVAS = (1232, 520)


def load_vertices(path, model):
    motion = torch.load(path, map_location="cpu", weights_only=True)
    world = motion["smpl_params_global"]
    frame_count = len(world["transl"])
    batches = []
    with torch.inference_mode():
        for start in range(0, frame_count, 32):
            batch = {key: world[key][start:start + 32] for key in ("transl", "global_orient", "body_pose")}
            n = len(batch["transl"])
            betas = world["betas"].reshape(-1, world["betas"].shape[-1])
            batch["betas"] = betas.expand(n, -1) if len(betas) == 1 else betas[start:start + n]
            for key, width in (("left_hand_pose", 45), ("right_hand_pose", 45), ("jaw_pose", 3),
                               ("leye_pose", 3), ("reye_pose", 3), ("expression", 10)):
                batch[key] = torch.zeros(n, width)
            batches.append(model(**batch).vertices.numpy())
    return np.concatenate(batches), float(motion.get("fps", 30))


def render_comparison(original, faster, fps, faces, output, speed):
    yaw, pitch = np.deg2rad([22, 12])
    camera_right = np.array([np.cos(yaw), 0, np.sin(yaw)])
    camera_flat = np.array([-np.sin(yaw), 0, np.cos(yaw)])
    vertical = np.array([0, 1, 0])
    camera_up = vertical * np.cos(pitch) - camera_flat * np.sin(pitch)
    camera_near = vertical * np.sin(pitch) + camera_flat * np.cos(pitch)
    basis = np.column_stack([camera_right, camera_up, camera_near])

    view = original @ basis
    low = view.min(axis=(0, 1))
    high = view.max(axis=(0, 1))
    center = (high + low) / 2
    scale = min((PANEL[0] - 88) / (high[0] - low[0]), (PANEL[1] - 84) / (high[1] - low[1]))
    clip_scale = np.array([2 * scale / PANEL[0], 2 * scale / PANEL[1], -0.2])
    floor_y = float(np.quantile(original[..., 1].min(axis=1), 0.2))

    def project(points):
        return (np.asarray(points) @ basis - center)[..., :2] * [scale, -scale] + np.asarray(PANEL) / 2

    floor = Image.new("RGBA", PANEL, "#132235")
    grid = ImageDraw.Draw(floor)
    x_low, x_high = original[..., 0].min() - 2.5, original[..., 0].max() + 2.5
    for x in np.arange(x_low, x_high, 0.35):
        grid.line([tuple(p) for p in project([[x, floor_y, -2], [x, floor_y, 2]])], fill="#263b50")
    for z in np.arange(-2, 2.1, 0.35):
        grid.line([tuple(p) for p in project([[x_low, floor_y, z], [x_high, floor_y, z]])], fill="#263b50")

    ctx = moderngl.create_standalone_context()
    ctx.enable(moderngl.DEPTH_TEST)
    program = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 position;
            in vec3 normal;
            out vec3 shading_normal;
            void main() {
                gl_Position = vec4(position, 1.0);
                shading_normal = normal;
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 shading_normal;
            uniform vec3 color;
            out vec4 rgba;
            void main() {
                vec3 n = normalize(shading_normal);
                float key = max(dot(n, normalize(vec3(-0.45, 0.8, 0.75))), 0.0);
                float fill = max(dot(n, normalize(vec3(0.8, 0.35, -0.4))), 0.0);
                rgba = vec4(color * (0.42 + 0.53 * key + 0.18 * fill), 1.0);
            }
        """,
    )
    buffer = ctx.buffer(reserve=original.shape[1] * 6 * 4)
    indices = ctx.buffer(np.asarray(faces, dtype=np.int32).tobytes())
    vao = ctx.vertex_array(program, [(buffer, "3f 3f", "position", "normal")], indices)
    target = ctx.simple_framebuffer(PANEL, components=4)
    target.use()

    def draw_mesh(vertices, color):
        surface = vertices @ basis
        triangle = surface[faces]
        face_normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        vert_normals = np.zeros_like(surface)
        for corner in faces.T:
            np.add.at(vert_normals, corner, face_normals)
        geometry = np.column_stack(((surface - center) * clip_scale, vert_normals)).astype("f4")
        buffer.write(geometry.tobytes())
        program["color"].value = color
        target.clear(0, 0, 0, 0, depth=1)
        vao.render()
        rgba = target.read(components=4, alignment=1)
        return Image.frombytes("RGBA", PANEL, rgba).transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output), CANVAS, fps=fps, codec="libx264", macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    panels = [
        (original, "Original / 1x", (0.76, 0.84, 0.97)),
        (faster, f"Retimed / {speed:g}x", (0.42, 0.94, 0.72)),
    ]
    try:
        for frame in range(max(len(original), len(faster)) + round(fps * 0.4)):
            image = Image.new("RGB", CANVAS, "#0c1522")
            draw = ImageDraw.Draw(image)
            draw.text((12, 12), "WALKEDIT  /  SMPL-X", font=small, fill="#b7c9dc")
            for panel, (animation, label, color) in enumerate(panels):
                x = 12 + panel * 608
                current = min(frame, len(animation) - 1)
                shot = Image.alpha_composite(floor, draw_mesh(animation[current], color))
                image.paste(shot, (x, 78))
                draw.text((x, 42), label, font=font, fill="#f0f5fd")
                draw.text((x + 510, 47), f"{len(animation) / fps:.2f} s", font=small, fill="#b7c9dc")
                draw.text((x, 489), f"{current / fps:.2f} s", font=small, fill="#b7c9dc")
            writer.send(image.tobytes())
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render two SMPL-X motions side by side with one fixed camera.")
    parser.add_argument("original", type=Path)
    parser.add_argument("retimed", type=Path)
    parser.add_argument("body_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speed", type=float, default=1.5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = smplx.SMPLX(str(args.body_model), num_betas=10, use_pca=False)
    original, fps = load_vertices(args.original, model)
    faster, faster_fps = load_vertices(args.retimed, model)
    if faster_fps != fps:
        parser.error("Resample both input motions to the same frame rate first.")
    render_comparison(original, faster, fps, model.faces, args.output, args.speed)
    print(f"Rendered {len(original)} original and {len(faster)} retimed frames at {fps:g} fps.")
