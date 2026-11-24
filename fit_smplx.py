import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import smplx
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform, batch_rodrigues


# Match SMPL-X joints to MediaPipe landmarks, with virtual pelvis, neck, and head.
SMPL_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]
LANDMARKS = [33, 23, 24, 25, 26, 27, 28, 31, 32, 34, 35, 11, 12, 13, 14, 15, 16]


def observe(video, landmarker_model):
    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_model)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    frames = imageio_ffmpeg.read_frames(str(video))
    metadata = next(frames)
    width, height = metadata["size"]
    fps = metadata["fps"]
    xyz, xy, confidence = [], [], []
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            for i, frame in enumerate(frames):
                rgb = np.frombuffer(frame, np.uint8).reshape(height, width, 3)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detected = detector.detect_for_video(image, round(1000 * i / fps))
                if detected.pose_landmarks:
                    screen, spatial = detected.pose_landmarks[0], detected.pose_world_landmarks[0]
                    xyz.append([(p.x, -p.y, -p.z) for p in spatial])
                    xy.append([(p.x * width, -p.y * height) for p in screen])
                    confidence.append([min(p.visibility, p.presence) for p in screen])
                else:
                    xyz.append(np.full((33, 3), np.nan))
                    xy.append(np.full((33, 2), np.nan))
                    confidence.append(np.zeros(33))
    finally:
        frames.close()

    if len(xyz) < 3:
        raise ValueError("The input video must contain at least three frames.")
    valid = np.isfinite(np.asarray(xyz)[:, 0, 0])
    if valid.sum() < max(3, len(valid) * 0.8):
        raise ValueError(f"Pose detected in only {valid.sum()} of {len(valid)} frames.")
    print(f"Pose detected in {valid.sum()}/{len(valid)} frames at {fps:g} fps.", flush=True)

    def prepare(points):
        points = np.asarray(points)
        index = np.arange(len(points))
        for marker in range(points.shape[1]):
            for axis in range(points.shape[2]):
                points[:, marker, axis] = np.interp(index, index[valid], points[valid, marker, axis])
        hip = points[:, [23, 24]].mean(axis=1)
        shoulder = points[:, [11, 12]].mean(axis=1)
        head = points[:, [7, 8]].mean(axis=1)
        pelvis = hip + 0.14 * (shoulder - hip)
        neck = shoulder + 0.15 * (head - shoulder)
        virtual_joints = np.stack([pelvis, neck, head], axis=1)
        return gaussian_filter1d(np.concatenate([points, virtual_joints], axis=1), 1.0, axis=0)

    weights = np.concatenate([confidence, np.ones((len(valid), 3)) * valid[:, None]], axis=1)
    return prepare(xyz), prepare(xy), weights[:, LANDMARKS], fps


def fit(model, spatial, screen, confidence):
    count = len(spatial)
    rest = (model.J_regressor @ model.v_template)[:22].detach()
    shape_offsets = torch.einsum("jv,vck->jck", model.J_regressor[:22], model.shapedirs[:, :, :10]).detach()
    hip_width = np.linalg.norm(np.asarray(rest[1] - rest[2]))
    shoulder_width = np.linalg.norm(np.asarray(rest[16] - rest[17]))
    hip_axis = (spatial[:, 23] - spatial[:, 24]) / hip_width
    shoulder_axis = (spatial[:, 11] - spatial[:, 12]) / shoulder_width
    right = hip_axis + 1.7 * shoulder_axis
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-5)
    up = spatial[:, 34] - spatial[:, 33]
    forward = np.cross(right, up)
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-5)
    up = np.cross(forward, right)
    initial = Rotation.from_matrix(np.stack([right, up, forward], axis=2)).as_rotvec()

    # Estimate a single weak-perspective camera scale for the fixed-camera clip.
    height_px = np.mean(screen[:, 35, 1:2] - screen[:, [27, 28], 1], axis=1)
    height_m = float(rest[15, 1] - rest[[7, 8], 1].mean())
    pixels_per_meter = np.quantile(height_px, 0.8) / height_m
    if pixels_per_meter <= 0:
        raise ValueError("Could not estimate the camera scale.")
    points3d = spatial[:, LANDMARKS] - spatial[:, 33:34]
    thigh = np.linalg.norm(spatial[:, [23, 24]] - spatial[:, [25, 26]], axis=2)
    shin = np.linalg.norm(spatial[:, [25, 26]] - spatial[:, [27, 28]], axis=2)
    rest_thigh = (rest[[1, 2]] - rest[[4, 5]]).norm(dim=1).mean()
    rest_shin = (rest[[4, 5]] - rest[[7, 8]]).norm(dim=1).mean()
    leg_m = float(rest_thigh + rest_shin)
    points3d *= leg_m / np.median(thigh + shin)
    target3d = torch.tensor(points3d, dtype=torch.float32)
    target2d = torch.tensor(screen[:, LANDMARKS] / pixels_per_meter, dtype=torch.float32)
    weights = torch.tensor(confidence, dtype=torch.float32)[..., None]
    weights[:, 3:9] *= 1.7

    orient = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32))
    pose = torch.nn.Parameter(torch.zeros(count, 21, 3))
    betas = torch.nn.Parameter(torch.zeros(10))
    translation_xy = torch.nn.Parameter(target2d[:, 0].clone())
    optimizer = torch.optim.Adam([orient, pose, betas, translation_xy], lr=0.035)
    for step in range(260):
        optimizer.zero_grad()
        rotations = batch_rodrigues(torch.cat([orient[:, None], pose], dim=1).reshape(-1, 3))
        shaped = rest + torch.einsum("jck,k->jc", shape_offsets, betas)
        joints, _ = batch_rigid_transform(
            rotations.reshape(count, 22, 3, 3),
            shaped.expand(count, -1, -1),
            model.parents[:22],
        )
        relative = joints[:, SMPL_JOINTS] - joints[:, :1]
        residual3d = relative - target3d
        residual2d = relative[:, :, :2] + translation_xy[:, None] - target2d
        loss = (weights * residual3d.square()).mean() + 2.2 * (weights * residual2d.square()).mean()
        loss = loss + 0.0015 * pose.square().mean() + 0.0007 * betas.square().mean()
        loss = loss + 0.065 * torch.diff(pose, n=2, dim=0).square().mean()
        loss = loss + 0.16 * torch.diff(orient, n=2, dim=0).square().mean()
        loss.backward()
        optimizer.step()
        if step == 169:
            optimizer.param_groups[0]["lr"] = 0.012

    with torch.no_grad():
        rotations = batch_rodrigues(torch.cat([orient[:, None], pose], dim=1).reshape(-1, 3))
        shaped = rest + torch.einsum("jck,k->jc", shape_offsets, betas)
        joints, _ = batch_rigid_transform(
            rotations.reshape(count, 22, 3, 3), shaped.expand(count, -1, -1), model.parents[:22]
        )
        relative = joints[:, SMPL_JOINTS] - joints[:, :1]
        error2d = relative[:, :, :2] + translation_xy[:, None] - target2d
        error3d = relative - target3d
        report = {
            "frames": count,
            "mean_landmark_confidence": float(np.mean(confidence)),
            "reprojection_rmse_px": float(
                ((weights * error2d.square()).sum() / (2 * weights.sum())).sqrt() * pixels_per_meter
            ),
            "pseudo_3d_rmse_m": float(
                ((weights * error3d.square()).sum() / (3 * weights.sum())).sqrt()
            ),
            "reference": "MediaPipe landmarks extracted from the input video",
            "interpretation": "Training residuals against estimated landmarks. Ground-truth 3D accuracy is unmeasured.",
            "camera": "Fixed weak-perspective camera with estimated scale and zero root depth translation",
        }
        shaped_root = rest[0] + shape_offsets[0] @ betas
        center = translation_xy - translation_xy[:1].mean(dim=0)
        translation = torch.nn.functional.pad(center, (0, 1)) - shaped_root
        print(f"Weighted fit objective: {loss.item():.5f}", flush=True)
        parameters = {
            "global_orient": orient.detach(),
            "body_pose": pose.detach().reshape(count, 63),
            "betas": betas.detach(),
            "transl": translation.detach(),
        }
        return parameters, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit SMPL-X body motion to a short fixed-camera video on CPU.")
    parser.add_argument("video", type=Path)
    parser.add_argument("landmarker_model", type=Path)
    parser.add_argument("body_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spatial, screen, confidence, fps = observe(args.video, args.landmarker_model)
    model = smplx.SMPLX(str(args.body_model), num_betas=10, flat_hand_mean=True)
    world, report = fit(model, spatial, screen, confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"smpl_params_global": world, "fps": fps}, args.output)
    report["fps"] = fps
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
