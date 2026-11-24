# Method and motion format

## Reconstruction

MediaPipe extracts image and pseudo-3D landmarks from video. Missing detections are interpolated and observations are smoothed. SMPL-X body rotations, global orientation, shared shape, and image-plane root translation are optimized with Adam for 260 iterations. The loss combines confidence-weighted reprojection, pseudo-3D residuals, pose and shape priors, and temporal second differences.

The camera is weak perspective, with estimated scale and fixed root depth translation. Hands, face, and expression use model defaults. The output is a body-motion fit.

The metrics file reports weighted per-coordinate pixel RMSE and pseudo-3D RMSE against estimated landmarks. These are fitting residuals. Ground-truth 3D accuracy has not been measured.

## Stage interface

Each stage reads and writes files so reconstruction, editing, and rendering can run separately. The motion file is a PyTorch dictionary loaded with weights_only=True.

| Field | Shape | Meaning |
| --- | --- | --- |
| fps | scalar | Frame rate, default 30 for external files |
| smpl_params_global.global_orient | T × 3 | Axis-angle root rotation |
| smpl_params_global.body_pose | T × 63 | Axis-angle rotations of 21 body joints |
| smpl_params_global.transl | T × 3 | Root translation |
| smpl_params_global.betas | 10, 1 × 10, or T × 10 | Shape coefficients |

A new reconstruction backend can export this dictionary and reuse editing and rendering. The current renderer expects flat body rotations. Expressive pose fields and extra upstream metadata need an adapter. External GVHMR demo body parameters follow the same naming convention, with a default frame rate of 30.

## Editing

source_times defines a monotonic map from output timestamps to source timestamps. The retimer uses linear interpolation for translation and quaternion SLERP for each joint. It preserves the supplied shape trajectory under the new clock.

For constant speed s, a source interval [a, b] takes (b − a) / s output seconds. With --transition, a quintic smoothstep ramps from speed 1 to the requested speed and back inside the interval. The speed stays at the requested value between ramps. The reciprocal speed is integrated on a dense source-time grid, then inverted to obtain the output clock.

Smooth transitions have their own duration, which is generally longer than a hard speed switch. The speed argument specifies peak speed. Each ramp must fit within half the edited interval.

The BVH and SMPL-X paths share this time mapping. New temporal edits can be implemented here without modifying landmark fitting or mesh generation.

Frame timestamps span 0 to (N − 1) / FPS. Encoded duration is N / FPS. Duration is quantized to the output frame clock.

## Demonstration

The mesh demonstration uses KTH person01_walking_d2_uncomp.avi, cropped from 0.84 to 2.84 seconds. The original has 25 FPS and 160 × 120 resolution. The experiment resamples to 30 FPS and upscales to 640 × 480, which adds no source detail.

The BVH example uses ACCAD Figure Quick Walk, cropped from seconds 2–8. It is a separate experiment.

SMPL-X meshes are generated in batches. Both comparison panels share an orthographic camera derived from the original motion. The faster panel holds its final pose after finishing, and the comparison adds a 0.4-second tail.
