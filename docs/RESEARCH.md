# Experiments and next steps

The current baseline reconstructs body motion from a single-person video and supports hard and smooth temporal edits. The code separates reconstruction, source-time mapping, resampling, and rendering through a documented motion format.

## Experiments to run

| Comparison | Variable | Measurement |
| --- | --- | --- |
| Hard retiming vs smooth retiming | Speed-transition profile | Boundary root acceleration, angular acceleration, achieved duration |
| Smooth retiming vs contact correction | Foot-contact constraints | Stance-foot displacement, ground penetration |
| Uniform vs confidence-weighted fitting | Observation weights | Held-out annotated 2D error and 3D error when available |
| CPU fitter vs external recovery backend | Reconstruction source | The same downstream edit and contact measurements |

Annotate stance intervals on source motions and transfer them through the time map. Use the same input motions, speed requests, and cameras across comparisons. Keep subjects disjoint when tuning or training learned components.

## Extension points

- Export the motion dictionary from a stronger reconstruction backend.
- Add a new source-time map while reusing rotation and translation resampling.
- Insert contact-aware inverse kinematics after editing and before mesh generation.
- Export the same motion into another renderer or animation tool.
- Translate a text request into validated speed, interval, and transition arguments.

Contact correction, ground-truth benchmark evaluation, and language control remain future work. Claims about reconstruction accuracy or research novelty require those experiments and comparisons with prior work.

## Related work

- [SMPL-X](https://smpl-x.is.tue.mpg.de/) supplies the expressive parametric body model.
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) supplies image and pseudo-3D observations.
- [GVHMR](https://github.com/zju3dv/GVHMR) provides world-grounded motion recovery.
- [Footskate Cleanup](https://pages.cs.wisc.edu/~kovar/footskateCleanup.pdf) motivates contact correction.
- [MotionFix](https://motionfix.is.tue.mpg.de/) studies text-driven motion editing.
