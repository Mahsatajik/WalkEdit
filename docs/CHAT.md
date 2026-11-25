# Conversational timing control

The chat controls the existing retimer; it is not a text-to-motion model. English commands are parsed into an `EditPlan(speed, start, end, transition)` or an undo action. No prompt is passed to a shell, Python evaluator, or language model. The renderer is invoked with an argument list and a validated JSON plan.

## Timing semantics

- Absolute `x` speeds are relative to the original trusted reconstruction.
- `20% slower` multiplies the current requested speed by 0.8; `50% faster` multiplies it by 1.5.
- Source intervals are expressed as `from 0.5 to 1.5 seconds`. They refer to the original source clock, even after earlier edits.
- A speed-only follow-up preserves the interval and transition. `whole motion` clears the interval and ramp. `Reset` restores 1x over the full source.
- Smooth transitions specify ramp duration in source seconds and must fit into half the edited interval. Automatic smoothing uses at most 0.25 seconds per edge.
- A requested speed is a multiplier, not meters per second. Absolute walking speed cannot be inferred reliably from an arbitrary monocular fit.

Requests with conflicting speeds, invalid boundaries, unsupported actions, unparsed numeric units, or missing speeds return clarification. The previous successful edit remains active. The command vocabulary is intentionally bounded; arbitrary free-form reasoning is not supported.

## Execution and persistence

`POST /api/sessions` creates an isolated conversation. `POST /api/sessions/{id}/messages` interprets a command and queues a render. Poll `GET /api/sessions/{id}` for completion. The response includes the active successful plan, frame counts, encoded duration, original source span, and generated artifact URLs through the render route.

Only one render runs at a time. The browser disables duplicate submissions, and the API also rejects them while busy. A command creates a unique revision directory. A successful render advances history; a failed render does not change the active plan or preview. Undo restores a prior successful revision, rendering the original timing if necessary. New edits after undo start from that restored plan. Previous files remain on disk.

Sources are trusted local `.pt` files loaded with `weights_only=True`. The body model path is configured at server startup. Sessions are bound to a digest of both assets, preventing follow-up commands from silently switching reconstructions when startup paths change. Browser commands cannot choose arbitrary file paths. API artifact names are allowlisted. Video range requests support scrubbing. This local server has no authentication or upload interface and must not be exposed as a public service.

## Verify changes

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

All 15 local tests passed. Core timing/command tests do not require PyTorch, body models, or OpenGL. Five workspace tests use an injected renderer and verify command-to-plan behavior, transactional failure handling, undo, isolated sessions, changed-source rejection, duplicate submissions, artifact access, and range requests. Those five tests skip when the optional web test dependencies are absent. The existing hosted timing workflow installs only the core requirements; use `requirements-test.txt` to run all workspace tests locally.

Real mesh rendering must be tested separately with licensed assets. The local sample is the existing KTH reconstruction described in `METHOD.md` and `THIRD_PARTY.md`. A 1.5x chat edit converted its 60 frames at 30 FPS into 40 frames (1.33 seconds); the 2.00-second original was retained. Output `.pt` parameters, time maps, and the rendered comparison should all be inspected. This is a timing/rendering check, not a 3D reconstruction accuracy benchmark.

Additional real browser checks: a 2x edit over source seconds 0.5-1.5 with 0.2-second ramps produced 47 frames (1.57 seconds). A relative 20% slowdown from 1.5x produced 1.2x and 50 frames (1.67 seconds). Undo restored the cached 1.5x render, and an unsupported jump request preserved the active edit. Exported translations matched interpolation on the saved source-time maps.
