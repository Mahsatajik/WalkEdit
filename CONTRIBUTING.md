# Contributing

Use English for code, comments, documentation, issues, and commit messages.

Install requirements.txt for the skeleton workflow or requirements-fit.txt for video fitting. Run the timing checks before submitting changes.

```bash
python -m unittest discover -s tests -v
```

Describe the input, exact command, intended behavior, and observed result. Include before/after media for visual changes. Record frame rates, camera assumptions, and coordinate conventions for new backends.

Add regression checks for timing and rotation fixes. Tests should run without body models or large downloads.

Keep local data in data/ and generated experiments in outputs/. Curated media under docs/assets/ need source attribution. Obtain third-party assets under their upstream terms.
