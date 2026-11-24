"""Download the ACCAD sample and render a reproducible CPU-only comparison."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
URL = "https://accad.ohio-state.edu/sites/accad.osu.edu/files/quickwalk_BVH.ZIP"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--transition", type=float, default=0)
    parser.add_argument("--offline", action="store_true", help="Use a previously downloaded sample.")
    args = parser.parse_args()
    archive = ROOT / "data" / "accad-quickwalk.zip"
    if not archive.exists():
        if args.offline:
            parser.error("The sample is missing. Run once without --offline to download it.")
        print("Downloading Figure Quick Walk from ACCAD / The Ohio State University.", flush=True)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as response:
            content = response.read()
        from zipfile import ZipFile
        from io import BytesIO

        with ZipFile(BytesIO(content)) as sample:
            if sample.testzip() is not None:
                raise ValueError("The sample archive failed its integrity check.")
        archive.write_bytes(content)
    print(f"Sample SHA-256: {hashlib.sha256(archive.read_bytes()).hexdigest()}", flush=True)
    command = [
        sys.executable, str(ROOT / "preview_walk.py"), str(archive),
        str(ROOT / "outputs" / "demo.mp4"), "--clip", "2", "8",
        "--speed", str(args.speed), "--title", "QUICK WALK / ACCAD - THE OHIO STATE UNIVERSITY",
        "--transition", str(args.transition),
    ]
    if args.start is not None:
        command.extend(["--start", str(args.start)])
    if args.end is not None:
        command.extend(["--end", str(args.end)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
