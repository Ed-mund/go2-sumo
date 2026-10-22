#!/usr/bin/env python3
"""Read-only setup diagnostics; never imports Torch/Isaac or allocates GPU memory."""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import site
import subprocess
import sys
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[3]
EXPECTED = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "numpy": "1.26.4",
    "isaacsim": "4.5.0.0",
    "gymnasium": "1.2.3",
    "warp-lang": "1.5.0",
    "transformers": "4.41.2",
    "huggingface-hub": "0.23.4",
    "imageio": "2.37.3",
    "imageio-ffmpeg": "0.6.0",
    "pillow": "11.0.0",
}


def diagnose(cpu_only=False):
    errors = []
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": {}}
    if platform.machine() == "aarch64":
        # Isambard-AI is aarch64. Isaac Sim 4.5 publishes no aarch64 wheel;
        # the cluster jobs install Isaac Sim 5.1 / Python 3.11 / Torch 2.9 cu130.
        if sys.version_info[:2] != (3, 11):
            errors.append("Use Python 3.11 for the aarch64 Isaac Sim 5.1 stack")
    elif sys.version_info[:2] != (3, 10):
        errors.append("Use Python 3.10 for this pinned Isaac Sim 4.5 stack")
    if site.ENABLE_USER_SITE:
        errors.append("Set PYTHONNOUSERSITE=1 before launching Python")
    if not cpu_only and sys.platform != "linux":
        errors.append("The campaign uses Linux process groups/flock; native Windows is unsupported")
    pins = dict(EXPECTED)
    if platform.machine() == "aarch64":
        pins["torch"] = "2.9.0"
        pins["torchvision"] = "0.24.0"
        pins["isaacsim"] = "5.1.0.0"
    expected = {k: v for k, v in pins.items() if not cpu_only or k in ("torch", "numpy")}
    for name, version in expected.items():
        try:
            dist = metadata.distribution(name)
            report["packages"][name] = {"version": dist.version, "location": str(dist.locate_file(""))}
            found = dist.version.split("+")[0]
            isaacsim_ok = (
                platform.machine() == "aarch64" and name == "isaacsim" and found.startswith("5.1.0")
            )
            if found != version and not isaacsim_ok:
                errors.append(f"{name}: expected {version}, found {dist.version}")
        except metadata.PackageNotFoundError:
            errors.append(f"Missing {name}")
    if not cpu_only:
        for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
            try:
                dist = metadata.distribution(name)
                direct = json.loads(dist.read_text("direct_url.json") or "{}")
                actual = Path(unquote(urlparse(direct.get("url", "")).path)).resolve()
                target = REPO / "source" / name
                if not direct.get("dir_info", {}).get("editable") or actual != target:
                    errors.append(f"Reinstall {name} editable from this checkout: {target}")
            except metadata.PackageNotFoundError:
                errors.append(f"Missing editable {name}")
        gpu = shutil.which("nvidia-smi")
        if gpu:
            result = subprocess.run(
                [
                    gpu,
                    "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            report["gpu"] = result.stdout.strip()
            if result.returncode:
                errors.append("nvidia-smi failed; check the driver")
        else:
            errors.append("nvidia-smi is missing")
    report["free_disk_gib"] = round(shutil.disk_usage(REPO).free / 1024**3, 1)
    report["errors"] = errors
    report["passed"] = not errors
    report["scope"] = "metadata only: no simulator, CUDA, rendering, asset access or physics validation"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    report = diagnose(args.cpu_only)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)
