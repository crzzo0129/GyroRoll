"""Build a small, self-contained upload bundle without local runtimes or logs."""

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = (
    "CLOUD_README.md",
    "requirements-mjlab.txt",
    "package_cloud.py",
    "discuss/直接强化学习训练.md",
    "wheelbot/nested_model.xml",
    "tests/test_rl_velocity_model.py",
    "tests/test_rl_heading_rate.py",
)


def main():
    paths = sorted([ROOT / name for name in FILES] + list((ROOT / "miniwheelbot_rl").glob("*.py")))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    target = ROOT / "dist" / "miniwheelbot_mjlab130_cloud.zip"
    target.parent.mkdir(exist_ok=True)
    manifest = {"format": 1, "mjlab": "1.3.0", "trained_policy_included": False,
                "files": {path.relative_to(ROOT).as_posix():
                          hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    prefix = "miniwheelbot_mjlab130/"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, prefix + path.relative_to(ROOT).as_posix())
        bundle.writestr(prefix + "bundle_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(target) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Bundle CRC validation failed")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    print(f"{target}\n{target.stat().st_size:,} bytes\nSHA256 {digest}")


if __name__ == "__main__":
    main()
