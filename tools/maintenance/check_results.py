"""Check migrated data hashes and local HTML links without running simulations."""
import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT / "results/_shared/organization_20261020.json").read_text(encoding="utf-8"))
    mappings = sorted(manifest["moves"].items(), key=lambda item: len(item[0]), reverse=True)
    missing, altered = [], []
    text_extensions = {".py", ".md", ".html"}
    unchanged = 0
    for old, expected in manifest["files_before"].items():
        current = old
        for src, dst in mappings:
            if old == src or old.startswith(src + "/"):
                current = dst + old[len(src):]
                break
        path = ROOT / current
        if not path.is_file():
            missing.append(current)
        elif path.suffix not in text_extensions:
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            if h != expected:
                altered.append(current)
            else:
                unchanged += 1
    links = 0
    broken = []
    for path in [*(ROOT / "results").rglob("*.html"),
                 ROOT / "wheelbot/report.html", ROOT / "steerbot/report.html"]:
        if not path.exists():
            continue
        for value in re.findall(r"""(?:href|src)=["']([^"']+)["']""", path.read_text(encoding="utf-8")):
            parsed = urlsplit(value)
            if parsed.scheme or not parsed.path or value.startswith("#"):
                continue
            links += 1
            if not (path.parent / unquote(parsed.path)).exists():
                broken.append([path.relative_to(ROOT).as_posix(), value])
    syntax_files = 0
    for folder in (".", "miniwheelbot_rl", "structure_rl", "roll_linkage", "optimize", "tests", "tools"):
        paths = ROOT.glob("*.py") if folder == "." else (ROOT / folder).rglob("*.py")
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            syntax_files += 1
    result = dict(original_files=len(manifest["files_before"]), unchanged_data_files=unchanged,
                  missing_files=missing, altered_data_files=altered,
                  checked_html_links=links, broken_html_links=broken,
                  python_syntax_files=syntax_files,
                  passed=not (missing or altered or broken))
    (ROOT / "results/_shared/layout_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
