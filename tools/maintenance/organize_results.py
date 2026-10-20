"""One-time, hash-verified migration of the pre-2026-10-20 results layout.

Run without --apply to inspect the moves. Never overwrites a destination.
Historical JSON/CSV/log/ZIP contents remain byte-identical; their recorded paths
are provenance, not live local links. Explicit path mappings are saved separately.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "_shared/organization_20261020.json"
SKIP = {".git", ".venv", ".venv-mjlab", ".reference", ".cache", "logs", "dist", "__pycache__"}
TEXT = {".py", ".md", ".html", ".ps1", ".cmd"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def family(name):
    if name.startswith(("wheelbot_nested", "nested_", "chamfer_", "local_rise")):
        return "wheelbot_cargo"
    if name.startswith("wheelbot"):
        return "wheelbot_legacy"
    if name.startswith("steerbot"):
        return "steerbot_legacy"
    return None


def plan():
    moves = {}
    for path in sorted(RESULTS.iterdir()):
        group = family(path.name)
        if group and path.name not in ("wheelbot_cargo", "wheelbot_legacy", "steerbot_legacy"):
            moves[path.relative_to(ROOT).as_posix()] = f"results/{group}/{path.name}"
    moves.update({
        "results/flywheel_coupling": "results/twin_flywheel/coupling",
        "results/structure_comparison/twin": "results/twin_flywheel/local_lqr",
        "results/structure_comparison/steer": "results/raked_steer/local_lqr",
        "results/structure_comparison": "results/comparisons/structure_concepts",
        "results/structure_rl/downloads/2026-10-19_compare01/raked_steer": "results/raked_steer/rl/2026-10-19_compare01",
        "results/structure_rl/downloads/2026-10-19_compare01/twin_sync": "results/twin_flywheel/rl/2026-10-19_compare01",
        "results/structure_rl/downloads/2026-10-19_compare01.zip": "results/comparisons/structure_rl/2026-10-19_compare01.zip",
        "results/structure_rl/downloads/steer_v2_final": "results/raked_steer/rl/steer_v2_final",
        "results/structure_rl/downloads/steer_v2_summary.zip": "results/raked_steer/rl/steer_v2_summary.zip",
        "results/structure_rl/build_final_summary.py": "tools/reporting/build_steer_summary.py",
        "results/structure_rl/deploy/launch_steer_v2.py": "tools/cloud/launch_steer_v2.py",
        "results/structure_rl/deploy": "results/_shared/deploy",
        "results/structure_rl/cloud_run.json": "results/_shared/cloud_runs/structure_rl.json",
        "results/structure_rl/cloud_run_steer_v2.json": "results/_shared/cloud_runs/steer_v2.json",
        "results/structure_rl/__pycache__": ".cache/organization_20261020/archived_results_pycache",
        "roll_linkage/mechanism.mp4": "results/roll_linkage/concept_v1/mechanism.mp4",
        "roll_linkage/preview.png": "results/roll_linkage/concept_v1/preview.png",
        "roll_linkage/validation.json": "results/roll_linkage/concept_v1/validation.json",
        "qzcli_local.py": "tools/cloud/qzcli_local.py",
        "qzcli_session.py": "tools/cloud/qzcli_session.py",
    })
    return {a: b for a, b in moves.items() if (ROOT / a).exists()}


def remap(path, moves):
    for old in sorted(moves, key=len, reverse=True):
        if path == old or path.startswith(old + "/"):
            return moves[old] + path[len(old):]
    return path


def rewrite_paths(text, moves):
    # Join literal pathlib segments so ordinary path replacement handles them.
    text = re.sub(r"""(['"])results\1\s*/\s*(['"])([^'"]+)\2""",
                  lambda m: repr("results/" + m[3]), text)
    # Replace all known results paths in one pass; no cascading replacements.
    aliases = {a: b for a, b in moves.items() if a.startswith("results/")}
    aliases.update({
        "results/nested_rl": "results/wheelbot_cargo/rl",
        "results/steer_v2_trajectories": "results/raked_steer/trajectories",
        "results/steer_v2_final": "results/raked_steer/rl/steer_v2_final",
    })
    pattern = re.compile("|".join(re.escape(s) for s in sorted(aliases, key=len, reverse=True)))
    def replace(match):
        end = match.end()
        if end < len(text) and (text[end].isalnum() or text[end] in "_-."):
            return match[0]
        return aliases[match[0]]
    return pattern.sub(replace, text)


def fix_links(text, old_file, new_file, moves):
    def link(match):
        value = match[2]
        parsed = urlsplit(value)
        if not parsed.path or parsed.scheme or value.startswith(("/", "#")):
            return match[0]
        target = (ROOT / old_file).parent / unquote(parsed.path)
        target = target.resolve()
        if not target.is_relative_to(ROOT):
            return match[0]
        old_target = target.relative_to(ROOT).as_posix()
        new_target = ROOT / remap(old_target, moves)
        if not new_target.exists():
            return match[0]
        relative = os.path.relpath(new_target, (ROOT / new_file).parent).replace("\\", "/")
        if parsed.query:
            relative += "?" + parsed.query
        if parsed.fragment:
            relative += "#" + parsed.fragment
        return match[1] + relative + match[3]
    return re.sub(r"""((?:href|src)=["'])([^"']+)(["'])""", link, text)


def source_files():
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            path = Path(directory) / name
            if path.suffix in TEXT and path != Path(__file__).resolve():
                yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if MANIFEST.exists():
        raise SystemExit("Migration already recorded; use its mapping to inspect the layout.")
    moves = plan()
    print(json.dumps(moves, indent=2))
    if not args.apply:
        return
    # Check every resolved source/destination before any filesystem mutation.
    for old, new in moves.items():
        src, dst = (ROOT / old).resolve(), (ROOT / new).resolve()
        if not src.is_relative_to(ROOT) or not dst.is_relative_to(ROOT):
            raise ValueError((src, dst))
        if dst.exists() or src.is_symlink():
            raise FileExistsError((src, dst))
    originals = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8-sig")
                 for p in source_files()}
    before = {}
    for old in moves:
        src = ROOT / old
        for p in ([src] if src.is_file() else src.rglob("*")):
            if p.is_file():
                before[p.relative_to(ROOT).as_posix()] = sha(p)
    # Preserve original source/report text outside the working tree for undo.
    backup = ROOT / ".cache/organization_20261020/text_before.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise FileExistsError(backup)
    backup.write_text(json.dumps(originals, ensure_ascii=False), encoding="utf-8")
    # Children first; parents then carry only their remaining shared content.
    for old in sorted(moves, key=len, reverse=True):
        src, dst = ROOT / old, ROOT / moves[old]
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    for old, expected in before.items():
        assert sha(ROOT / remap(old, moves)) == expected, old
    changed = []
    for old_file, text in originals.items():
        new_file = remap(old_file, moves)
        # Historical cloud launch scripts retain their remote experiment paths.
        if new_file == "tools/cloud/launch_steer_v2.py":
            continue
        rewritten = rewrite_paths(text, moves)
        if Path(old_file).suffix == ".html":
            # Resolve actual link targets using pre-move context, then update prose.
            rewritten = rewrite_paths(fix_links(text, old_file, new_file, moves), moves)
        if rewritten != text:
            (ROOT / new_file).write_text(rewritten, encoding="utf-8")
            changed.append(new_file)
    # Only remove known empty containers left behind by the split.
    for name in ("results/structure_rl/downloads/2026-10-19_compare01",
                 "results/structure_rl/downloads", "results/structure_rl"):
        path = (ROOT / name).resolve()
        assert path.is_relative_to(RESULTS.resolve())
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "date": "2026-10-20", "moves": moves, "files_before": before,
        "moved_files": len(before), "rewritten_text_files": changed,
        "original_data_policy": "JSON/CSV/log/ZIP content preserved; historical paths are provenance",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Moved and hash-verified {len(before)} files; updated {len(changed)} text files.")


if __name__ == "__main__":
    main()
