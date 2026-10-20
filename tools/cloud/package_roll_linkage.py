"""Bundle reproducible roll-linkage training code without environments or results."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[2]
    files = [root / name for name in (
        'requirements-mjlab.txt', 'result_paths.py',
        'tests/test_structure_rl.py', 'tests/test_steer_training.py', 'tests/test_linkage_training.py',
        'tools/cloud/render_structure_checkpoint.py',
        'tools/cloud/package_roll_linkage.py',
        'structure_rl/linkage_seed_gains.json',
    )]
    for directory in ('miniwheelbot_rl', 'structure_rl', 'roll_linkage', 'wheelbot'):
        files.extend(p for p in (root / directory).rglob('*')
                     if p.suffix in ('.py', '.xml', '.md') and '__pycache__' not in p.parts)
    files = sorted(set(files))
    manifest = {
        'morphology': 'roll_linkage', 'model_variant': 'level_torso_v2',
        'profiles': ['linkage_v2','linkage_v3','linkage_v4'], 'mjlab': '1.3.0',
        'files': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in files},
    }
    target = root / 'dist/roll_linkage_rl_cloud.zip'
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(root).as_posix())
        bundle.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(target) as bundle:
        assert bundle.testzip() is None
    print(json.dumps({'path': str(target), 'bytes': target.stat().st_size,
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
