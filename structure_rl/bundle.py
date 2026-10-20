"""Create a small, self-contained source snapshot for an isolated cloud folder."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('results/structure_rl/deploy/structure_training.zip'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = list((root / 'structure_rl').glob('*.py'))
    files += list((root / 'structure_rl/models').glob('*.xml'))
    files += list((root / 'miniwheelbot_rl').glob('*.py'))
    files += [root / 'structure_rl/README.md', root / 'requirements-mjlab.txt', root / 'tests/test_structure_rl.py',root / 'tests/test_steer_training.py']
    manifest = {'files': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(files)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.sha256').write_text(digest + '\n', encoding='utf-8')
    with zipfile.ZipFile(args.output) as archive:
        for name, checksum in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
    print(json.dumps({'bundle': str(args.output.resolve()), 'files': len(files),
                      'bytes': args.output.stat().st_size, 'sha256': digest}, indent=2))


if __name__ == '__main__':
    main()
