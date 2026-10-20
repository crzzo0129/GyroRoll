"""Run the existing local qz-cli installation with the working Python runtime."""
import os
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / '.venv' / 'Lib' / 'site-packages'))
    # Match the desktop MCP server's per-user state, even if shell HOME differs.
    os.environ['QZCLI_HOME'] = str(Path(os.environ['USERPROFILE']) / '.qzcli')
    runpy.run_module('qzcli.cli', run_name='__main__')
