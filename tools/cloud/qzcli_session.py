"""Keep one qz-cli Jupyter connection in memory for a long deployment session.

JSON-line commands: exec(command, timeout), attach(job_id, timeout),
upload(local, remote), close. Access tokens are never printed or saved.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / '.venv/Lib/site-packages'))
    os.environ['QZCLI_HOME'] = str(Path(os.environ['USERPROFILE']) / '.qzcli')
    os.environ['QZCLI_SESSION_ID'] = 'structure_20261019'
    from qzcli.cli import _find_notebook_jupyter_info, _exec_launch, _exec_poll
    import requests

    class QuietDisplay:
        def print(self, *args, **kwargs): pass
        def print_error(self, *args, **kwargs): pass
        def print_success(self, *args, **kwargs): pass
    display = QuietDisplay()
    info = _find_notebook_jupyter_info(args.target, display)
    if info is None:
        raise SystemExit('Connection unavailable; refresh qz-cli login first')
    print(json.dumps({'ready': True, 'notebook_id': info['notebook_id']}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request['action']
            if action == 'close': break
            if action == 'exec':
                # A final newline also keeps shell heredoc delimiters intact
                # when qz-cli wraps the command inside a redirected subshell.
                job = _exec_launch(info, request['command'].rstrip() + '\n', display)
                if job is None: raise RuntimeError('qz-cli could not launch command')
                if request.get('detach', False):
                    result = {'job_id': job, 'finished': False}
                else:
                    code, output, finished = _exec_poll(info, job, display, timeout=request.get('timeout', 20))
                    result = {'job_id': job, 'exit_code': code, 'output': output, 'finished': finished}
            elif action == 'attach':
                code, output, finished = _exec_poll(info, request['job_id'], display, timeout=request.get('timeout', 20))
                result = {'job_id': request['job_id'], 'exit_code': code, 'output': output, 'finished': finished}
            elif action == 'upload':
                local = (root / request['local']).resolve()
                if not local.is_relative_to(root): raise ValueError('Upload must come from this workspace')
                remote = request['remote']
                if not remote.startswith('_qzcli/') or '..' in remote.split('/'):
                    raise ValueError('Upload destination must be a qz-cli staging path')
                payload = local.read_bytes()
                response = requests.put(info['base_url'] + '/api/contents/' + remote,
                    headers={'authorization': 'token ' + info['token']},
                    json={'type': 'file', 'format': 'base64', 'content': base64.b64encode(payload).decode()}, timeout=30)
                if response.status_code not in (200, 201): raise RuntimeError(f'Upload HTTP {response.status_code}')
                result = {'uploaded': True, 'bytes': len(payload), 'remote': remote}
            else:
                raise ValueError('Unknown session operation')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as error:
            # Network exceptions can contain token-bearing URLs; don't echo them.
            print(json.dumps({'error': type(error).__name__, 'operation_failed': True}), flush=True)


if __name__ == '__main__':
    main()
