#!/usr/bin/env python3
"""Reject binaries and generated artifacts in the Git index or all history."""
import argparse
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRS = {
    'target', 'build', 'dist', '__pycache__', '.cache', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', '.tox', '.nox', '.venv', 'venv', 'env',
    'node_modules', '.next', '.parcel-cache', 'CMakeFiles', '.ipynb_checkpoints',
}
ARTIFACT_SUFFIXES = {
    '.pyc', '.pyo', '.pyd', '.o', '.obj', '.mod', '.smod', '.so', '.dylib',
    '.dll', '.exe', '.a', '.lib', '.rlib', '.rmeta', '.out', '.err',
}
CACHE_NAMES = {'.DS_Store', 'Thumbs.db', 'CACHEDIR.TAG', 'CMakeCache.txt',
               '.coverage', '.rustc_info.json', 'meson-log.txt', 'helloworld'}


def generated(path):
    parts = PurePosixPath(path).parts
    return (any(p in CACHE_DIRS or p.endswith(('.egg-info', '.dist-info'))
                or p.startswith('.mesonpy-') for p in parts)
            or parts[-1] in CACHE_NAMES
            or PurePosixPath(path).suffix.lower() in ARTIFACT_SUFFIXES)


def binary(data):
    """Source and numerical tables must be UTF-8 text with ordinary whitespace."""
    if b'\0' in data:
        return True
    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        return True
    return any(c < 32 and c not in (9, 10, 12, 13) for c in data)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-history', action='store_true',
                        help='Scan every blob and path reachable from any local ref')
    args = parser.parse_args()
    objects = {}
    violations = set()
    if args.all_history:
        revisions = git('rev-list', '--all').decode().splitlines()
        entries = (git('ls-tree', '-rz', rev) for rev in revisions)
    else:
        entries = [git('ls-files', '--stage', '-z')]
    for listing in entries:
        for entry in listing.split(b'\0'):
            if not entry:
                continue
            meta, path_bytes = entry.split(b'\t', 1)
            fields = meta.split()
            oid = fields[2] if args.all_history else fields[1]
            path = path_bytes.decode('utf-8')
            if generated(path):
                violations.add('generated artifact: ' + path)
            if fields[0] == b'160000':
                violations.add('unscanned submodule: ' + path)
                continue
            objects.setdefault(oid, path)
    process = subprocess.Popen(['git', '-C', str(ROOT), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for oid, path in objects.items():
            process.stdin.write(oid + b'\n')
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[1] != b'blob':
                raise RuntimeError(f'Cannot read blob {oid!r}: {header!r}')
            data = process.stdout.read(int(header[2]))
            if process.stdout.read(1) != b'\n':
                raise RuntimeError('Truncated git cat-file output')
            if binary(data):
                violations.add('binary content: ' + path)
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    if violations:
        print('\n'.join(sorted(violations)), file=sys.stderr)
        return 1
    scope = 'all reachable history' if args.all_history else 'the index'
    print(f'Checked {len(objects)} distinct blobs in {scope}: no binaries or generated artifacts.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
