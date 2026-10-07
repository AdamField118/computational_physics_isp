#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${COMPHYS_PYTHON:-python3}" "$project_root/scripts/build.py" fem "$@"
