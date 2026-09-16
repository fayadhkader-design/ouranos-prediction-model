#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  ouranos_python=.venv/bin/python
elif [[ -x ../../work/venv/bin/python ]]; then
  ouranos_python=../../work/venv/bin/python
else
  python3 -m venv .venv
  ouranos_python=.venv/bin/python
  "$ouranos_python" -m pip install -r requirements.txt
fi
exec "$ouranos_python" -m streamlit run app.py --server.address 127.0.0.1 "$@"
