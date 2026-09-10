#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT="$(cd "${1:-$(dirname "$0")/..}" && pwd)"
if [[ -f "$INPUT_ROOT/pyproject.toml" ]]; then
  SOURCE_ROOT="$INPUT_ROOT"
elif [[ -f "$INPUT_ROOT/src/pyproject.toml" ]]; then
  SOURCE_ROOT="$INPUT_ROOT/src"
else
  echo "CN Travel runtime pyproject.toml was not found under $INPUT_ROOT" >&2
  exit 2
fi
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/cn-travel-standalone.XXXXXX")"
trap 'rm -rf "$TEMP_ROOT"' EXIT

WHEEL_DIR="$TEMP_ROOT/wheel"
VENV_DIR="$TEMP_ROOT/venv"
mkdir -p "$WHEEL_DIR"
uv build --project "$SOURCE_ROOT" --wheel --out-dir "$WHEEL_DIR"

WHEEL_PATH="$(find "$WHEEL_DIR" -maxdepth 1 -name 'cn_travel-*.whl' -print -quit)"
test -n "$WHEEL_PATH"
uv venv --python 3.10 "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" "$WHEEL_PATH" httpx pytest

cd "$TEMP_ROOT"
env -u PYTHONPATH \
  "$VENV_DIR/bin/python" -c \
  'from cn_travel.deployment import deployment_status; assert deployment_status()["ready"]; print("standalone imports and deployment artifacts ok")'
env -u PYTHONPATH \
  "$VENV_DIR/bin/python" -m pytest -q \
  -c "$SOURCE_ROOT/tests/pytest.ini" \
  "$SOURCE_ROOT/tests/test_runtime_package.py" \
  "$SOURCE_ROOT/tests/test_agent_logic.py" \
  "$SOURCE_ROOT/tests/test_tools.py" \
  "$SOURCE_ROOT/tests/test_tools_v2.py" \
  -m "not live and not model and not policy and not reviews and not rag and not amap"
env -u PYTHONPATH "$VENV_DIR/bin/cn-travel" validate
