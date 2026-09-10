#!/usr/bin/env bash
# Read train_config.json, resolve paths from the application root, then launch train.py from this directory.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
APP_ROOT="$(cd "$DIR/../../.." && pwd)"
cd "$APP_ROOT"
export PYTHONPATH="$APP_ROOT:$APP_ROOT/src:$APP_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" - "$DIR" "$APP_ROOT" <<'PY'
import json, os, pathlib, sys
d = sys.argv[1]
app_root = pathlib.Path(sys.argv[2])
cfg = json.load(open(os.path.join(d, "train_config.json"), encoding="utf-8"))
flags = cfg.pop("flags", [])
for key in ("train_file", "model_name_or_path", "output_dir", "tools_file"):
    if cfg.get(key) and not pathlib.Path(cfg[key]).is_absolute():
        cfg[key] = str(app_root / cfg[key])
cmd = [sys.executable, os.path.join(d, "train.py")]
for k, v in cfg.items():
    cmd += [f"--{k}", str(v)]
cmd += [f"--{f}" for f in flags]
cmd += ["--log_file", os.path.join(cfg.get("output_dir", d), "train.log")]
print("launch:", " ".join(cmd[:4]), f"... ({len(cmd)} args)", flush=True)
os.execv(sys.executable, cmd)
PY
