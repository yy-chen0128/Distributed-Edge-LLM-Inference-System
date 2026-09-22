#!/usr/bin/env bash
# Retry the 7B download (now with a browser User-Agent) and compare cold vs warm
# reads on /mnt/d. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
mkdir -p .models/logs
PY="$HOME/venvs/pair/bin/python"

echo "=== [0] probe: is the 403 a User-Agent problem? ==="
echo "--- urllib default UA ---"
"$PY" - <<'EOF'
import urllib.request, urllib.error
url = "https://hf-mirror.com/api/models/Qwen/Qwen2.5-7B-Instruct/tree/main"
for label, hdrs in (("default", {}),
                    ("browser UA", {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})):
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read(200)
        print(f"  {label}: HTTP {r.status}, {len(body)} bytes")
    except Exception as exc:
        print(f"  {label}: {type(exc).__name__}: {exc}")
EOF

echo
echo "=== [1] restart 7B download ==="
if [ -f .models/logs/7b.pid ] && kill -0 "$(cat .models/logs/7b.pid)" 2>/dev/null; then
  echo "still running: pid $(cat .models/logs/7b.pid)"
else
  rm -f .models/logs/7b.pid
  nohup "$PY" scripts/hf_mirror_download.py \
    --repo Qwen/Qwen2.5-7B-Instruct \
    --dest .models/Qwen2.5-7B-Instruct \
    > .models/logs/7b_download.log 2>&1 &
  echo $! > .models/logs/7b.pid
  echo "started: pid $(cat .models/logs/7b.pid)"
fi
sleep 20
echo "--- log tail ---"
tail -c 600 .models/logs/7b_download.log 2>/dev/null || echo "(no log)"
echo
echo "--- downloaded so far ---"
du -sh .models/Qwen2.5-7B-Instruct 2>/dev/null || echo "(nothing yet)"

echo
echo "=== [2] /mnt/d cold vs warm read (0.5B safetensors, 384 MiB) ==="
echo "--- COLD (page cache dropped) ---"
"$PY" scripts/measure_fs_read.py .models/Qwen2.5-0.5B-Instruct/model.safetensors --mb 384 --rounds 3
echo "--- WARM (cache kept; back-to-back reads) ---"
"$PY" scripts/measure_fs_read.py .models/Qwen2.5-0.5B-Instruct/model.safetensors --mb 384 --rounds 4 --warm
