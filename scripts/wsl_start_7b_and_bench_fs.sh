#!/usr/bin/env bash
# Start the 7B download in the background (detached) and measure filesystem read
# throughput for /mnt/d (DrvFs) vs ext4. ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
mkdir -p .models/logs

PY="$HOME/venvs/pair/bin/python"
PIDF=".models/logs/7b.pid"

echo "=== [1] 7B download ==="
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "already running: pid $(cat "$PIDF")"
else
  nohup "$PY" scripts/hf_mirror_download.py \
    --repo Qwen/Qwen2.5-7B-Instruct \
    --dest .models/Qwen2.5-7B-Instruct \
    > .models/logs/7b_download.log 2>&1 &
  echo $! > "$PIDF"
  echo "started: pid $(cat "$PIDF")"
fi
sleep 8
echo "--- log ---"
tail -3 .models/logs/7b_download.log 2>/dev/null || echo "(no log yet)"
echo "--- free space on /mnt/d ---"
df -h /mnt/d | tail -1

echo
echo "=== [2] cold read throughput: /mnt/d (DrvFs) vs ext4 ==="
EXT4="$(find "$HOME/venvs/pair/lib" -name libtorch_cuda.so 2>/dev/null | head -1)"
echo "ext4 test file: $EXT4"
"$PY" scripts/measure_fs_read.py \
  .models/Qwen2.5-0.5B-Instruct/model.safetensors \
  "$EXT4" \
  --mb 384 --rounds 3

echo
echo "=== [3] does fadvise actually drop DrvFs cache? (read same file twice) ==="
"$PY" scripts/measure_fs_read.py .models/Qwen2.5-0.5B-Instruct/model.safetensors --mb 384 --rounds 2
