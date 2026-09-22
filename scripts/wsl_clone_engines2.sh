#!/usr/bin/env bash
# Check clone state; (re)start missing clones detached so they do not block.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1
mkdir -p .models/logs

state() {
  local d="$1"
  if [ -d "$d/.git" ]; then
    local n
    n=$(git -C "$d" log -1 --format='%h %cd' --date=short 2>/dev/null || echo '?')
    printf '%-18s OK   %s  %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$n"
    return 0
  fi
  printf '%-18s MISSING/INCOMPLETE\n' "$d"
  return 1
}

echo "=== state ==="
for d in ollama lmstudio-python lms; do state "$d"; done

echo
echo "=== relaunching missing ones in the background ==="
start() {
  local url="$1" dir="$2"
  if [ -d "$dir/.git" ]; then echo "[skip] $dir"; return 0; fi
  rm -f ".models/logs/$dir.clone.log"
  nohup git clone --depth 1 "$url" "$dir" > ".models/logs/$dir.clone.log" 2>&1 &
  echo "[bg] $dir pid=$!"
}
start git@github.com:lmstudio-ai/lmstudio-python.git lmstudio-python
start git@github.com:lmstudio-ai/lms.git lms

# ollama is the big one: clone it last so the small ones get through first
if [ ! -d ollama/.git ]; then
  # a partial clone cannot be reused in place; use a different directory name
  nohup git clone --depth 1 --filter=blob:none --no-checkout \
    git@github.com:ollama/ollama.git ollama-src > .models/logs/ollama.clone.log 2>&1 &
  echo "[bg] ollama-src pid=$! (blobless, no checkout: fastest metadata fetch)"
else
  echo "[skip] ollama"
fi

sleep 25
echo
echo "=== after 25s ==="
for d in ollama ollama-src lmstudio-python lms; do
  [ -e "$d" ] && printf '%-18s %s\n' "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo "--- logs ---"
tail -n 2 .models/logs/lmstudio-python.clone.log 2>/dev/null
tail -n 2 .models/logs/lms.clone.log 2>/dev/null
tail -n 2 .models/logs/ollama.clone.log 2>/dev/null
echo
df -h /mnt/d | tail -1
