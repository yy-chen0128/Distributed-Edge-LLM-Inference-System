#!/usr/bin/env bash
# Clone third-party engines for source-level verification (read-only reference).
# They are NOT part of our repo: added to .gitignore, same convention as vllm/.
# ASCII ONLY.
set -u
cd "$HOME/pair" || exit 1

clone() {
  local url="$1" dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "[skip] $dir already cloned"
    return 0
  fi
  echo "[clone] $url -> $dir"
  git clone --depth 1 "$url" "$dir" 2>&1 | tail -3 || echo "[FAIL] $dir"
}

clone git@github.com:ollama/ollama.git ollama
clone git@github.com:lmstudio-ai/lmstudio-python.git lmstudio-python
clone git@github.com:lmstudio-ai/lms.git lms

echo
echo "=== sizes ==="
du -sh ollama lmstudio-python lms 2>/dev/null
echo
echo "=== disk ==="
df -h /mnt/d | tail -1
echo
echo "=== versions ==="
for d in ollama lmstudio-python lms; do
  if [ -d "$d/.git" ]; then
    printf '%-18s %s %s\n' "$d" "$(git -C "$d" log -1 --format=%cd --date=short)" \
      "$(git -C "$d" log -1 --format=%h)"
  fi
done
