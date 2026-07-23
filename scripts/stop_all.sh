#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for f in "$ROOT"/logs/*.pid; do
  [[ -f "$f" ]] || continue
  pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $(basename "$f" .pid) (pid $pid)"
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
done
echo "Done."
