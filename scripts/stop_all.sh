#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
stopped=0
for f in "$ROOT"/logs/*.pid; do
  [[ -f "$f" ]] || continue
  pid="$(cat "$f")"
  name="$(basename "$f" .pid)"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping ${name} (pid $pid)"
    kill "$pid" 2>/dev/null || true
    stopped=1
  fi
  rm -f "$f"
done
if [[ "$stopped" -eq 0 ]]; then
  echo "No FTAAS processes to stop."
else
  echo "Done."
fi
