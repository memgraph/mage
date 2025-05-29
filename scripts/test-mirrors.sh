#!/usr/bin/env bash
set -euo pipefail

LP_URL='https://launchpad.net/ubuntu/+archivemirrors'
TIMEOUT=0.2         # seconds per connect attempt
BEST_TIME=999999
BEST_URL=""

# Fetch HTML (requires curl or wget)
if command -v curl &>/dev/null; then
  HTML=$(curl -fsSL "$LP_URL")
else
  HTML=$(wget -qO- "$LP_URL")
fi

# Extract only http://*.archive.ubuntu.com URLs
mapfile -t MIRRORS < <(
  grep -oE 'http://[A-Za-z0-9.-]+\.archive\.ubuntu\.com(/[^\"]*)?' <<<"$HTML" \
    | sed 's#/$##' \
    | sort -u
)

for url in "${MIRRORS[@]}"; do
  host=${url#*://}
  host=${host%%/*}

  # Timestamp before attempting to open socket
  start_ms=$(date +%s%3N)

  # Try to open a TCP socket on FD 3
  if timeout $TIMEOUT bash -c "exec 3<>/dev/tcp/$host/80"; then
    elapsed=$(( $(date +%s%3N) - start_ms ))

    # Close FD 3
    exec 3<&-; exec 3>&-

    if (( elapsed < BEST_TIME )); then
      BEST_TIME=$elapsed
      BEST_URL=$url
    fi
  fi
done

if [[ -n $BEST_URL ]]; then
  echo "$BEST_URL"
else
  echo "Error: no mirror reachable within ${TIMEOUT}s" >&2
  exit 1
fi
