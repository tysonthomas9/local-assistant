#!/usr/bin/env bash
# Start (or restart) the local SearXNG metasearch engine used by the robot's web_search tool.
# Listens on http://127.0.0.1:8888 only. SearXNG forwards queries to public engines (Google, Bing,
# DuckDuckGo, ...) from this machine; nothing is tied to an account.
#   local_backend/start_searxng.sh          start
#   docker stop reachy-searxng              stop
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${REACHY_SEARXNG_IMAGE:-searxng/searxng:latest}"
NAME=reachy-searxng

# Runtime config dir (git-ignored). The container chowns it to its own user (uid 977) on start,
# so the committed template lives outside it: local_backend/searxng.template.yml.
CONF="$HERE/searxng"
if [ ! -f "$CONF/settings.yml" ]; then
    mkdir -p "$CONF"
    sed "s/__SECRET__/$(openssl rand -hex 32)/" "$HERE/searxng.template.yml" > "$CONF/settings.yml"
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p 127.0.0.1:8888:8080 \
    -v "$CONF:/etc/searxng:rw" "$IMAGE" >/dev/null
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:8888/search?q=test&format=json" >/dev/null; then
        echo "SearXNG ready at http://127.0.0.1:8888 ($(docker inspect -f '{{.Config.Image}} {{.Image}}' "$NAME" | cut -c1-40))"
        exit 0
    fi
    sleep 1
done
echo "SearXNG did not answer; see: docker logs $NAME" >&2
exit 1
