#!/usr/bin/env bash
# observability_up.sh — download and start Prometheus + Grafana against a running server's
# --metrics-port, provisioned with configs/observability/.
#
# This is a convenience for a test box, not a deployment tool: it fetches release tarballs into
# a scratch directory and binds everything to loopback. Reach it with an SSH tunnel:
#
#   ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 user@box
#
# Then http://127.0.0.1:3000 -> "Qwen3-TTS serving".
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OBS_DIR="${OBS_DIR:-$HOME/.qwen-obs}"
MPORT="${MPORT:-9109}"
PROM_PORT="${PROM_PORT:-9090}"
GRAF_PORT="${GRAF_PORT:-3000}"

case "$(uname -m)" in
    aarch64|arm64) GOARCH=arm64 ;;
    x86_64)        GOARCH=amd64 ;;
    *) echo "unsupported arch $(uname -m)"; exit 1 ;;
esac
mkdir -p "$OBS_DIR" && cd "$OBS_DIR" || exit 1

if [ ! -x "$OBS_DIR/prometheus/prometheus" ]; then
    PV=$(curl -s https://api.github.com/repos/prometheus/prometheus/releases/latest \
         | grep -m1 '"tag_name"' | sed 's/.*"v\([^"]*\)".*/\1/')
    [ -n "$PV" ] || { echo "cannot resolve the latest prometheus version"; exit 1; }
    echo "fetching prometheus $PV ($GOARCH)"
    curl -sL -o p.tgz "https://github.com/prometheus/prometheus/releases/download/v${PV}/prometheus-${PV}.linux-${GOARCH}.tar.gz" || exit 1
    tar xzf p.tgz && rm -f p.tgz && mv "prometheus-${PV}.linux-${GOARCH}" prometheus
fi

if [ ! -x "$OBS_DIR/grafana/bin/grafana" ] && [ ! -x "$OBS_DIR/grafana/bin/grafana-server" ]; then
    GV=$(curl -s https://api.github.com/repos/grafana/grafana/releases/latest \
         | grep -m1 '"tag_name"' | sed 's/.*"v\([^"]*\)".*/\1/')
    [ -n "$GV" ] || { echo "cannot resolve the latest grafana version"; exit 1; }
    echo "fetching grafana $GV ($GOARCH)"
    curl -sL -o g.tgz "https://dl.grafana.com/oss/release/grafana-${GV}.linux-${GOARCH}.tar.gz" || exit 1
    tar xzf g.tgz && rm -f g.tgz
    mv "grafana-v${GV}" grafana 2>/dev/null || mv "grafana-${GV}" grafana
fi

sed "s/127\.0\.0\.1:9109/127.0.0.1:${MPORT}/" \
    "$REPO/configs/observability/prometheus.yml" > "$OBS_DIR/prometheus.yml"

mkdir -p "$OBS_DIR/prov/datasources" "$OBS_DIR/prov/dashboards" "$OBS_DIR/dash"
cp "$REPO/configs/observability/grafana-dashboard.json" "$OBS_DIR/dash/"
cat > "$OBS_DIR/prov/datasources/ds.yml" <<YML
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:${PROM_PORT}
    isDefault: true
YML
cat > "$OBS_DIR/prov/dashboards/dash.yml" <<YML
apiVersion: 1
providers:
  - name: qwen
    folder: ''
    type: file
    options:
      path: ${OBS_DIR}/dash
YML

GBIN="$OBS_DIR/grafana/bin/grafana"
[ -x "$GBIN" ] || GBIN="$OBS_DIR/grafana/bin/grafana-server"

PROM_CMD="cd $OBS_DIR && ./prometheus/prometheus --config.file=$OBS_DIR/prometheus.yml \
--storage.tsdb.path=$OBS_DIR/tsdb --web.listen-address=127.0.0.1:${PROM_PORT}"
GRAF_CMD="cd $OBS_DIR/grafana && GF_PATHS_PROVISIONING=$OBS_DIR/prov \
GF_SERVER_HTTP_ADDR=127.0.0.1 GF_SERVER_HTTP_PORT=${GRAF_PORT} \
GF_AUTH_ANONYMOUS_ENABLED=true GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
$GBIN server --homepath $OBS_DIR/grafana"

if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t qwen-obs 2>/dev/null
    tmux new-session -d -s qwen-obs "$PROM_CMD > $OBS_DIR/prom.log 2>&1"
    tmux new-window  -t qwen-obs   "$GRAF_CMD > $OBS_DIR/grafana.log 2>&1"
    echo "started in tmux session 'qwen-obs'"
else
    nohup sh -c "$PROM_CMD" > "$OBS_DIR/prom.log" 2>&1 &
    nohup sh -c "$GRAF_CMD" > "$OBS_DIR/grafana.log" 2>&1 &
    echo "started with nohup (logs in $OBS_DIR)"
fi

sleep 20
printf 'prometheus: '; curl -s -m 5 "http://127.0.0.1:${PROM_PORT}/-/ready" || echo "not ready"
printf 'grafana:    '; curl -s -m 5 -o /dev/null -w 'http %{http_code}\n' "http://127.0.0.1:${GRAF_PORT}/login"
echo 'scrape target:'
curl -s -m 5 "http://127.0.0.1:${PROM_PORT}/api/v1/targets" | python3 -c "
import json,sys
try:
    for t in json.load(sys.stdin)['data']['activeTargets']:
        print('  ', t['scrapeUrl'], '->', t['health'], t.get('lastError',''))
except Exception as e:
    print('  cannot read targets:', e)
"
echo
echo "tunnel:  ssh -L ${GRAF_PORT}:127.0.0.1:${GRAF_PORT} -L ${PROM_PORT}:127.0.0.1:${PROM_PORT} user@box"
echo "then:    http://127.0.0.1:${GRAF_PORT}  ->  Qwen3-TTS serving"
