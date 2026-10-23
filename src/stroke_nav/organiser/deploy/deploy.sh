#!/usr/bin/env bash
# Deploy the stroke-nav bundle on the organiser's stack (HCMR-EV-OpenNG3 ./sys, HCMR-EV-RL).
#
#   ./deploy.sh install                       copy the scenario and run folders into HCMR-EV-RL
#   ./deploy.sh bc [EPISODES] [--visible]     behaviour-cloned SAC policy: their showcase, unchanged
#   ./deploy.sh astar-roll [EPISODES] [--visible]
#                                             A* route + calibrated entry roll: their showcase stack,
#                                             their evaluator with the controller in place of SAC.load
#   ./deploy.sh results                       success, steps and final distance of the latest evaluations
#   ./deploy.sh shutdown                      ./sys shutdown (also finalises the LeRobot rollout dataset)
#
# Paths: HCMR_OPENNG3_PATH (default: the current folder if it holds ./sys, else ../HCMR-EV-OpenNG3)
# and HCMR_RL_REPO_PATH (default as in ./sys: ../HCMR-EV-RL next to OpenNG3, else ~/HCMR-EV-RL).
# The values below are filled in by scripts/export_deployment.py.
set -euo pipefail

SCENARIO="@SCENARIO@"
BC_RUN="@BC_RUN@"
ROLL_RUN="@ROLL_RUN@"

HERE="$(cd "$(dirname "$0")" && pwd -P)"
if [[ -z "${HCMR_OPENNG3_PATH:-}" ]]; then
  if [[ -x ./sys ]]; then HCMR_OPENNG3_PATH="$PWD"; else HCMR_OPENNG3_PATH="$(cd "$HERE/.." && pwd -P)/HCMR-EV-OpenNG3"; fi
fi
[[ -x "$HCMR_OPENNG3_PATH/sys" ]] || { echo "ERROR: no ./sys in $HCMR_OPENNG3_PATH (set HCMR_OPENNG3_PATH)" >&2; exit 1; }
if [[ -z "${HCMR_RL_REPO_PATH:-}" ]]; then
  if [[ -d "$HCMR_OPENNG3_PATH/../HCMR-EV-RL" ]]; then HCMR_RL_REPO_PATH="$(cd "$HCMR_OPENNG3_PATH/../HCMR-EV-RL" && pwd -P)"
  else HCMR_RL_REPO_PATH="$HOME/HCMR-EV-RL"; fi
fi
export HCMR_RL_REPO_PATH
[[ -f "$HCMR_RL_REPO_PATH/compose.yaml" ]] || { echo "ERROR: HCMR-EV-RL not found at $HCMR_RL_REPO_PATH (set HCMR_RL_REPO_PATH)" >&2; exit 1; }
if docker compose version >/dev/null 2>&1; then COMPOSE=(docker compose); else COMPOSE=(docker-compose); fi
CONTAINER=hcmr-online-rl   # their evaluator container name (ONLINE_RL_CONTAINER in ./sys)

sys_() { (cd "$HCMR_OPENNG3_PATH" && ./sys "$@"); }

install() {
  mkdir -p "$HCMR_RL_REPO_PATH/scenarios" "$HCMR_RL_REPO_PATH/runs"
  rm -rf "$HCMR_RL_REPO_PATH/scenarios/$SCENARIO" "$HCMR_RL_REPO_PATH/runs/$BC_RUN" "$HCMR_RL_REPO_PATH/runs/$ROLL_RUN"
  cp -r "$HERE/HCMR-EV-RL/scenarios/$SCENARIO" "$HCMR_RL_REPO_PATH/scenarios/"
  cp -r "$HERE/HCMR-EV-RL/runs/$BC_RUN" "$HERE/HCMR-EV-RL/runs/$ROLL_RUN" "$HCMR_RL_REPO_PATH/runs/"
  echo "Installed scenario $SCENARIO and runs $BC_RUN, $ROLL_RUN into $HCMR_RL_REPO_PATH"
  if python3 -c 'import numpy' >/dev/null 2>&1; then
    "$HCMR_RL_REPO_PATH/scenario" validate "$SCENARIO" && echo "Scenario validated with their loader."
  else
    echo "(python3 with NumPy not on the host: ./sys startup validates the scenario in their RL image)"
  fi
}

parse_run_args() {  # EPISODES [--visible]
  EPISODES=5; VISIBLE=()
  for a in "$@"; do
    case "$a" in
      --visible) VISIBLE=(--visible) ;;
      ''|*[!0-9]*) echo "ERROR: unexpected argument $a" >&2; exit 2 ;;
      *) EPISODES="$a" ;;
    esac
  done
}

require_installed() {
  [[ -f "$HCMR_RL_REPO_PATH/runs/$1/resolved-config.json" && -f "$HCMR_RL_REPO_PATH/scenarios/$SCENARIO/scenario.json" ]] ||
    { echo "ERROR: run ./deploy.sh install first" >&2; exit 1; }
}

bc() {
  parse_run_args "$@"; require_installed "$BC_RUN"
  sys_ startup --mode showcase --scenario "$SCENARIO" --run-id "$BC_RUN" --model final-model.zip \
    --episodes "$EPISODES" "${VISIBLE[@]}"
  echo "Evaluating; follow with: docker logs -f $CONTAINER   then: ./deploy.sh results"
}

astar_roll() {
  parse_run_args "$@"; require_installed "$ROLL_RUN"
  # Their showcase brings up everything for this run (scenario renderer with the run's device
  # parameters, robot, bridge, reset coordinator, rollout recorder) and starts openng3-rl-evaluate,
  # which can only SAC.load a model. Stop that evaluator while it is still waiting for stream
  # readiness (it has not acted yet) and start their evaluator again with the controller.
  local rc=0
  sys_ startup --mode showcase --scenario "$SCENARIO" --run-id "$ROLL_RUN" --model controller.json \
    --episodes "$EPISODES" "${VISIBLE[@]}" || rc=$?
  if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "ERROR: ./sys startup failed before starting the evaluator (exit $rc)" >&2; exit 1
  fi
  echo 'Replacing the SAC evaluator with the entry-roll controller...'
  docker stop -s SIGINT -t 30 "$CONTAINER" >/dev/null 2>&1 || true
  docker rm "$CONTAINER" >/dev/null 2>&1 || true
  local rollout_id="${ROLL_RUN}-$(date -u +%Y%m%d-%H%M%S-%N)"
  (cd "$HCMR_RL_REPO_PATH" && HCMR_INTEGRATION_NETWORK="${HCMR_INTEGRATION_NETWORK:-hcmr-ev-integration}" \
    "${COMPOSE[@]}" run -d --no-deps --name "$CONTAINER" --user "$(id -u):$(id -g)" --entrypoint python3 openng3-rl \
      "/workspace/openng3_rl/runs/$ROLL_RUN/evaluate_controller.py" \
      --run-dir "/workspace/openng3_rl/runs/$ROLL_RUN" --model controller.json --episodes "$EPISODES" \
      --capture-rollouts --rollout-id "$rollout_id" \
      --scenario-root /workspace/openng3_rl/scenarios \
      --scenario-runtime /workspace/openng3_rl/runs/scenario-runtime.json >/dev/null)
  sleep 3
  docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true ||
    { echo "ERROR: controller evaluator exited; inspect: docker logs $CONTAINER" >&2; exit 1; }
  echo "Evaluating; follow with: docker logs -f $CONTAINER   then: ./deploy.sh results"
}

results() {
  local run latest dir
  for run in "$BC_RUN" "$ROLL_RUN"; do
    latest="$HCMR_RL_REPO_PATH/runs/$run/latest-evaluation.txt"
    [[ -f "$latest" ]] || { echo "$run: no evaluation yet"; continue; }
    dir="$HCMR_RL_REPO_PATH/runs/$run/evaluations/$(<"$latest")"
    echo "== $run ($(basename "$dir"))"
    python3 - "$dir" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
status = json.loads((d / "status.json").read_text())
print("state:", status.get("state"), "| summary:", status.get("summary") or status.get("error", ""))
eps = d / "episodes.json"
for e in (json.loads(eps.read_text())["episodes"] if eps.is_file() else []):
    dist = e.get("final_true_distance_m")
    dist = f"{dist * 1000:.1f} mm" if dist is not None else "n/a"
    print(f"  episode {e['episode_index']}: {e['outcome']:8s} steps {e['steps']:4d} ({e['steps'] / 5:.1f} s)  final distance {dist}")
PY
  done
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  install) install ;;
  bc) bc "$@" ;;
  astar-roll) astar_roll "$@" ;;
  results) results ;;
  shutdown) sys_ shutdown ;;
  *) sed -n '2,15p' "$0"; [[ "$cmd" == help ]] || exit 2 ;;
esac
