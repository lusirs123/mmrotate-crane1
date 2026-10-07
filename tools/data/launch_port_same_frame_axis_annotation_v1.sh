#!/usr/bin/env bash
# Local GUI only. Annotation inputs and configuration are Git-synced.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKET="$PROJECT_ROOT/annotation_materials/port_independent_size_labels_v1_20261006_final"
INPUTS="$PACKET/size_readout_inputs_v1"
LABELING_PROGRAM="${PORT_AXIS_LABELING_BIN:-/opt/anaconda3/envs/x-anylabeling-cpu/bin/xanylabeling}"
if [[ ! -x "$LABELING_PROGRAM" ]]; then
  echo "X-AnyLabeling executable unavailable: $LABELING_PROGRAM" >&2
  exit 2
fi
if [[ ! -d "$INPUTS/axis_annotations" || ! -f "$INPUTS/.xanylabelingrc" ]]; then
  echo "Prepared Git-synced axis annotation directory/configuration missing" >&2
  exit 2
fi
exec "$LABELING_PROGRAM" \
  --filename "$PACKET/images" \
  --output "$INPUTS/axis_annotations" \
  --config "$INPUTS/.xanylabelingrc" \
  --work-dir "$INPUTS" \
  --labels axis --validatelabel exact --nodata --no-auto-update-check
