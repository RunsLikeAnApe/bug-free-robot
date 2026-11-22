#!/bin/sh
# Rebuilds coverage maps and unit maps from the traces.
set -e
CORES="${CORES:-0,1}"
WORK_DIR="${BR_WORK_DIR:-$PWD/work}"
SAMPLE_DIR="${BR_SAMPLE_DIR:-$PWD/sample}"
docker run --rm --cpuset-cpus "$CORES" --cpus 2 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$WORK_DIR/sqlglot:/repo:ro" -v "$WORK_DIR/rt:/rt:ro" -v "$WORK_DIR/trace:/trace:ro" \
  -v "$SAMPLE_DIR:/data:ro" -v "$WORK_DIR/maps:/out" -v "$WORK_DIR/deftime:/deftime:ro" \
  -v "$WORK_DIR/units2/raw:/units:ro" -v "$WORK_DIR/fullcov2/raw:/fullcov:ro" -w /repo br-label \
  python /rt/build_maps.py /repo /trace/agg.pkl /data/tests.json /out /deftime/defmap.pkl /units /fullcov 2>&1 | grep -v "nanobind\|leaked\|skipped" | tail -3
docker run --rm --cpuset-cpus "$CORES" --cpus 2 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$WORK_DIR/sqlglot:/repo:ro" -v "$WORK_DIR/rt:/rt:ro" -v "$WORK_DIR/units2/raw:/units:ro" \
  -v "$WORK_DIR/maps:/out" -v "$WORK_DIR/classes/raw:/classes:ro" -w /repo br-label \
  python /rt/build_unitmap.py /repo /units /out /classes 2>&1 | grep -v "nanobind\|leaked\|skipped" | tail -2
docker run --rm -v "$WORK_DIR/rt:/rt:ro" -v "$WORK_DIR/timing:/timing:ro" \
  -v "$SAMPLE_DIR:/data:ro" -v "$WORK_DIR/maps:/out" br-label \
  python /rt/build_costs.py /timing/full.json /data/tests.json /out
du -sh "$WORK_DIR/maps"
