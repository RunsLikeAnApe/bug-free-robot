"""Launch the selection: the static selector first, then the run-based selector (selector.py).

The static selector (/screen/selector.py) prints, per commit, a set of tests that
holds every test the commit can break. The pipeline then runs candidates on the
patched code and keeps the ones that fail. Each answer is a superset of the
failing tests, so their intersection is one too: the pipeline keeps, for every
commit, only the tests that both select. When the static selector fails, takes
too long or leaves a commit out, the pipeline's own answer stands for it.
"""
import json
import os
import subprocess
import sys
import time

import selector as replay_selector

STATIC_SELECTOR_PATH = os.environ.get("BR_SCREEN", "/screen/selector.py")
STATIC_TIMEOUT_SECONDS = float(os.environ.get("BR_SCREEN_SECONDS", "60"))


def read_static_selections():
    if os.environ.get("BR_NO_SCREEN") == "1" or not os.path.exists(STATIC_SELECTOR_PATH):
        return None
    env = dict(os.environ)
    env.setdefault("SOURCE", os.environ.get("BR_ROOT", "/repo"))
    env.setdefault("DATA", os.environ.get("BR_DATA", "/data"))
    # The pipeline runs tests on the patched code itself, so the static selector's own
    # replays (unittest and TPC-DS) would only repeat that work: its static answer is enough.
    env.setdefault("SCREEN_REPLAY_BUDGET", "0")
    env.setdefault("SCREEN_TPCDS_BUDGET", "0")
    try:
        completed = subprocess.run([sys.executable, STATIC_SELECTOR_PATH], stdout=subprocess.PIPE,
                                   stdin=subprocess.DEVNULL, env=env, timeout=STATIC_TIMEOUT_SECONDS)
        output = completed.stdout
    except subprocess.TimeoutExpired as e:
        output = e.stdout or b""
        print("static selector timed out", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print("static selector failed:", repr(e), file=sys.stderr)
        return None
    selections = {}
    for line in output.decode(errors="replace").splitlines():
        try:
            row = json.loads(line)
            selections[row["commit"]] = frozenset(
                int(test_id) for worker in row["workers"] for test_id in worker
            )
        except (ValueError, KeyError, TypeError):
            continue  # a partial line (a timeout): that commit keeps the pipeline's answer
    return selections


if __name__ == "__main__":
    started_at = time.monotonic()
    static_selections = read_static_selections()
    replay_selector.RESTRICT = static_selections
    print("static answers:", None if static_selections is None else len(static_selections),
          "in", round(time.monotonic() - started_at, 2), "s", file=sys.stderr)
    replay_selector.WALL_LIMIT = max(30.0, replay_selector.WALL_LIMIT - (time.monotonic() - started_at))
    # a large batch gets up to WALL_MAX seconds (selector.wall_limit); the time that the static
    # selector took counts against it, so that both together stay inside the final's 4.5 minutes
    replay_selector.WALL_MAX = max(30.0, replay_selector.WALL_MAX - (time.monotonic() - started_at))
    replay_selector.main()
