"""Run every script in the book and report pass/fail.

    uv run python scripts/validate_all.py            # everything (~10-20 min, ~$1-2)
    uv run python scripts/validate_all.py ch08 ch10  # only some chapters

Runs in memory-mode Qdrant by default so scripts never fight over the embedded on-disk store;
set QDRANT_MODE=local (or QDRANT_URL=...) in the environment to test the other modes.
Scripts that accept `--limit` get `--limit 8` to keep eval loops short; rag_cli.py gets a
question; the Chapter 17 server (app.py) is started, exercised with client_demo.py, then stopped.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
ENV.setdefault("QDRANT_MODE", "memory")   # QDRANT_MODE=local uv run python scripts/validate_all.py -> on-disk mode
TIMEOUT = 900

only = set(sys.argv[1:])
scripts = sorted(p for p in ROOT.glob("code/ch*/*.py") if p.name != "__init__.py")
if only:
    scripts = [p for p in scripts if p.parent.name in only]

results: list[tuple[str, int, float, str]] = []


def run(cmd: list[str], label: str) -> None:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=TIMEOUT)
        code, out = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        code, out = 124, (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else "timeout"
    tail = "\n".join(out.strip().splitlines()[-3:])
    if code in (134, -6) and "recursive_mutex lock failed" in out:   # 134 via shell, -6 (SIGABRT) via subprocess
        # Known macOS shutdown race in a native library (see SETUP.md troubleshooting):
        # the script finished and printed everything, then crashed while exiting.
        code = 0
        label += "   [finished; native crash at exit, see SETUP.md]"
    results.append((label, code, time.perf_counter() - t0, tail))
    print(f"{'PASS' if code == 0 else 'FAIL'}  {label}  ({results[-1][2]:.0f}s)")
    if code != 0:
        print("\n".join("      " + l for l in out.strip().splitlines()[-15:]))


for script in scripts:
    rel = str(script.relative_to(ROOT))
    src = script.read_text()
    if script.name == "app.py":                       # server: start, hit it, stop
        server = subprocess.Popen([sys.executable, rel], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):                           # poll /health instead of a fixed sleep
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2); break
            except Exception:
                time.sleep(1)
        run([sys.executable, "code/ch17/client_demo.py"], "code/ch17/app.py + client_demo.py")
        server.terminate(); server.wait(timeout=30)
        continue
    if script.name == "client_demo.py":
        continue                                      # covered above
    args = [sys.executable, rel]
    if "--limit" in src:
        args += ["--limit", "8"]
    if script.name == "rag_cli.py":
        args += ["How many PTO days do I get?"]
    if script.name == "diagnose.py":          # Chapter 27 needs a question to diagnose
        args += ["--golden", "q17"]
    if script.name == "triage.py":            # Chapter 26 takes a question positionally
        args += ["--golden", "q17"] if "--golden" in src else ["Which database does Beacon use?"]
    run(args, rel)

print()
passed = sum(1 for r in results if r[1] == 0)
print(f"passed {passed}/{len(results)}   total {sum(r[2] for r in results)/60:.1f} min")
for label, code, _, tail in results:
    if code != 0:
        print(f"  FAIL {label} (exit {code})")
sys.exit(0 if passed == len(results) else 1)
