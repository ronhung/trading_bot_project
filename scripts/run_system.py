"""
Orchestration CLI -- Single command launcher for C++ engine + Python brain.

Features:
  - Eliminates the need to open two separate terminals.
  - Automatically manages process lifetimes, signal handling, and clean shutdown.
  - Supports both 'backtest' and 'live' modes with YAML config.
"""

import os
import sys
import time
import signal
import argparse
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

BUILD_DIR = os.path.join(_PROJECT_ROOT, "live_engine", "build_cmake")
BACKTEST_EXE = os.path.join(BUILD_DIR, "backtest_engine.exe")
LIVE_EXE = os.path.join(BUILD_DIR, "live_engine.exe")
PYTHON_BOT = os.path.join(_PROJECT_ROOT, "live_strategy", "live_trend_bot.py")


def run(config_path: str, mode: str = "backtest", csv_path: str | None = None):
    if not os.path.exists(config_path):
        print(f"[Error] Config not found: {config_path}")
        sys.exit(1)

    is_backtest = (mode.lower() == "backtest")
    target_exe = BACKTEST_EXE if is_backtest else LIVE_EXE

    if not os.path.exists(target_exe):
        print(f"[Error] C++ executable not found: {target_exe}")
        print("   Run .\\build.ps1 first to build the engines.")
        sys.exit(1)

    cpp_args = [target_exe]
    if is_backtest and csv_path:
        cpp_args.append(os.path.abspath(csv_path))

    py_args = [sys.executable, PYTHON_BOT, "--config", config_path]
    if is_backtest:
        py_args.append("--no-warmup")

    print("=" * 60)
    print(f"[System] Launching in [{mode.upper()}] mode")
    print(f"   Config: {config_path}")
    print(f"   C++ Engine: {target_exe}")
    print(f"   Python Brain: {PYTHON_BOT}")
    print("=" * 60)

    # 1. Start Python brain first so it connects to ZMQ
    print("[1/2] Starting Python brain...")
    py_proc = subprocess.Popen(py_args, cwd=_PROJECT_ROOT)
    time.sleep(1.0)

    # 2. Start C++ engine
    print("[2/2] Starting C++ engine...")
    cpp_proc = subprocess.Popen(cpp_args, cwd=BUILD_DIR)

    def cleanup(sig=None, frame=None):
        print("\n[System] Shutting down processes...")
        for p in [cpp_proc, py_proc]:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        if is_backtest:
            cpp_exit = cpp_proc.wait()
            print(f"\n[System] C++ Backtest engine finished with code {cpp_exit}")
            time.sleep(0.5)
            if py_proc.poll() is None:
                py_proc.terminate()
                py_proc.wait(timeout=2)
            sys.exit(cpp_exit)
        else:
            while True:
                if cpp_proc.poll() is not None:
                    print("[Warning] C++ engine stopped unexpectedly!")
                    break
                if py_proc.poll() is not None:
                    print("[Warning] Python brain stopped unexpectedly!")
                    break
                time.sleep(0.5)
            cleanup()
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Trading System Runner")
    parser.add_argument("--config", type=str, default="config/trend_breakout_highfreq.yaml",
                        help="Path to strategy YAML configuration")
    parser.add_argument("--mode", type=str, choices=["backtest", "live"], default="backtest",
                        help="Run mode (backtest or live)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Custom CSV path for backtest data replay")
    args = parser.parse_args()

    run(config_path=args.config, mode=args.mode, csv_path=args.csv)
