import argparse
import ctypes
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from pathlib import Path

import krpc
import numpy as np
import torch

from agent import PPOAgent
from client import KSPClient
from environment import KSPEnvironment
from training import train

user32 = ctypes.WinDLL("user32", use_last_error=True)
WindowCallback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = [WindowCallback, wintypes.LPARAM]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]


def connect_games(workers, args):
	for worker in workers:
		index = worker["index"]
		rpc_port = args.base_rpc_port + 2 * index
		stream_port = rpc_port + 1
		deadline = time.monotonic() + args.connection_timeout
		last_error = None

		print(f"Waiting for worker {index}: RPC {rpc_port}, stream {stream_port}")

		while time.monotonic() < deadline:
			process = worker.get("process")
			if process is not None and process.poll() is not None:
				raise RuntimeError(f"Worker {index} exited before connecting.")

			remaining = deadline - time.monotonic()
			if remaining <= 0:
				break

			try:
				connection = krpc.connect(name=f"KSP RL Worker {index}", address="127.0.0.1", rpc_port=rpc_port, stream_port=stream_port, timeout=min(5.0, remaining))
			except OSError as error:
				last_error = error
				time.sleep(0.5)
				continue

			worker["connection"] = connection
			print(f"Connected to worker {index}")
			break
		else:
			pass

		if "connection" not in worker:
			raise TimeoutError(f"Worker {index} did not accept a connection on ports {rpc_port}/{stream_port}.") from last_error


def find_window(process_id):
	windows = []

	@WindowCallback
	def callback(hwnd, _):
		owner = wintypes.DWORD()
		user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
		if owner.value == process_id and user32.IsWindowVisible(hwnd):
			windows.append(hwnd)
		return True

	user32.EnumWindows(callback, 0)
	return windows[0] if windows else None


def position_windows(workers, args):
	spacing_x = args.width + 40
	spacing_y = args.height + 70
	columns = max(1, user32.GetSystemMetrics(0) // spacing_x)

	for worker in workers:
		index = worker["index"]
		hwnd = find_window(worker["process"].pid)
		if hwnd is None:
			raise RuntimeError(f"No visible window for worker {index}.")

		x = 10 + (index % columns) * spacing_x
		y = 10 + (index // columns) * spacing_y
		flags = 0x0004 | 0x0010

		if not user32.SetWindowPos(hwnd, None, x, y, args.width, args.height, flags):
			raise ctypes.WinError(ctypes.get_last_error())

		print(f"Positioned worker {index} at ({x}, {y})")


def launch_games(args, workers):
	executable = Path(args.ksp).expanduser().resolve()
	if not args.reuse_games and not executable.is_file():
		raise FileNotFoundError(f"KSP executable not found: {executable}")

	for index in range(args.instances):
		started = time.monotonic()
		worker = {"index": index, "process": None}
		if args.curriculum_dir is not None:
			if not args.ksp_save_name:
				raise ValueError("--ksp-save-name is required with --curriculum-dir")

			worker["save_dir"] = executable.parent / "saves" / args.ksp_save_name
			pad_save = worker["save_dir"] / "kerbal1_pad.sfs"

			if not pad_save.is_file():
				raise FileNotFoundError(f"Worker {index} is missing {pad_save}")
		workers.append(worker)

		if args.reuse_games:
			print(f"\nReconnecting to existing worker {index}...")
		else:
			rpc_port = args.base_rpc_port + 2 * index
			stream_port = rpc_port + 1
			command = [
				str(executable),
				"-screen-fullscreen",
				"0",
				"-screen-width",
				str(args.width),
				"-screen-height",
				str(args.height),
				f"--ksp-rl-server=RL-{index}",
				f"--ksp-rl-rpc-port={rpc_port}",
				f"--ksp-rl-stream-port={stream_port}",
				"-batchmode",
				"-nographics",
			]
			if args.curriculum_dir is not None:
				log_dir = args.curriculum_dir / "ksp_logs"
				log_dir.mkdir(parents=True, exist_ok=True)

				command += ["-logFile", str(log_dir / f"worker-{index:02d}.log")]
			print(f"\nLaunching worker {index}...")
			worker["process"] = subprocess.Popen(command, cwd=executable.parent)

		connect_games([worker], args)
		wait_for_flights([worker], timeout=args.connection_timeout)
		# if worker["process"] is not None:
		# 	position_windows([worker], args)

		worker["connection"].krpc.paused = True
		elapsed = time.monotonic() - started
		print(f"Worker {index} ready after {elapsed:.1f} seconds.")


def wait_for_flights(workers, timeout=120.0):
	deadline = time.monotonic() + timeout
	pending = workers.copy()

	while pending:
		for worker in pending.copy():
			connection = worker["connection"]

			process = worker.get("process")
			if process is not None and process.poll() is not None:
				raise RuntimeError(f"Worker {worker['index']} exited.")

			if connection.krpc.game_scene != connection.krpc.GameScene.flight:
				continue

			vessel = connection.space_center.active_vessel
			if vessel is None:
				continue

			pending.remove(worker)
			print(f"Worker {worker['index']} loaded: {vessel.name}")

		if pending:
			if time.monotonic() >= deadline:
				missing = [worker["index"] for worker in pending]
				raise TimeoutError(f"Workers did not reach flight: {missing}")
			time.sleep(0.5)

	# Unpause only after every worker has reached flight.
	for worker in workers:
		connection = worker["connection"]
		connection.krpc.paused = False

		if connection.krpc.paused:
			raise RuntimeError(f"Worker {worker['index']} is still paused.")

		print(f"Worker {worker['index']} unpaused")


def main():
	parser = argparse.ArgumentParser(description="Launch KSP training instances.")
	parser.add_argument("--ksp", required=True, help="Path to KSP_x64.exe")
	parser.add_argument("--instances", type=int, default=2)
	parser.add_argument("--width", type=int, default=1)
	parser.add_argument("--height", type=int, default=1)
	parser.add_argument("--startup-timeout", type=float, default=240.0)
	parser.add_argument("--base-rpc-port", type=int, default=50000)
	parser.add_argument("--connection-timeout", type=float, default=900.0)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--test-steps", type=int, default=100)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--updates", type=int, default=10)
	parser.add_argument("--rollout-steps", type=int, default=256)
	parser.add_argument("--episode-seconds", type=float, default=600.0)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--checkpoint", default="ppo_parallel.pt")
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--expand-observations", action="store_true")
	parser.add_argument("--wandb", action="store_true")
	parser.add_argument("--wandb-project", default="ksp-rl")
	parser.add_argument("--wandb-entity", default=None)
	parser.add_argument("--wandb-name", default=None)
	parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
	parser.add_argument("--wandb-run-id", default=None)
	parser.add_argument("--reuse-games", action="store_true")
	parser.add_argument("--curriculum-dir", type=Path, default=None, help="Enable the altitude curriculum and store its flight saves and logs here.")
	parser.add_argument("--ksp-save-name", help="Folder under KSP/saves containing the current quicksave.sfs.")
	args = parser.parse_args()

	if min(args.instances, args.width, args.height, args.startup_timeout) <= 0:
		parser.error("Instance count, dimensions and timeout must be positive.")

	workers = []
	if args.curriculum_dir is not None:
		repo_dir = Path(__file__).resolve().parent
		args.curriculum_dir = (repo_dir / args.curriculum_dir).resolve()

	try:
		launch_games(args, workers)

		train(args, workers)

	finally:
		for worker in workers:
			connection = worker.get("connection")
			if connection is not None:
				try:
					connection.krpc.paused = False
				finally:
					connection.close()


if __name__ == "__main__":
	main()
