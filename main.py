import argparse
import ctypes
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

import krpc

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
			if worker["process"].poll() is not None:
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


def launch_games(args):
	executable = Path(args.ksp).expanduser().resolve()
	if not executable.is_file():
		raise FileNotFoundError(f"KSP executable not found: {executable}")

	workers = []

	for index in range(args.instances):
		command = [str(executable), "-screen-fullscreen", "0", "-screen-width", str(args.width), "-screen-height", str(args.height), f"--ksp-rl-server=RL-{index}"]
		process = subprocess.Popen(command, cwd=executable.parent)
		workers.append({"index": index, "process": process})
		print(f"Started worker {index}: PID {process.pid}")

	return workers


def wait_for_flights(workers, timeout=120.0):
	deadline = time.monotonic() + timeout
	pending = workers.copy()

	while pending:
		for worker in pending.copy():
			connection = worker["connection"]

			if worker["process"].poll() is not None:
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
	parser.add_argument("--width", type=int, default=640)
	parser.add_argument("--height", type=int, default=480)
	parser.add_argument("--startup-timeout", type=float, default=240.0)
	parser.add_argument("--base-rpc-port", type=int, default=50000)
	parser.add_argument("--connection-timeout", type=float, default=900.0)
	args = parser.parse_args()

	if min(args.instances, args.width, args.height, args.startup_timeout) <= 0:
		parser.error("Instance count, dimensions and timeout must be positive.")

	workers = launch_games(args)
	try:
		connect_games(workers, args)
		print("\n All games connected successfully.")
		wait_for_flights(workers, timeout=args.connection_timeout)
		position_windows(workers, args)
		input("Press Enter to disconnect Python. The games will stay open...")
	finally:
		for worker in workers:
			connection = worker.get("connection")
			if connection.krpc.paused:
				connection.krpc.paused = False
				print(f"Unpaused worker {worker['index']}")
			if connection.krpc.paused:
				raise RuntimeError(f"Worker {worker['index']} is still paused")


if __name__ == "__main__":
	main()
