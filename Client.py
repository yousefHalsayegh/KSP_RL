import math
import shutil
import time
from pathlib import Path

import krpc  # type: ignore

from curriculum import SaveUnavailable


class KSPClient:
	def __init__(self, name="KSP Client", connection=None):
		self.conn = connection if connection is not None else krpc.connect(name=name)
		self.sc = self.conn.space_center
		self.telemetry = {}

		self._bind_vessel()

	def _bind_vessel(self):
		self._clear_stage_streams()
		self.vessel = self.sc.active_vessel
		self.control = self.vessel.control
		self.ap = self.vessel.auto_pilot

		self.surface_frame = self.vessel.surface_reference_frame
		self.flight = self.vessel.flight(self.vessel.orbit.body.reference_frame)
		self.root_part = self.vessel.parts.root
		self._initialize_telemetry()

	def _initialize_telemetry(self):
		self.attitude = self.vessel.flight(self.surface_frame)

		def stream_property(obj, name):
			stream = self.conn.add_stream(getattr, obj, name)
			stream.rate = 10
			return stream

		self.telemetry = {
			"ut": stream_property(self.sc, "ut"),
			"altitude": stream_property(self.flight, "mean_altitude"),
			"surface_altitude": stream_property(self.flight, "surface_altitude"),
			"vertical_speed": stream_property(self.flight, "vertical_speed"),
			"horizontal_speed": stream_property(self.flight, "horizontal_speed"),
			"surface_speed": stream_property(self.flight, "speed"),
			"dynamic_pressure": stream_property(self.flight, "dynamic_pressure"),
			"pitch": stream_property(self.attitude, "pitch"),
			"heading": stream_property(self.attitude, "heading"),
			"roll": stream_property(self.attitude, "roll"),
			"mass": stream_property(self.vessel, "mass"),
			"thrust": stream_property(self.vessel, "thrust"),
			"available_thrust": stream_property(self.vessel, "available_thrust"),
			"throttle": stream_property(self.control, "throttle"),
			"pitch_input": stream_property(self.control, "pitch"),
			"yaw_input": stream_property(self.control, "yaw"),
			"roll_input": stream_property(self.control, "roll"),
			"sas": stream_property(self.control, "sas"),
			"situation": stream_property(self.vessel, "situation"),
			"apoapsis": stream_property(self.vessel.orbit, "apoapsis_altitude"),
			"periapsis": stream_property(self.vessel.orbit, "periapsis_altitude"),
			"time_to_apoapsis": stream_property(self.vessel.orbit, "time_to_apoapsis"),
			"eccentricity": stream_property(self.vessel.orbit, "eccentricity"),
			"inclination": stream_property(self.vessel.orbit, "inclination"),
			"orbital_speed": stream_property(self.vessel.orbit, "speed"),
		}

		for field, resource in {"liquid_fuel": "LiquidFuel", "oxidizer": "Oxidizer", "solid_fuel": "SolidFuel", "electric_charge": "ElectricCharge", "monopropellant": "MonoPropellant"}.items():
			stream = self.conn.add_stream(self.vessel.resources.amount, resource)
			stream.rate = 10
			self.telemetry[field] = stream

		frame = self.vessel.orbit.body.non_rotating_reference_frame
		for name, method in {"position": self.vessel.position, "velocity": self.vessel.velocity, "rotation": self.vessel.rotation, "angular_velocity": self.vessel.angular_velocity}.items():
			stream = self.conn.add_stream(method, frame)
			stream.rate = 10
			self.telemetry[name] = stream

	def vessel_exists(self):
		for vessel in self.sc.vessels:
			try:
				if self.root_part in vessel.parts.all:
					return True

			except self.conn.krpc.ObjectDestroyedException:
				continue
			except RuntimeError as exc:
				if "The vessel has no parts" in str(exc):
					continue

				raise

		return False

	def prepare_autopilot(self):
		self.ap.engaged = False

		self.control.throttle = 0.0
		self.control.pitch = 0.0
		self.control.yaw = 0.0
		self.control.roll = 0.0
		self.control.sas = False

		self.ap.reference_frame = self.surface_frame
		self.ap.target_roll = float("nan")
		self.ap.target_direction = (1.0, 0.0, 0.0)
		self.ap.engaged = True

	def apply_action(self, throttle, pitch):
		self.control.throttle = max(0.0, min(1.0, throttle))

		self.ap.target_pitch_and_heading(max(0.0, min(90.0, pitch)), 90.0)

	def manual_control(self):
		self.ap.engaged = False
		self.control.sas = False

	def apply_controls(self, throttle, pitch, yaw, roll):

		self.control.throttle = max(0.0, min(1.0, throttle))
		self.control.pitch = max(-1.0, min(1.0, pitch))
		self.control.yaw = max(-1.0, min(1.0, yaw))
		self.control.roll = max(-1.0, min(1.0, roll))

	def activate_next_stage(self):
		before = self.control.current_stage
		if before <= 0:
			return None
		self._clear_stage_streams()
		self.control.activate_next_stage()
		after = self.control.current_stage
		return before - 1 if after < before else None

	def _clear_stage_streams(self):
		streams = getattr(self, "_stage_resource_streams", [])
		self._stage_resource_streams = []
		self._stage_cache_key = None
		self._stage_info = {}
		for stream, _ in streams:
			try:
				stream.remove()
			except (self.conn.krpc.ObjectDestroyedException, self.conn.krpc.StreamRemovedException):
				pass

	def read_next_stage(self, current_stage):
		for attempt in range(3):
			try:
				info = self._read_next_stage(current_stage)
				info["stage"] = current_stage
				return info
			except self.conn.krpc.ObjectDestroyedException:
				self._clear_stage_streams()
				if attempt == 2 or not self.vessel_exists():
					raise
				current_stage = self.control.current_stage

	def _read_next_stage(self, current_stage):
		if current_stage != self._stage_cache_key:
			self._clear_stage_streams()
			info = {
				name: 0.0
				for name in (
					"next_stage_available",
					"next_stage_engine_count",
					"next_stage_vacuum_thrust",
					"next_stage_decoupled_part_count",
					"next_stage_dry_mass",
					"next_stage_propellant_mass",
					"next_stage_parachute_count",
					"next_stage_launch_clamp_count",
				)
			}
			if current_stage > 0:
				next_stage = current_stage - 1
				parts = self.vessel.parts.all
				activation_parts = [part for part in parts if part.stage == next_stage]
				detached_parts = [part for part in parts if part.decouple_stage == next_stage]
				info["next_stage_available"] = 1.0
				for part in activation_parts:
					engine = part.engine
					if engine is not None:
						info["next_stage_engine_count"] += 1
						info["next_stage_vacuum_thrust"] += engine.max_vacuum_thrust * engine.thrust_limit
					info["next_stage_parachute_count"] += int(part.parachute is not None)
					info["next_stage_launch_clamp_count"] += int(part.launch_clamp is not None)
				info["next_stage_decoupled_part_count"] = len(detached_parts)
				info["next_stage_dry_mass"] = sum(part.dry_mass for part in detached_parts)
				for part in detached_parts:
					resources = part.resources
					names = set(resources.names)
					for name in ("LiquidFuel", "Oxidizer", "SolidFuel", "MonoPropellant"):
						if name in names:
							stream = self.conn.add_stream(resources.amount, name)
							stream.rate = 10
							self._stage_resource_streams.append((stream, self.sc.Resources.density(name)))
			self._stage_info = info
			self._stage_cache_key = current_stage
		info = self._stage_info.copy()
		info["next_stage_propellant_mass"] = sum(stream() * density for stream, density in self._stage_resource_streams)
		return info

	def read_observation(self):
		observation = {name: stream() for name, stream in self.telemetry.items()}
		observation["stage"] = self.control.current_stage
		observation.update(self.read_next_stage(observation["stage"]))
		observation["client_time"] = time.monotonic()
		return observation

	def step(self, throttle, pitch, yaw=0, roll=0, duration=0.2, manual=True):
		start_ut = self.sc.ut
		deadline = time.monotonic() + 5.0
		if manual:
			self.apply_controls(throttle, pitch, yaw, roll)
		else:
			self.apply_action(throttle, pitch)

		while self.sc.ut - start_ut < duration:
			if time.monotonic() >= deadline:
				raise TimeoutError("Game time stopped advancing.")
			time.sleep(0.01)

		observation = self.read_observation()
		end_ut = self.sc.ut
		observation["ut"] = end_ut
		observation["dt"] = end_ut - start_ut
		return observation

	def stop_controls(self):
		if not self.vessel_exists():
			return
		self.control.throttle = 0.0
		self.ap.engaged = False

		self.control.pitch = 0.0
		self.control.yaw = 0.0
		self.control.roll = 0.0

		self.control.sas = True
		self.control.sas_mode = self.sc.SASMode.stability_assist

	def revert_to_launch(self, delay=5.0):
		print("Reverting to launch...")

		if self.sc.can_revert_to_launch():
			self.sc.revert_to_launch()
		else:
			print("Revert to launch is unavailable.")

	def close(self):
		try:
			self._clear_stage_streams()
		finally:
			self.conn.close()

	def reset(self, delay=5.0, timeout=30.0):
		self.stop_controls()

		if not self.sc.can_revert_to_launch():
			raise RuntimeError("Revert to launch is unavailable.")

		self._clear_stage_streams()
		for stream in self.telemetry.values():
			stream.remove()

		self.telemetry.clear()

		self.revert_to_launch(delay=delay)

		deadline = time.monotonic() + timeout

		while time.monotonic() < deadline:
			try:
				vessel = self.sc.active_vessel
				ready = vessel is not None and vessel.situation == self.sc.VesselSituation.pre_launch
			except Exception:  # noqa: BLE001
				ready = False

			if ready:
				self._bind_vessel()
				return self.read_observation()

			time.sleep(0.1)

		raise TimeoutError("Launchpad flight did not become ready.")

	def capture_named_save(self, name, destination, *, save_dir, sampled_ut):
		can_save = getattr(self.sc, "can_save", None)
		if callable(can_save):
			can_save = can_save()
		if can_save is False:
			print(f"[capture test] can_save=False; trying named save {name!r} (throttle={self.control.throttle:.2f}, thrust={self.vessel.thrust:.0f} N)", flush=True)

		source = Path(save_dir) / f"{name}.sfs"
		destination = Path(destination)
		temporary = destination.with_suffix(".sfs.tmp")

		try:
			if abs(float(self.sc.ut) - sampled_ut) > 0.25:
				raise SaveUnavailable("Crossing telemetry is too old to capture.")

			self.sc.save(name)

			if not source.is_file() or source.stat().st_size == 0:
				raise RuntimeError(f"KSP did not create a usable save at {source}")
			if abs(float(self.sc.ut) - sampled_ut) > 0.25:
				raise SaveUnavailable("Game time moved too far during capture.")

			shutil.copyfile(source, temporary)
			temporary.replace(destination)
		finally:
			temporary.unlink(missing_ok=True)
			source.unlink(missing_ok=True)

	def load_named_save(self, name, *, expected=None, timeout=30.0):

		self._clear_stage_streams()
		for stream in self.telemetry.values():
			stream.remove()
		self.telemetry.clear()

		self.sc.load(name)
		deadline = time.monotonic() + timeout

		while time.monotonic() < deadline:
			try:
				vessel = self.sc.active_vessel
				if vessel is not None:
					if expected is None:
						ready = vessel.situation == self.sc.VesselSituation.pre_launch
					else:
						ready = (
							vessel.name == expected["vessel"]
							and vessel.orbit.body.name == expected["body"]
							and vessel.control.current_stage == expected["stage"]
							and abs(float(self.sc.ut) - expected["ut"]) <= 0.5
						)

					if ready:
						break
			except (RuntimeError, self.conn.krpc.ObjectDestroyedException):
				# Vessel handles can be temporarily invalid during the scene switch.
				pass

			time.sleep(0.05)
		else:
			raise TimeoutError(f"KSP did not load the expected flight from {name!r}")

		self._bind_vessel()
		self.manual_control()
		observation = self.read_observation()

		if expected is not None:
			if abs(observation["altitude"] - expected["altitude"]) > 50.0:
				raise RuntimeError(f"{name!r} loaded at an unexpected altitude")

			for resource, amount in expected["fuel"].items():
				if not math.isclose(observation[resource], amount, rel_tol=0.02, abs_tol=0.5):
					raise RuntimeError(f"{name!r} loaded with unexpected {resource}")

		return observation
