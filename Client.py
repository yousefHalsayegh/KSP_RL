import math
import shutil
import time
from pathlib import Path

import krpc  # type: ignore

from curriculum import SaveUnavailable


class CheckpointMismatch(RuntimeError):
	"""The live vessel disagrees with the recorded checkpoint."""


class KSPClient:
	PROPELLANT_NAMES = {"liquid_fuel": "LiquidFuel", "oxidizer": "Oxidizer", "solid_fuel": "SolidFuel", "monopropellant": "MonoPropellant"}

	def __init__(self, name="KSP Client", connection=None, *, recovery_save="kerbal1_pad"):
		self.conn = connection if connection is not None else krpc.connect(name=name)
		self.sc = self.conn.space_center
		self.telemetry = {}

		try:
			self._bind_vessel()
		except (RuntimeError, self.conn.krpc.ObjectDestroyedException) as error:
			if not self._vessel_unavailable(error):
				raise
			previous_pause = self.conn.krpc.paused
			try:
				self.conn.krpc.paused = False
				if recovery_save is not None:
					print(f"[ksp] Recovering the active vessel from {recovery_save!r}", flush=True)
					self.sc.load(recovery_save)
				self._wait_and_bind(require_pad=recovery_save is not None, deadline=time.monotonic() + 45.0, label=recovery_save or "active flight")
			finally:
				self.conn.krpc.paused = previous_pause

	def _vessel_unavailable(self, error):
		return isinstance(error, self.conn.krpc.ObjectDestroyedException) or "The vessel has no parts" in str(error) or "No active vessel yet." in str(error)

	def _bind_vessel(self, vessel=None):
		if vessel is None:
			vessel = self.sc.active_vessel
		if vessel is None:
			raise RuntimeError("No active vessel yet.")

		root = vessel.parts.root
		if root is None:
			raise RuntimeError("The vessel has no parts.")

		self._clear_telemetry()
		self.vessel = vessel
		self.root_part = root
		try:
			self.control = vessel.control
			self.ap = vessel.auto_pilot
			self.surface_frame = vessel.surface_reference_frame
			self.flight = vessel.flight(vessel.orbit.body.reference_frame)
			self._initialize_telemetry()
		except Exception:
			self._clear_telemetry()
			raise

	def _wait_and_bind(self, *, require_pad, deadline, label):
		last_error = None
		while time.monotonic() < deadline:
			try:
				vessel = self.sc.active_vessel
				if vessel is not None and (not require_pad or vessel.situation == self.sc.VesselSituation.pre_launch):
					self._bind_vessel(vessel)
					return
			except (RuntimeError, self.conn.krpc.ObjectDestroyedException) as error:
				if not self._vessel_unavailable(error):
					raise
				last_error = error
			time.sleep(0.05)
		raise TimeoutError(f"KSP did not expose a usable vessel for {label!r}") from last_error

	def _initialize_telemetry(self):
		self.attitude = self.vessel.flight(self.surface_frame)

		def stream_property(field, obj, name):
			stream = self.conn.add_stream(getattr, obj, name)
			self.telemetry[field] = stream
			stream.rate = 10

		properties = {
			"ut": (self.sc, "ut"),
			"altitude": (self.flight, "mean_altitude"),
			"surface_altitude": (self.flight, "surface_altitude"),
			"vertical_speed": (self.flight, "vertical_speed"),
			"horizontal_speed": (self.flight, "horizontal_speed"),
			"surface_speed": (self.flight, "speed"),
			"dynamic_pressure": (self.flight, "dynamic_pressure"),
			"pitch": (self.attitude, "pitch"),
			"heading": (self.attitude, "heading"),
			"roll": (self.attitude, "roll"),
			"mass": (self.vessel, "mass"),
			"thrust": (self.vessel, "thrust"),
			"available_thrust": (self.vessel, "available_thrust"),
			"throttle": (self.control, "throttle"),
			"pitch_input": (self.control, "pitch"),
			"yaw_input": (self.control, "yaw"),
			"roll_input": (self.control, "roll"),
			"sas": (self.control, "sas"),
			"situation": (self.vessel, "situation"),
			"apoapsis": (self.vessel.orbit, "apoapsis_altitude"),
			"periapsis": (self.vessel.orbit, "periapsis_altitude"),
			"time_to_apoapsis": (self.vessel.orbit, "time_to_apoapsis"),
			"eccentricity": (self.vessel.orbit, "eccentricity"),
			"inclination": (self.vessel.orbit, "inclination"),
			"orbital_speed": (self.vessel.orbit, "speed"),
		}

		for field, (obj, name) in properties.items():
			stream_property(field, obj, name)

		for field, resource in {"liquid_fuel": "LiquidFuel", "oxidizer": "Oxidizer", "solid_fuel": "SolidFuel", "electric_charge": "ElectricCharge", "monopropellant": "MonoPropellant"}.items():
			stream = self.conn.add_stream(self.vessel.resources.amount, resource)
			self.telemetry[field] = stream
			stream.rate = 10

		frame = self.vessel.orbit.body.non_rotating_reference_frame
		for name, method in {"position": self.vessel.position, "velocity": self.vessel.velocity, "rotation": self.vessel.rotation, "angular_velocity": self.vessel.angular_velocity}.items():
			stream = self.conn.add_stream(method, frame)
			self.telemetry[name] = stream
			stream.rate = 10

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

	def _clear_telemetry(self):
		self._clear_stage_streams()
		for stream in self.telemetry.values():
			try:
				stream.remove()
			except (self.conn.krpc.ObjectDestroyedException, self.conn.krpc.StreamRemovedException):
				pass
		self.telemetry.clear()

	def close(self):
		try:
			self._clear_telemetry()
		finally:
			self.conn.close()

	def reset(self, delay=5.0, timeout=30.0):
		self.stop_controls()
		if not self.sc.can_revert_to_launch():
			raise RuntimeError("Revert to launch is unavailable.")

		self._clear_telemetry()
		self.revert_to_launch(delay=delay)
		self._wait_and_bind(require_pad=True, deadline=time.monotonic() + timeout, label="launchpad")
		return self.read_observation()

	def capture_named_save(self, name, destination, *, save_dir, sampled_ut, expected=None):
		source = Path(save_dir) / f"{name}.sfs"
		destination = Path(destination)
		temporary = destination.with_suffix(".sfs.tmp")
		if source.exists():
			raise FileExistsError(f"Capture requires a unique KSP save name: {source}")

		krpc_service = self.conn.krpc
		previous_pause = krpc_service.paused
		try:
			try:
				krpc_service.paused = True
				if not krpc_service.paused:
					raise SaveUnavailable("KSP did not pause for capture.")
				frozen_ut = float(self.sc.ut)
				if abs(frozen_ut - sampled_ut) > 0.25:
					raise SaveUnavailable("Crossing telemetry is too old at pause.")
				if expected is not None:
					try:
						self._verify_checkpoint(name, expected, altitude_tolerance=10.0)
					except CheckpointMismatch as error:
						raise SaveUnavailable(str(error)) from error

				self.sc.save(name)
				if not krpc_service.paused or abs(float(self.sc.ut) - frozen_ut) > 1e-6:
					raise SaveUnavailable("Game time advanced during the paused save.")
			finally:
				krpc_service.paused = previous_pause

			if not source.is_file() or source.stat().st_size == 0:
				raise RuntimeError(f"KSP did not create a usable save at {source}")
			shutil.copyfile(source, temporary)
			temporary.replace(destination)
		finally:
			temporary.unlink(missing_ok=True)
			source.unlink(missing_ok=True)

	def _verify_checkpoint(self, name, expected, *, altitude_tolerance):
		ut = float(self.sc.ut)
		altitude = float(self.flight.mean_altitude)
		vessel = self.vessel
		identity = (vessel.name, vessel.orbit.body.name, self.control.current_stage)
		if identity != (expected["vessel"], expected["body"], expected["stage"]):
			raise CheckpointMismatch(f"{name!r} vessel, body or stage mismatch: {identity!r}")
		if not math.isfinite(altitude) or abs(altitude - expected["altitude"]) > altitude_tolerance:
			raise CheckpointMismatch(f"{name!r} altitude mismatch: expected={expected['altitude']:.2f} m, actual={altitude:.2f} m, UT delta={ut - expected['ut']:+.3f} s")

		resources = vessel.resources
		fuel = {}
		for field, amount in expected["fuel"].items():
			actual = float(resources.amount(self.PROPELLANT_NAMES[field]))
			if not math.isclose(actual, amount, rel_tol=0.02, abs_tol=0.5):
				raise CheckpointMismatch(f"{name!r} fuel mismatch for {field}: expected={amount:.2f}, actual={actual:.2f}")
			fuel[field] = actual
		return {"ut": ut, "altitude": altitude, **fuel}

	def load_named_save(self, name, *, expected=None, timeout=30.0):
		self._clear_telemetry()
		krpc_service = self.conn.krpc
		previous_pause = krpc_service.paused
		try:
			krpc_service.paused = False
			self.sc.load(name)
			deadline = time.monotonic() + timeout
			if expected is None:
				self._wait_and_bind(require_pad=True, deadline=deadline, label=name)
				self.manual_control()
				return self.read_observation()

			checked = self._bind_loaded_checkpoint(name, expected, deadline)
			self.manual_control()
			observation = self.read_observation()
			observation.update(checked)
			return observation
		finally:
			krpc_service.paused = previous_pause

	def _bind_loaded_checkpoint(self, name, expected, deadline):
		sc = self.sc
		krpc_service = self.conn.krpc
		vessel_name = expected["vessel"]
		body_name = expected["body"]
		stage = int(expected["stage"])
		expected_ut = float(expected["ut"])
		min_ut = expected_ut - 0.25
		max_ut = expected_ut + 0.50

		def pause_at_checkpoint():
			ut = sc.ut
			if ut < min_ut or ut > max_ut:
				return False, ut
			vessel = sc.active_vessel
			if vessel is None:
				return False, ut
			if vessel.name != vessel_name or vessel.orbit.body.name != body_name:
				return False, ut
			if vessel.control.current_stage != stage:
				return False, ut
			root = vessel.parts.root
			if root is None:
				return False, ut
			krpc_service.paused = True
			return True, ut

		closest_ut = None
		last_error = None
		while time.monotonic() < deadline:
			try:
				ready, ut = self.conn.run(pause_at_checkpoint)
				if closest_ut is None or abs(ut - expected_ut) < abs(closest_ut - expected_ut):
					closest_ut = ut
				if ready:
					if not krpc_service.paused:
						raise CheckpointMismatch(f"{name!r} did not pause on reload.")
					self._bind_vessel()
					return self._verify_checkpoint(name, expected, altitude_tolerance=50.0)
			except (RuntimeError, self.conn.krpc.ObjectDestroyedException) as error:
				if not self._vessel_unavailable(error):
					raise
				last_error = error
				if krpc_service.paused:
					krpc_service.paused = False
			time.sleep(0.02)

		raise TimeoutError(f"KSP did not load {name!r} within the checkpoint window: expected UT={expected_ut:.3f}, closest sampled UT={closest_ut!r}") from last_error
