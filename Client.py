import time

import krpc  # type: ignore


class KSPClient:
	def __init__(self, name="KSP Client"):
		self.conn = krpc.connect(name=name)
		self.sc = self.conn.space_center
		self.telemetry = {}

		self._bind_vessel()

	def _bind_vessel(self):
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
			"stage": stream_property(self.control, "current_stage"),
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

			except RuntimeError as exc:
				if "The vessel has no parts" in str(exc):
					continue

				# Preserve unrelated server errors.
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
		self.control.activate_next_stage()

	def read_observation(self):
		observation = {name: stream() for name, stream in self.telemetry.items()}
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
		observation["dt"] = observation["ut"] - start_ut
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
		self.conn.close()

	def reset(self, delay=5.0, timeout=30.0):
		self.stop_controls()

		if not self.sc.can_revert_to_launch():
			raise RuntimeError("Revert to launch is unavailable.")

		# Remove streams attached to the previous flight.
		for stream in self.telemetry.values():
			stream.remove()

		self.telemetry.clear()

		self.revert_to_launch(delay=delay)

		# Wait for the launchpad flight to become available.
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
