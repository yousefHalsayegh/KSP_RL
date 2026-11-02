import math

import numpy as np


class KSPEnvironment:
	def __init__(self, client, task="orbit", orbit_weight=5.0, fuel_weight=0.1, time_weight=0.01, episode_seconds=600.0, confirmation_seconds=5.0):
		if task != "orbit":
			raise ValueError("Only the orbit task is implemented currently.")

		self.client = client
		self.task = task
		self.stage_requested = False

		self.orbit_weight = orbit_weight
		self.fuel_weight = fuel_weight
		self.time_weight = time_weight
		self.episode_seconds = episode_seconds
		self.confirmation_seconds = confirmation_seconds

		self.episode_start_ut = None
		self.has_been_airborne = False
		self.episode_finished = False
		self.orbit_valid_since = None

		body = client.vessel.orbit.body
		self.atmosphere_depth = body.atmosphere_depth

		resource_names = {"liquid_fuel": "LiquidFuel", "oxidizer": "Oxidizer", "solid_fuel": "SolidFuel", "monopropellant": "MonoPropellant"}

		self.fuel_densities = {field: client.sc.Resources.density(resource) for field, resource in resource_names.items()}

		initial = client.read_observation()
		self.initialize_observation_scales(initial)
		self.last_state = self.encode_observation(initial)

		self.initial_propellant_mass = self.propellant_mass(initial)

		if self.initial_propellant_mass <= 0:
			raise ValueError("The spacecraft has no measured propellant.")

		self.previous_observation = initial.copy()
		self.previous_orbit_progress = self.orbit_progress(initial)
		self.reward_components = {}
		self.client.manual_control()

	def initialize_observation_scales(self, observation):
		body = self.client.vessel.orbit.body

		self.radius_scale = body.equatorial_radius
		self.velocity_scale = math.sqrt(body.gravitational_parameter / self.radius_scale)
		self.gravity_scale = body.gravitational_parameter / self.radius_scale**2

		self.mass_scale = max(observation["mass"], 1.0)

		self.resource_scales = {field: max(observation[field], 1.0) for field in ("liquid_fuel", "oxidizer", "solid_fuel", "monopropellant", "electric_charge")}

	def step(self, action, duration=0.2):
		"""
		action: [throttle, pitch, yaw, roll, stage_signal]
		Each value is in [-1, 1].
		"""
		if self.episode_finished:
			raise RuntimeError("Episode finished. Start a new episode before stepping.")

		if self.episode_start_ut is None:
			self.episode_start_ut = self.client.sc.ut

		if len(action) != 5:
			raise ValueError("Expected five action values.")

		values = [float(value) for value in action]

		if not all(math.isfinite(value) for value in values):
			raise ValueError("Actions must contain finite numbers.")

		throttle, pitch, yaw, roll, stage_signal = [max(-1.0, min(1.0, value)) for value in values]

		requested = stage_signal > 0.5
		activate_stage = requested and not self.stage_requested
		self.stage_requested = requested

		if activate_stage:
			self.client.activate_next_stage()

		try:
			observation = self.client.step(throttle=(throttle + 1.0) / 2.0, pitch=pitch, yaw=yaw, roll=roll, duration=duration, manual=True)

		except Exception:
			if self.client.vessel_exists():
				# The craft still exists: don't disguise an unrelated error
				# as a valid training transition.
				raise

			return self.end_failed_episode("spacecraft_destroyed")

		if not self.client.vessel_exists():
			return self.end_failed_episode("spacecraft_destroyed")

		mass = float(observation["mass"])

		if not math.isfinite(mass):
			raise ValueError("Received non-finite spacecraft mass.")

		if mass <= 0:
			return self.end_failed_episode("spacecraft_mass_lost")

		reward = self.calculate_reward(observation)
		terminated, truncated, info = self.check_episode_end(observation)

		# Add the task-specific success bonus or failure penalty.
		terminal_reward = self.outcome_reward(info["end_reason"])
		reward += terminal_reward
		self.reward_components["outcome_reward"] = terminal_reward

		self.episode_finished = terminated or truncated

		state = self.encode_observation(observation)
		self.last_state = state.copy()

		info["telemetry"] = observation.copy()
		info["reward_components"] = self.reward_components.copy()

		return state, reward, terminated, truncated, info

	def propellant_mass(self, observation):
		return sum(observation[field] * density for field, density in self.fuel_densities.items())

	def orbit_progress(self, observation):
		apoapsis = observation["apoapsis"]
		periapsis = observation["periapsis"]

		if not (math.isfinite(apoapsis) and math.isfinite(periapsis)):
			return None

		apoapsis_progress = min(apoapsis / self.atmosphere_depth, 1.0)
		periapsis_progress = min(periapsis / self.atmosphere_depth, 1.0)

		return 0.5 * (apoapsis_progress + periapsis_progress)

	def calculate_reward(self, observation):
		current_progress = self.orbit_progress(observation)

		progress_change = 0.0

		if current_progress is not None:
			if self.previous_orbit_progress is not None:
				progress_change = current_progress - self.previous_orbit_progress

			self.previous_orbit_progress = current_progress

		previous_fuel = self.propellant_mass(self.previous_observation)
		current_fuel = self.propellant_mass(observation)

		fuel_spent = max(0.0, previous_fuel - current_fuel)
		fuel_fraction_spent = fuel_spent / self.initial_propellant_mass

		time_penalty = self.time_weight * observation["dt"] / self.episode_seconds
		orbit_reward = self.orbit_weight * progress_change
		fuel_penalty = self.fuel_weight * fuel_fraction_spent

		self.reward_components = {"orbit_reward": orbit_reward, "fuel_penalty": fuel_penalty, "time_penalty": time_penalty, "outcome_reward": 0.0}

		self.previous_observation = observation.copy()

		return orbit_reward - fuel_penalty - time_penalty

	def check_episode_end(self, observation):
		situations = self.client.sc.VesselSituation
		situation = observation["situation"]
		ut = observation["ut"]

		if situation in (situations.flying, situations.sub_orbital, situations.orbiting, situations.escaping):
			self.has_been_airborne = True

		apoapsis = observation["apoapsis"]
		periapsis = observation["periapsis"]
		eccentricity = observation["eccentricity"]
		thrust = observation["thrust"]

		valid_orbit = (
			all(math.isfinite(value) for value in (apoapsis, periapsis, eccentricity, thrust))
			and 0.0 <= eccentricity < 1.0
			and apoapsis > self.atmosphere_depth
			and periapsis > self.atmosphere_depth
			and 0.0 <= thrust < 1.0  # Effectively no thrust, in newtons
		)

		if valid_orbit:
			if self.orbit_valid_since is None:
				self.orbit_valid_since = ut
		else:
			self.orbit_valid_since = None

		success = self.orbit_valid_since is not None and ut - self.orbit_valid_since >= self.confirmation_seconds

		returned_to_surface = self.has_been_airborne and situation in (situations.landed, situations.splashed)

		terminated = success or returned_to_surface

		elapsed = ut - self.episode_start_ut
		truncated = not terminated and elapsed >= self.episode_seconds

		reason = None

		if success:
			reason = "orbit_achieved"
		elif returned_to_surface:
			reason = "returned_to_surface"
		elif truncated:
			reason = "time_limit"

		return terminated, truncated, {"elapsed": elapsed, "success": success, "end_reason": reason, "apoapsis": apoapsis, "periapsis": periapsis}

	def reset(self):
		observation = self.client.reset(delay=5.0)
		self.initialize_observation_scales(observation)
		self.stage_requested = False
		self.episode_start_ut = None
		self.has_been_airborne = False
		self.episode_finished = False
		self.orbit_valid_since = None

		self.initial_propellant_mass = self.propellant_mass(observation)

		if self.initial_propellant_mass <= 0:
			raise ValueError("The spacecraft has no measured propellant.")

		self.previous_observation = observation.copy()
		self.previous_orbit_progress = self.orbit_progress(observation)
		self.reward_components = {}

		self.last_state = self.encode_observation(observation)
		return self.last_state.copy()

	def encode_observation(self, observation):
		features = {}

		def add_vector(name, values, scale=1.0):
			for index, value in enumerate(values):
				features[f"{name}_{index}"] = float(value) / scale

		def add_orbital_value(name):
			value = float(observation[name])
			valid = math.isfinite(value)

			features[f"{name}_valid"] = float(valid)

			if not valid:
				features[name] = 0.0
				return

			if name != "eccentricity":
				value /= self.atmosphere_depth

			# Preserve sign while compressing extreme orbital values.
			features[name] = math.copysign(math.log1p(abs(value)), value)

		add_vector("position", observation["position"], self.radius_scale)
		add_vector("velocity", observation["velocity"], self.velocity_scale)

		rotation = np.asarray(observation["rotation"], dtype=np.float64)

		if rotation.shape != (4,) or not np.isfinite(rotation).all():
			raise ValueError("Invalid orientation quaternion.")

		norm = np.linalg.norm(rotation)

		if norm < 1e-8:
			raise ValueError("Orientation quaternion has zero length.")

		rotation /= norm

		# q and -q describe the same orientation.
		if rotation[3] < 0:
			rotation = -rotation

		add_vector("rotation", rotation)
		add_vector("angular_velocity", observation["angular_velocity"])

		features["altitude"] = observation["altitude"] / self.atmosphere_depth
		features["surface_altitude"] = observation["surface_altitude"] / self.atmosphere_depth
		features["vertical_speed"] = observation["vertical_speed"] / self.velocity_scale
		features["horizontal_speed"] = observation["horizontal_speed"] / self.velocity_scale

		for name in ("apoapsis", "periapsis", "eccentricity"):
			add_orbital_value(name)

		features["dynamic_pressure"] = observation["dynamic_pressure"] / 50_000.0

		mass = float(observation["mass"])

		if not math.isfinite(mass) or mass <= 0:
			raise ValueError("Invalid spacecraft mass.")

		features["mass"] = mass / self.mass_scale
		features["thrust_to_weight"] = observation["thrust"] / (mass * self.gravity_scale)
		features["available_thrust_to_weight"] = observation["available_thrust"] / (mass * self.gravity_scale)

		for field, scale in self.resource_scales.items():
			features[field] = observation[field] / scale

		# Six is the initial staging count for our current Kerbal 1.
		features["stage"] = observation["stage"] / 6.0
		features["stage_requested"] = float(self.stage_requested)

		elapsed = 0.0 if self.episode_start_ut is None else observation["ut"] - self.episode_start_ut
		features["elapsed"] = elapsed / self.episode_seconds

		features["throttle"] = observation["throttle"]
		features["pitch_input"] = observation["pitch_input"]
		features["yaw_input"] = observation["yaw_input"]
		features["roll_input"] = observation["roll_input"]

		state = np.asarray(list(features.values()), dtype=np.float32)

		if not np.isfinite(state).all():
			raise ValueError("Policy observation contains invalid telemetry.")

		self.observation_names = tuple(features.keys())

		return state

	def end_failed_episode(self, reason):
		self.episode_finished = True
		reward = self.outcome_reward(reason)

		self.reward_components = {"orbit_reward": 0.0, "fuel_penalty": 0.0, "time_penalty": 0.0, "outcome_reward": reward}

		info = {
			"elapsed": self.client.sc.ut - self.episode_start_ut,
			"success": False,
			"end_reason": reason,
			"telemetry": self.previous_observation.copy(),
			"reward_components": self.reward_components.copy(),
			"telemetry_stale": True,
		}

		return self.last_state.copy(), reward, True, False, info

	def outcome_reward(self, reason):
		if self.task == "orbit":
			rewards = {"orbit_achieved": 10.0, "returned_to_surface": -10.0, "spacecraft_destroyed": -20.0, "spacecraft_mass_lost": -20.0, "time_limit": -5.0}

			if reason is None:
				return 0.0

			if reason not in rewards:
				raise ValueError(f"Unknown orbit outcome: {reason}")

			return rewards[reason]

		raise NotImplementedError(f"Outcome rewards are undefined for task: {self.task}")
