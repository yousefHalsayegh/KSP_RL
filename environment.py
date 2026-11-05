import math

import numpy as np


class KSPEnvironment:
	def __init__(
		self,
		client,
		task="orbit",
		orbit_weight=5.0,
		fuel_weight=5.0,
		time_weight=0.1,
		episode_seconds=600.0,
		confirmation_seconds=5.0,
		staging_reward_budget=1.0,
		airtime_weight=10.0,
		altitude_reward_per_step=0.1,
	):

		self.client = client
		self.task = task
		self.stage_requested = False
		self.staging_reward_budget = staging_reward_budget
		self.staging_reward_paid = 0.0
		self.rewarded_stages = set()
		self.stationary_since = None
		self.stationary_timeout = 10.0
		self.stationary_speed = 0.5

		self.orbit_weight = orbit_weight
		self.fuel_weight = fuel_weight
		self.time_weight = time_weight
		self.episode_seconds = episode_seconds
		self.confirmation_seconds = confirmation_seconds
		self.airtime_weight = airtime_weight

		self.episode_start_ut = None
		self.has_been_airborne = False
		self.episode_finished = False
		self.orbit_valid_since = None
		self.surface_valid_since = None

		body = client.vessel.orbit.body
		self.atmosphere_depth = body.atmosphere_depth
		self.altitude_step_m = self.atmosphere_depth / 20.0
		self.altitude_reward_per_step = altitude_reward_per_step

		resource_names = {"liquid_fuel": "LiquidFuel", "oxidizer": "Oxidizer", "solid_fuel": "SolidFuel", "monopropellant": "MonoPropellant"}

		self.fuel_densities = {field: client.sc.Resources.density(resource) for field, resource in resource_names.items()}

		initial = client.read_observation()
		self.initialize_observation_scales(initial)
		self.last_state = self.encode_observation(initial)
		self.previous_altitude_band = self.altitude_band(initial["altitude"])
		self.highest_altitude_band = self.previous_altitude_band

		self.initial_propellant_mass = self.propellant_mass(initial)

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
		self.initial_stage_count = max(1, int(observation["stage"]))

		self.resource_scales = {field: max(observation[field], 1.0) for field in ("liquid_fuel", "oxidizer", "solid_fuel", "monopropellant", "electric_charge")}

	def altitude_band(self, altitude):
		altitude = max(0.0, float(altitude))
		return min(10, math.floor(altitude / self.altitude_step_m))

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

		staging_reward = 0.0
		activated_stage = None
		try:
			if activate_stage:
				activated_stage = self.client.activate_next_stage()
				if activated_stage is not None and activated_stage not in self.rewarded_stages:
					self.rewarded_stages.add(activated_stage)
					remaining = max(0.0, self.staging_reward_budget - self.staging_reward_paid)
					staging_reward = min(self.staging_reward_budget / self.initial_stage_count, remaining)
					self.staging_reward_paid += staging_reward
			observation = self.client.step(throttle=(throttle + 1.0) / 2.0, pitch=pitch, yaw=yaw, roll=roll, duration=duration, manual=True)

		except Exception:
			if self.client.vessel_exists():
				raise

			return self.end_failed_episode("spacecraft_destroyed", staging_reward)

		if not self.client.vessel_exists():
			return self.end_failed_episode("spacecraft_destroyed", staging_reward)

		mass = float(observation["mass"])

		if not math.isfinite(mass):
			raise ValueError("Received non-finite spacecraft mass.")

		if mass <= 0:
			return self.end_failed_episode("spacecraft_mass_lost", staging_reward)

		reward = self.calculate_reward(observation, staging_reward)
		terminated, truncated, info = self.check_episode_end(observation)

		terminal_reward = self.outcome_reward(info["end_reason"])
		reward += terminal_reward
		self.reward_components["outcome_reward"] = terminal_reward

		self.episode_finished = terminated or truncated

		state = self.encode_observation(observation)
		self.last_state = state.copy()

		info["telemetry"] = observation.copy()
		info["reward_components"] = self.reward_components.copy()
		info["stage_activated"] = activated_stage

		return state, reward, terminated, truncated, info

	def propellant_mass(self, observation):
		return sum(observation[field] * density for field, density in self.fuel_densities.items())

	def orbit_progress(self, observation):
		apoapsis = observation["apoapsis"]
		periapsis = observation["periapsis"]

		if not (math.isfinite(apoapsis) and math.isfinite(periapsis)):
			return None

		height = self.atmosphere_depth
		return (min(apoapsis, height) + min(periapsis, height)) / (2.0 * height)

	def calculate_reward(self, observation, staging_reward=0.0):
		current_progress = self.orbit_progress(observation)

		orbit_reward = 0.0
		if current_progress is not None and self.previous_orbit_progress is not None:
			orbit_reward = self.orbit_weight * (current_progress - self.previous_orbit_progress)
		fuel_used = max(0.0, self.propellant_mass(self.previous_observation) - self.propellant_mass(observation))
		fuel_penalty = self.fuel_weight * fuel_used / self.initial_propellant_mass
		time_penalty = self.time_weight * observation["dt"] / self.episode_seconds
		situations = self.client.sc.VesselSituation
		airborne = observation["situation"] in (situations.flying, situations.sub_orbital, situations.orbiting, situations.escaping)
		airtime_reward = self.airtime_weight * observation["dt"] / self.episode_seconds if airborne else 0.0
		current_band = self.altitude_band(observation["altitude"])

		new_bands = max(0, current_band - self.highest_altitude_band)
		lost_bands = max(0, self.previous_altitude_band - current_band)

		altitude_reward = (new_bands - lost_bands) * self.altitude_reward_per_step

		self.highest_altitude_band = max(self.highest_altitude_band, current_band)
		self.previous_altitude_band = current_band

		self.reward_components = {
			"orbit_reward": orbit_reward,
			"fuel_penalty": fuel_penalty,
			"time_penalty": time_penalty,
			"staging_reward": staging_reward,
			"airtime_reward": airtime_reward,
			"altitude_reward": altitude_reward,
			"outcome_reward": 0.0,
		}

		self.previous_observation = observation.copy()
		if current_progress is not None:
			self.previous_orbit_progress = current_progress

		return orbit_reward - fuel_penalty - time_penalty + staging_reward + airtime_reward + altitude_reward

	def check_episode_end(self, observation):
		situations = self.client.sc.VesselSituation
		situation = observation["situation"]
		ut = observation["ut"]
		elapsed = ut - self.episode_start_ut
		speed = float(observation["surface_speed"])
		if not math.isfinite(speed) or speed < 0:
			raise ValueError("Received invalid surface-relative speed.")

		if speed <= self.stationary_speed:
			if self.stationary_since is None:
				self.stationary_since = ut
		else:
			self.stationary_since = None

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
			and 0.0 <= thrust < 1.0
		)

		if valid_orbit:
			if self.orbit_valid_since is None:
				self.orbit_valid_since = ut
		else:
			self.orbit_valid_since = None

		success = self.orbit_valid_since is not None and ut >= self.orbit_valid_since + self.confirmation_seconds

		on_surface = self.has_been_airborne and situation in (situations.landed, situations.splashed)
		if not on_surface:
			self.surface_valid_since = None
		elif self.surface_valid_since is None:
			self.surface_valid_since = ut

		returned_to_surface = self.surface_valid_since is not None and ut >= self.surface_valid_since + self.confirmation_seconds
		surface_confirmation_pending = self.surface_valid_since is not None and not returned_to_surface
		timed_out = self.stationary_since is not None and ut >= self.stationary_since + self.stationary_timeout and not surface_confirmation_pending
		terminated = success or returned_to_surface or timed_out
		truncated = False

		reason = None

		if success:
			reason = "orbit_achieved"
		elif returned_to_surface:
			reason = "returned_to_surface"
		elif timed_out:
			reason = "time_limit"

		return (
			terminated,
			truncated,
			{"elapsed": elapsed, "success": success, "end_reason": reason, "apoapsis": apoapsis, "periapsis": periapsis, "surface_confirmation_pending": surface_confirmation_pending},
		)

	def reset(self):
		observation = self.client.reset(delay=5.0)
		self.initialize_observation_scales(observation)
		self.previous_altitude_band = self.altitude_band(observation["altitude"])
		self.highest_altitude_band = self.previous_altitude_band
		self.stage_requested = False
		self.staging_reward_paid = 0.0
		self.rewarded_stages = set()
		self.episode_start_ut = None
		self.has_been_airborne = False
		self.episode_finished = False
		self.orbit_valid_since = None
		self.surface_valid_since = None
		self.stationary_since = None

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

		features["stage"] = observation["stage"] / 6.0
		features["stage_requested"] = float(self.stage_requested)

		elapsed = 0.0 if self.episode_start_ut is None else observation["ut"] - self.episode_start_ut
		features["elapsed"] = elapsed / self.episode_seconds

		features["throttle"] = observation["throttle"]
		features["pitch_input"] = observation["pitch_input"]
		features["yaw_input"] = observation["yaw_input"]
		features["roll_input"] = observation["roll_input"]

		features["next_stage_available"] = observation["next_stage_available"]
		for name in ("next_stage_engine_count", "next_stage_decoupled_part_count", "next_stage_parachute_count", "next_stage_launch_clamp_count"):
			features[name] = math.log1p(observation[name])
		features["next_stage_vacuum_twr"] = observation["next_stage_vacuum_thrust"] / (mass * self.gravity_scale)
		features["next_stage_dry_mass_fraction"] = observation["next_stage_dry_mass"] / self.mass_scale
		features["next_stage_propellant_mass_fraction"] = observation["next_stage_propellant_mass"] / self.mass_scale

		state = np.asarray(list(features.values()), dtype=np.float32)

		if not np.isfinite(state).all():
			raise ValueError("Policy observation contains invalid telemetry.")

		self.observation_names = tuple(features.keys())

		return state

	def end_failed_episode(self, reason, staging_reward=0.0):
		self.episode_finished = True
		outcome_reward = self.outcome_reward(reason)
		reward = outcome_reward + staging_reward

		self.reward_components = {
			"orbit_reward": 0.0,
			"fuel_penalty": 0.0,
			"time_penalty": 0.0,
			"airtime_reward": 0.0,
			"altitude_reward": 0,
			"staging_reward": staging_reward,
			"outcome_reward": outcome_reward,
		}

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
			rewards = {"orbit_achieved": 10.0, "returned_to_surface": -10.0, "spacecraft_destroyed": -15.0, "spacecraft_mass_lost": -5.0, "time_limit": -20.0}

			if reason is None:
				return 0.0

			if reason not in rewards:
				raise ValueError(f"Unknown orbit outcome: {reason}")

			return rewards[reason]

		raise NotImplementedError(f"Outcome rewards are undefined for task: {self.task}")
