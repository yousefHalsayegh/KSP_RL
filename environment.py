import math

import numpy as np


class KSPEnvironment:
	# Reserve task inputs now; transfer/landing still need their own environments.
	TASKS = ("orbit", "transfer", "land")
	SUPPORTED_TASKS = ("orbit",)
	OBSERVATION_VERSION = 3
	ASCENT_MILESTONE_FRACTIONS = (0.10, 0.25, 0.50, 0.75)
	PROPELLANTS = {"liquid_fuel": "LiquidFuel", "oxidizer": "Oxidizer", "solid_fuel": "SolidFuel", "monopropellant": "MonoPropellant"}
	_CHECKPOINT_FIELDS = (
		"initial_propellant_mass",
		"mass_scale",
		"resource_scales",
		"initial_stage_count",
		"initial_altitude_band",
		"highest_altitude_band",
		"_ascent_milestones_reached",
		"atmosphere_exit_achieved",
		"staging_reward_paid",
		"discarded_propellant_mass",
		"stage_requested",
	)

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
		altitude_reward_per_step=0.02,
		rotation_weight=0.1,
		angular_speed_scale=1.0,
		altitude_step_m=100.0,
		orbit_altitude=None,
		staging_cooldown_seconds=2.0,
		saved_fuel_weight=1.0,
		atmosphere_exit_bonus=5.0,
		ascent_milestone_bonus=1.0,
	):

		self.client = client
		self.task = task
		self._validate_task()
		self.stage_requested = False
		self.staging_reward_budget = staging_reward_budget
		self.staging_reward_paid = 0.0
		self.rewarded_stages = set()
		self.staging_cooldown_seconds = float(staging_cooldown_seconds)
		self.saved_fuel_weight = float(saved_fuel_weight)
		self.atmosphere_exit_bonus = float(atmosphere_exit_bonus)
		self.ascent_milestone_bonus = float(ascent_milestone_bonus)
		self.atmosphere_exit_achieved = False
		self.last_stage_ut = None
		self.pending_stages = []
		self.discarded_propellant_mass = 0.0
		self.stationary_since = None
		self.stationary_timeout = 10.0
		self.stationary_speed = 0.5

		self.orbit_weight = orbit_weight
		self.fuel_weight = fuel_weight
		self.time_weight = time_weight
		self.episode_seconds = episode_seconds
		self.confirmation_seconds = confirmation_seconds
		self.rotation_weight = rotation_weight
		self.angular_speed_scale = angular_speed_scale

		self.episode_start_ut = None
		self.has_been_airborne = False
		self.episode_finished = False
		self.orbit_valid_since = None
		self.surface_valid_since = None

		self.altitude_step_m = float(altitude_step_m)
		self.altitude_reward_per_step = altitude_reward_per_step
		self._requested_orbit_altitude = orbit_altitude

		for name in (
			"orbit_weight",
			"fuel_weight",
			"time_weight",
			"staging_reward_budget",
			"altitude_reward_per_step",
			"rotation_weight",
			"confirmation_seconds",
			"saved_fuel_weight",
			"atmosphere_exit_bonus",
			"ascent_milestone_bonus",
		):
			value = float(getattr(self, name))
			if not math.isfinite(value) or value < 0:
				raise ValueError(f"{name} must be finite and nonnegative.")
			setattr(self, name, value)
		if not math.isfinite(self.staging_cooldown_seconds) or not 0 < self.staging_cooldown_seconds <= 3.0:
			raise ValueError("staging_cooldown_seconds must be positive and at most 3 seconds.")
		for name in ("episode_seconds", "angular_speed_scale", "altitude_step_m"):
			value = float(getattr(self, name))
			if not math.isfinite(value) or value <= 0:
				raise ValueError(f"{name} must be finite and positive.")
			setattr(self, name, value)

		resource_names = self.PROPELLANTS

		self.fuel_densities = {field: client.sc.Resources.density(resource) for field, resource in resource_names.items()}

		initial = client.read_observation()
		self.initialize_observation_scales(initial)
		self.highest_altitude_band = self.altitude_band(initial["altitude"])
		self.initial_altitude_band = self.highest_altitude_band
		self.last_state = self.encode_observation(initial)

		self.initial_propellant_mass = self.propellant_mass(initial)
		if not math.isfinite(self.initial_propellant_mass) or self.initial_propellant_mass <= 0:
			raise ValueError("The spacecraft has no valid measured propellant mass.")

		self.previous_observation = initial.copy()
		self.previous_orbit_progress = self.orbit_progress(initial)
		self.reward_components = {}
		self.client.manual_control()

	def _validate_task(self):
		if self.task not in self.TASKS:
			raise ValueError(f"Unknown task: {self.task}. Expected one of {self.TASKS}.")
		if self.task not in self.SUPPORTED_TASKS:
			raise NotImplementedError(f"Task '{self.task}' has reserved inputs but its rewards, goals and ending conditions are not implemented yet.")

	def reward_configuration(self):
		"""Resolved settings for W&B/checkpoint metadata, not policy inputs."""
		return {
			"observation_version": self.OBSERVATION_VERSION,
			"task": self.task,
			"atmosphere_depth": self.atmosphere_depth,
			"ascent_milestone_altitudes": self.ascent_milestone_altitudes,
			"orbit_altitude": self.orbit_altitude,
			"altitude_step_m": self.altitude_step_m,
			"altitude_bands": self.max_altitude_band,
			"maximum_positive_altitude_reward": self.max_altitude_band * self.altitude_reward_per_step,
			"fuel_penalty_mode": "propellant_discarded_by_staging_only",
			"staging_reward_mode": "confirmed_ignition_or_empty_separation_with_flight_progress",
			"empty_tank_fraction_tolerance": 1e-4,
			**{
				name: getattr(self, name)
				for name in (
					"orbit_weight",
					"fuel_weight",
					"time_weight",
					"episode_seconds",
					"confirmation_seconds",
					"staging_reward_budget",
					"staging_cooldown_seconds",
					"saved_fuel_weight",
					"atmosphere_exit_bonus",
					"ascent_milestone_bonus",
					"altitude_reward_per_step",
					"rotation_weight",
					"angular_speed_scale",
					"stationary_timeout",
					"stationary_speed",
				)
			},
		}

	def initialize_observation_scales(self, observation):
		body = self.client.vessel.orbit.body
		self.atmosphere_depth = float(body.atmosphere_depth)
		self.ascent_milestone_altitudes = ()
		if self.atmosphere_depth > 0.0:
			self.ascent_milestone_altitudes = tuple(self.atmosphere_depth * fraction for fraction in self.ASCENT_MILESTONE_FRACTIONS)
		self._ascent_milestones_reached = sum(observation["altitude"] >= altitude for altitude in self.ascent_milestone_altitudes)
		self.orbit_altitude = self.atmosphere_depth if self._requested_orbit_altitude is None else float(self._requested_orbit_altitude)
		if not math.isfinite(self.orbit_altitude) or self.orbit_altitude <= 0:
			raise ValueError("A positive orbit_altitude is required on an airless body.")
		if self.orbit_altitude < self.atmosphere_depth:
			raise ValueError("orbit_altitude must be at or above the atmosphere boundary.")
		self.max_altitude_band = max(1, math.ceil(self.orbit_altitude / self.altitude_step_m))

		self.radius_scale = body.equatorial_radius
		self.velocity_scale = math.sqrt(body.gravitational_parameter / self.radius_scale)
		self.gravity_scale = body.gravitational_parameter / self.radius_scale**2

		self.mass_scale = max(observation["mass"], 1.0)
		self.initial_stage_count = max(1, int(observation["stage"]))

		self.resource_scales = {field: max(observation[field], 1.0) for field in ("liquid_fuel", "oxidizer", "solid_fuel", "monopropellant", "electric_charge")}

	def altitude_band(self, altitude):
		altitude = float(altitude)
		if not math.isfinite(altitude):
			raise ValueError("Received non-finite altitude.")
		if altitude >= self.orbit_altitude:
			return self.max_altitude_band
		return math.floor(max(0.0, altitude) / self.altitude_step_m)

	def staging_cooldown_remaining(self, ut):
		if self.last_stage_ut is None:
			return 0.0
		return min(self.staging_cooldown_seconds, max(0.0, self.staging_cooldown_seconds - (ut - self.last_stage_ut)))

	def _part_propellant(self, part):
		resources = part.resources
		names = set(resources.names)
		mass = capacity = 0.0
		for field, name in self.PROPELLANTS.items():
			if name in names:
				density = self.fuel_densities[field]
				mass += float(resources.amount(name)) * density
				capacity += float(resources.max(name)) * density
		if not all(math.isfinite(value) and value >= 0 for value in (mass, capacity)):
			raise ValueError("Invalid propellant mass in a stage part.")
		return mass, capacity

	def _snapshot_stage(self):
		"""Read parts only for a staging attempt, not on every control step."""
		for attempt in range(3):
			try:
				return self._snapshot_stage_parts()
			except self.client.conn.krpc.ObjectDestroyedException:
				if attempt == 2 or not self.client.vessel_exists():
					raise

	def _snapshot_stage_parts(self):
		# Remaining mass is sampled before the command; confirmed absent parts
		# are charged once. Burning between sampling and separation is approximate.
		next_stage = self.client.control.current_stage - 1
		if next_stage < 0:
			return None
		parts = self.client.vessel.parts.all
		detaching, igniting = [], []
		for part in parts:
			if part.decouple_stage == next_stage:
				mass, capacity = self._part_propellant(part)
				engine = part.engine
				detaching.append({"part": part, "mass": mass, "capacity": capacity, "depleted_engine": engine is not None and not engine.has_fuel})
			if part.stage == next_stage:
				engine = part.engine
				if engine is not None and not engine.active:
					igniting.append(part)
		before = self.client.read_observation()
		return {
			"stage": next_stage,
			"detaching": detaching,
			"igniting": igniting,
			"altitude": float(before["altitude"]),
			"orbit_progress": self.orbit_progress(before),
			"removed": [],
			"discarded_mass": 0.0,
		}

	def _evaluate_staging(self, observation):
		"""Return delayed, once-only staging reward and newly discarded fuel mass."""
		if not self.pending_stages:
			return 0.0, 0.0, []
		parts = self.client.vessel.parts.all
		ut = float(observation["ut"])
		progress = self.orbit_progress(observation)
		altitude_gained = self.altitude_band(observation["altitude"]) > self.initial_altitude_band
		reward = wasted_mass = 0.0
		keep, events = [], []
		for event in self.pending_stages:
			for item in event["detaching"]:
				part = item["part"]
				if part not in parts and part not in event["removed"]:
					event["removed"].append(part)
					event["discarded_mass"] += item["mass"]
					wasted_mass += item["mass"]

			all_removed = bool(event["detaching"]) and len(event["removed"]) == len(event["detaching"])
			capacity = sum(item["capacity"] for item in event["detaching"])
			tolerance = max(1e-6, capacity * 1e-4)
			propulsion_or_tank = capacity > 0 or any(item["depleted_engine"] for item in event["detaching"])
			empty_separation = all_removed and propulsion_or_tank and event["discarded_mass"] <= tolerance
			orbit_improved = progress is not None and event["orbit_progress"] is not None and progress > event["orbit_progress"] + 1e-6
			useful_flight = altitude_gained and (observation["vertical_speed"] > 0 or orbit_improved)
			climbed_after_ignition = observation["altitude"] >= event["altitude"] + 1.0 and observation["vertical_speed"] > 0
			ignition = False
			for part in event["igniting"]:
				if part in parts:
					try:
						engine = part.engine
						ignition |= engine is not None and engine.active and engine.thrust > 1.0
					except self.client.conn.krpc.ObjectDestroyedException:
						# The retained part can disappear after the parts list was read.
						continue
			clean_stage = event["discarded_mass"] <= tolerance and (not event["detaching"] or all_removed)
			within_window = ut <= event["expires_ut"]
			reason = None
			if within_window and clean_stage:
				if empty_separation and useful_flight:
					reason = "empty_separation_with_altitude_progress"
				elif ignition and (climbed_after_ignition or orbit_improved):
					reason = "ignition_with_flight_progress"

			paid = 0.0
			if reason is not None and event["stage"] not in self.rewarded_stages:
				remaining = max(0.0, self.staging_reward_budget - self.staging_reward_paid)
				paid = min(self.staging_reward_budget / self.initial_stage_count, remaining)
				self.staging_reward_paid += paid
				self.rewarded_stages.add(event["stage"])
				reward += paid
			if reason is not None or not within_window:
				events.append({"stage": event["stage"], "reason": reason or "not_qualified", "reward": paid, "discarded_propellant_mass": event["discarded_mass"]})
			else:
				keep.append(event)
		self.pending_stages = keep
		self.discarded_propellant_mass += wasted_mass
		return reward, wasted_mass, events

	def step(self, action, duration=0.2):
		"""
		action: [throttle, pitch, yaw, roll, stage_signal]
		Each value is in [-1, 1].
		"""
		self._validate_task()
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
		activate_stage = requested and not self.stage_requested and self.staging_cooldown_remaining(self.client.sc.ut) <= 0.0
		self.stage_requested = requested

		staging_reward = 0.0
		activated_stage = None
		stage_snapshot = None
		try:
			if activate_stage:
				stage_snapshot = self._snapshot_stage()
				if stage_snapshot is not None:
					activated_stage = self.client.activate_next_stage()
					if activated_stage is not None:
						self.last_stage_ut = self.client.sc.ut
						stage_snapshot["stage"] = activated_stage
						stage_snapshot["expires_ut"] = self.last_stage_ut + self.staging_cooldown_seconds
						self.pending_stages.append(stage_snapshot)
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

		try:
			staging_reward, wasted_mass, staging_events = self._evaluate_staging(observation)
		except Exception:
			if self.client.vessel_exists():
				raise
			return self.end_failed_episode("spacecraft_destroyed")
		reward = self.calculate_reward(observation, staging_reward, wasted_mass)
		terminated, truncated, info = self.check_episode_end(observation)
		info["task"] = self.task
		info["atmosphere_exit_achieved"] = self.atmosphere_exit_achieved

		terminal_reward = self.outcome_reward(info["end_reason"])
		reward += terminal_reward
		self.reward_components["outcome_reward"] = terminal_reward
		saved_fuel_reward = 0.0
		if info["success"]:
			fraction = min(1.0, max(0.0, self.propellant_mass(observation) / self.initial_propellant_mass))
			saved_fuel_reward = self.saved_fuel_weight * fraction
			reward += saved_fuel_reward
		self.reward_components["saved_fuel_reward"] = saved_fuel_reward

		self.episode_finished = terminated or truncated

		state = self.encode_observation(observation)
		self.last_state = state.copy()

		info["telemetry"] = observation.copy()
		info["reward_components"] = self.reward_components.copy()
		info["stage_activated"] = activated_stage
		info["staging_events"] = staging_events
		info["discarded_propellant_mass"] = wasted_mass

		return state, reward, terminated, truncated, info

	def propellant_mass(self, observation):
		return sum(observation[field] * density for field, density in self.fuel_densities.items())

	def orbit_progress(self, observation):
		apoapsis = observation["apoapsis"]
		periapsis = observation["periapsis"]

		if not (math.isfinite(apoapsis) and math.isfinite(periapsis)):
			return None

		height = self.orbit_altitude
		# Keep useful periapsis progress below sea level, while bounding the score.
		apoapsis_progress = min(1.0, max(0.0, apoapsis / height))
		periapsis_progress = min(1.0, max(0.0, (periapsis + self.radius_scale) / (height + self.radius_scale)))
		return 0.5 * (apoapsis_progress + periapsis_progress)

	def calculate_task_reward(self, observation):
		self._validate_task()
		if self.task == "orbit":
			return self.calculate_orbit_reward(observation)
		raise NotImplementedError(f"Progress rewards are undefined for task: {self.task}")

	def calculate_orbit_reward(self, observation):
		current_progress = self.orbit_progress(observation)

		orbit_reward = 0.0
		if current_progress is not None and self.previous_orbit_progress is not None:
			orbit_reward = self.orbit_weight * (current_progress - self.previous_orbit_progress)
		current_band = self.altitude_band(observation["altitude"])

		new_bands = max(0, current_band - self.highest_altitude_band)
		altitude_reward = new_bands * self.altitude_reward_per_step

		self.highest_altitude_band = max(self.highest_altitude_band, current_band)
		if current_progress is not None:
			self.previous_orbit_progress = current_progress

		reached = sum(observation["altitude"] >= altitude for altitude in self.ascent_milestone_altitudes)
		new_milestones = max(0, reached - self._ascent_milestones_reached)
		self._ascent_milestones_reached = max(self._ascent_milestones_reached, reached)

		atmosphere_exit_reward = 0.0
		if not self.atmosphere_exit_achieved and self.atmosphere_depth > 0.0 and self.previous_observation["altitude"] <= self.atmosphere_depth < observation["altitude"]:
			self.atmosphere_exit_achieved = True
			atmosphere_exit_reward = self.atmosphere_exit_bonus

		return {
			"orbit_reward": orbit_reward,
			"altitude_reward": altitude_reward,
			"ascent_milestone_reward": new_milestones * self.ascent_milestone_bonus,
			"atmosphere_exit_reward": atmosphere_exit_reward,
		}

	def calculate_reward(self, observation, staging_reward=0.0, wasted_propellant_mass=0.0):
		dt = float(observation["dt"])
		if not math.isfinite(dt) or dt < 0:
			raise ValueError("Received invalid elapsed game time.")
		task_rewards = self.calculate_task_reward(observation)
		# fuel_weight now prices discarded propellant, not ordinary engine burning.
		fuel_penalty = self.fuel_weight * wasted_propellant_mass / self.initial_propellant_mass
		time_penalty = self.time_weight * dt / self.episode_seconds

		angular_speed_squared = sum(float(value) ** 2 for value in observation["angular_velocity"])

		rotation_severity = min(angular_speed_squared / self.angular_speed_scale**2, 1.0)

		rotation_penalty = self.rotation_weight * rotation_severity * dt / self.episode_seconds

		self.reward_components = {
			**task_rewards,
			"wasted_fuel_penalty": fuel_penalty,
			"time_penalty": time_penalty,
			"staging_reward": staging_reward,
			"rotation_penalty": rotation_penalty,
			"outcome_reward": 0.0,
			"saved_fuel_reward": 0.0,
		}

		self.previous_observation = observation.copy()
		return sum(task_rewards.values()) - fuel_penalty - time_penalty + staging_reward - rotation_penalty

	def check_episode_end(self, observation):
		self._validate_task()
		if self.task == "orbit":
			return self.check_orbit_episode_end(observation)
		raise NotImplementedError(f"Ending conditions are undefined for task: {self.task}")

	def check_orbit_episode_end(self, observation):
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
			and apoapsis > self.orbit_altitude
			and periapsis > self.orbit_altitude
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
		return self.reset_from_observation(observation)

	def reset_from_observation(self, observation, metadata=None):
		self.initialize_observation_scales(observation)
		self.highest_altitude_band = self.altitude_band(observation["altitude"])
		self.initial_altitude_band = self.highest_altitude_band
		self.atmosphere_exit_achieved = False
		self.stage_requested = False
		self.staging_reward_paid = 0.0
		self.rewarded_stages = set()
		self.last_stage_ut = None
		self.pending_stages = []
		self.discarded_propellant_mass = 0.0
		self.episode_start_ut = None
		self.has_been_airborne = False
		self.episode_finished = False
		self.orbit_valid_since = None
		self.surface_valid_since = None
		self.stationary_since = None

		self.initial_propellant_mass = self.propellant_mass(observation)
		if metadata is not None:
			for name in self._CHECKPOINT_FIELDS:
				setattr(self, name, metadata[name])

			self.resource_scales = dict(metadata["resource_scales"])
			self.rewarded_stages = set(metadata["rewarded_stages"])
			ut = float(observation["ut"])
			self.episode_start_ut = ut - metadata["mission_elapsed"]
			self.has_been_airborne = True

			remaining = metadata["cooldown_remaining"]
			if remaining > 0:
				self.last_stage_ut = ut - (self.staging_cooldown_seconds - remaining)

		if not math.isfinite(self.initial_propellant_mass) or self.initial_propellant_mass <= 0:
			raise ValueError("The spacecraft has no valid measured propellant.")

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
				value /= self.orbit_altitude

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

		features["altitude"] = observation["altitude"] / self.orbit_altitude
		features["surface_altitude"] = observation["surface_altitude"] / self.orbit_altitude
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

		# Append context so the previous 47 feature names/order remain intact.
		for task in self.TASKS:
			features[f"task_{task}"] = float(self.task == task)
		features["orbit_target_altitude_fraction"] = self.orbit_altitude / self.radius_scale
		features["highest_altitude_band_fraction"] = self.highest_altitude_band / self.max_altitude_band
		features["staging_cooldown_fraction"] = self.staging_cooldown_remaining(observation["ut"]) / self.staging_cooldown_seconds
		features["staging_reward_remaining_fraction"] = max(0.0, 1.0 - self.staging_reward_paid / self.staging_reward_budget) if self.staging_reward_budget > 0 else 0.0
		features["pending_staging_fraction"] = len(self.pending_stages) / self.initial_stage_count

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
			"wasted_fuel_penalty": 0.0,
			"saved_fuel_reward": 0.0,
			"time_penalty": 0.0,
			"altitude_reward": 0,
			"ascent_milestone_reward": 0.0,
			"atmosphere_exit_reward": 0.0,
			"rotation_penalty": 0.0,
			"staging_reward": staging_reward,
			"outcome_reward": outcome_reward,
		}

		info = {
			"elapsed": 0.0 if self.episode_start_ut is None else self.client.sc.ut - self.episode_start_ut,
			"task": self.task,
			"success": False,
			"atmosphere_exit_achieved": self.atmosphere_exit_achieved,
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

	def checkpoint_metadata(self):
		observation = self.previous_observation
		ut = float(observation["ut"])
		vessel = self.client.vessel
		metadata = {name: getattr(self, name) for name in self._CHECKPOINT_FIELDS}
		metadata["resource_scales"] = dict(self.resource_scales)
		metadata["rewarded_stages"] = sorted(self.rewarded_stages)
		metadata["cooldown_remaining"] = self.staging_cooldown_remaining(ut)
		metadata["mission_elapsed"] = 0.0 if self.episode_start_ut is None else max(0.0, ut - self.episode_start_ut)
		metadata["expected"] = {
			"vessel": vessel.name,
			"body": vessel.orbit.body.name,
			"ut": ut,
			"stage": int(observation["stage"]),
			"altitude": float(observation["altitude"]),
			"fuel": {resource: float(observation[resource]) for resource in self.PROPELLANTS},
		}
		return metadata
