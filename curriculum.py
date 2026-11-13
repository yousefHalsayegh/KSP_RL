import copy
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4


class SaveUnavailable(RuntimeError):
	"""The live flight cannot be captured at this crossing."""


@dataclass
class Flight:
	worker: int
	generation: int
	phase: str
	start: dict | None
	episode_id: str = field(default_factory=lambda: uuid4().hex)
	previous_altitude: float | None = None
	reached: set = field(default_factory=set)
	pending_captures: set[int] = field(default_factory=set)
	next_capture_at: float = 0.0
	finished: bool = False
	prefix_return: float = 0.0


class AltitudeCurriculum:
	def __init__(self, directory, *, pad_workers=(0, 1), step_m=1000, ceiling_m=70000, window=100, promote=0.80, demote=0.60, probe_after=200):
		if step_m <= 0 or ceiling_m < step_m or ceiling_m % step_m:
			raise ValueError("The ceiling must be a positive multiple of step_m.")
		if not 0 < demote < promote <= 1 or window <= 0 or probe_after < window:
			raise ValueError("Invalid curriculum thresholds or attempt counts.")
		self.config = dict(pad_workers=list(pad_workers), step_m=step_m, ceiling_m=ceiling_m, window=window, promote=promote, demote=demote, probe_after=probe_after)
		self.targets = tuple(range(step_m, ceiling_m + 1, step_m))
		self.directory = Path(directory).resolve()
		self.directory.mkdir(parents=True, exist_ok=True)
		self._lock = RLock()
		self._logger = logging.Logger("ksp.curriculum")
		self._handler = logging.FileHandler(self.directory / "curriculum.jsonl", encoding="utf-8")
		self._logger.addHandler(self._handler)
		self._generation = 0
		self._phase = "advance"
		self._active = None
		self._promoted = {}
		self._candidates = {}
		self._recent = deque(maxlen=window)
		self._attempts = 0

	def close(self):
		self._logger.removeHandler(self._handler)
		self._handler.close()

	def _log(self, event, **fields):
		self._logger.info(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event, **fields}, allow_nan=False))

	@property
	def active_altitude(self):
		with self._lock:
			return self._promoted[self._active]["target_m"] if self._active else 0

	def begin(self, worker):
		"""Choose the start before loading it. Keep this token until the episode ends."""
		with self._lock:
			save_id = self._active
			if worker in self.config["pad_workers"]:
				save_id = None
			elif self._phase == "probe":
				save_id = self._promoted[self._active]["parent"]
			start = copy.deepcopy(self._promoted[save_id]) if save_id else None
			flight = Flight(worker, self._generation, self._phase, start)
			self._flight_log("load_requested", flight)
			return flight

	def _flight_log(self, event, flight, **fields):
		self._log(
			event,
			worker=flight.worker,
			episode=flight.episode_id,
			generation=flight.generation,
			phase=flight.phase,
			start_source="checkpoint" if flight.start else "pad",
			start_checkpoint_id=flight.start["id"] if flight.start else None,
			**fields,
		)

	def loaded(self, flight, altitude, *, seconds):
		flight.previous_altitude = float(altitude)
		flight.prefix_return = flight.start["metadata"]["prefix_return"] if flight.start else 0.0
		self._flight_log("load_completed", flight, altitude_m=altitude, seconds=seconds)

	def load_failed(self, flight, error):
		self._flight_log("load_failed", flight, error=str(error))

	def observe(self, flight, *, altitude, viable, episode_return, capture):
		"""Record first viable upward crossings and capture improved candidate saves.

		capture(save_name, destination) writes the .sfs and returns JSON-safe
		reward/observation metadata for that saved state. Raise SaveUnavailable
		for an expected KSP save restriction.
		"""
		if flight.previous_altitude is None or flight.finished:
			raise RuntimeError("observe() requires a loaded, unfinished flight.")
		previous = flight.previous_altitude
		flight.previous_altitude = float(altitude)
		if not viable or altitude <= previous:
			return
		score = flight.prefix_return + float(episode_return)
		if not math.isfinite(score):
			raise ValueError("Candidate return must be finite.")
		crossed = [target for target in self.targets if previous < target <= altitude and target not in flight.reached]
		flight.reached.update(crossed)
		flight.pending_capture.update(crossed)
		flight.pending_capture = {target for target in flight.pending_captures if target <= altitude <= target + 250.0}
		now = time.monotonic()
		if not flight.pending_captures or now < flight.next_capture_at:
			return

		with self._lock:
			if flight.generaton != self._generation or self._phase != "advance":
				return
			candidates = [target for target in sorted(flight.pending_captures) if target > self.active_altitude and score > self._candidates.get(target, {}).get("score", -math.inf)]
		if candidates:
			if self._capture(flight, candidates, altitude, score, capture):
				flight.pending_captures.difference_update(candidates)
			else:
				flight.next_capture_at = time.monotonic() + 1.0

	def _capture(self, flight, targets, altitude, score, capture):
		# One snapshot serves every threshold crossed in this action.
		save_id = f"rl_{max(targets):05d}m_w{flight.worker}_{uuid4().hex}"
		path = self.directory / (save_id + ".sfs")
		try:
			metadata = dict(capture(save_id, path))
			metadata["prefix_return"] = score
			json.dumps(metadata, allow_nan=False)
			if not path.is_file() or not path.stat().st_size:
				raise RuntimeError(f"Capture did not produce a complete save: {path}")
		except SaveUnavailable as error:
			path.unlink(missing_ok=True)
			self._flight_log("capture_unavailable", flight, targets_m=targets, error=str(error))
			return False
		except Exception as error:
			path.unlink(missing_ok=True)
			self._flight_log("capture_failed", flight, targets_m=targets, error=str(error))
			raise

		with self._lock:
			accepted = []
			if flight.generation == self._generation:
				for target in targets:
					if score > self._candidates.get(target, {}).get("score", -math.inf):
						self._candidates[target] = dict(id=save_id, target_m=target, saved_altitude_m=altitude, parent=self._active, score=score, metadata=metadata)
						accepted.append(target)
			if not accepted:
				path.unlink()
			self._flight_log("capture_completed", flight, save_id=save_id, targets_m=accepted, altitude_m=altitude, prefix_return=score)
		return True

	def finish(self, flight, *, episode_return, reason):
		"""Count one completed attempt. A stale flight contributes only to logs."""
		with self._lock:
			if flight.finished:
				raise RuntimeError("Flight was already counted.")
			flight.finished = True
			eligible = flight.generation == self._generation
			if eligible and self._phase == "probe":
				parent = self._promoted[self._active]["parent"]
				source = flight.start["id"] if flight.start else None
				eligible = source == parent
			self._flight_log("episode_finished", flight, eligible=eligible, reached_m=sorted(flight.reached), reason=reason, episode_return=float(episode_return))
			if not eligible:
				return
			self._recent.append(sorted(flight.reached))
			self._attempts += 1
			if len(self._recent) < self.config["window"]:
				return

			active = self.active_altitude
			if self._phase == "probe":
				hits = sum(active in crossed for crossed in self._recent)
				passed = hits / len(self._recent) >= self.config["demote"]
				parent = self._promoted[self._active]["parent"]
				self._log("probe_completed", target_m=active, successes=hits, attempts=len(self._recent), fallback=not passed)
				self._change_phase("advance", self._active if passed else parent, "probe_passed" if passed else "fallback")
				return

			rates = {target: sum(target in crossed for crossed in self._recent) / len(self._recent) for target in self.targets if target > active}
			qualified = [target for target, rate in rates.items() if rate >= self.config["promote"]]
			available = [target for target in qualified if target in self._candidates]
			if available:
				chosen = self._candidates[max(available)]
				self._promoted[chosen["id"]] = chosen
				self._log("promotion", from_m=active, to_m=chosen["target_m"], save_id=chosen["id"], reach_rate=rates[chosen["target_m"]], attempts=len(self._recent), prefix_return=chosen["score"])
				self._change_phase("advance", chosen["id"], "promotion")
			elif self._attempts % 10 == 0 or self._attempts == self.config["window"]:
				self._log("window", active_m=active, attempts=len(self._recent), phase_attempts=self._attempts, reach_rates=rates, qualified_without_save=qualified)
			if not available and not qualified and 0 < active and self._attempts >= self.config["probe_after"]:
				self._change_phase("probe", self._active, "frontier_stalled")

	def _change_phase(self, phase, active, reason):
		self._phase = phase
		self._active = active
		self._generation += 1
		self._recent.clear()
		self._candidates.clear()
		self._attempts = 0
		self._log(
			"phase_changed",
			phase=phase,
			active_m=self.active_altitude,
			reason=reason,
			generation=self._generation,
			highest_promoted_m=max((entry["target_m"] for entry in self._promoted.values()), default=0),
		)

	def state_dict(self):
		"""Embed this in the same atomic .pt as the policy and optimizer."""
		with self._lock:
			return copy.deepcopy(
				dict(
					version=1,
					config=self.config,
					generation=self._generation,
					phase=self._phase,
					active=self._active,
					promoted=self._promoted,
					candidates=list(self._candidates.values()),
					recent=list(self._recent),
					attempts=self._attempts,
				)
			)

	def restore(self, state):
		if state["version"] != 1 or state["config"] != self.config:
			raise ValueError("Saved curriculum version or settings do not match.")
		with self._lock:
			state = copy.deepcopy(state)
			entries = list(state["promoted"].values()) + state["candidates"]
			for entry in entries:
				path = self.directory / (entry["id"] + ".sfs")
				if not path.is_file() or not path.stat().st_size:
					raise FileNotFoundError(f"Curriculum save missing or empty: {path}")
			self._generation = state["generation"]
			self._phase = state["phase"]
			self._active = state["active"]
			self._promoted = state["promoted"]
			self._candidates = {entry["target_m"]: entry for entry in state["candidates"]}
			self._recent.clear()
			self._recent.extend(state["recent"])
			self._attempts = state["attempts"]
			self._log("restored", phase=self._phase, active_m=self.active_altitude, generation=self._generation, window_size=len(self._recent))
