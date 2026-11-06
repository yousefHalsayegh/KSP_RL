"""Shared-policy PPO training using already connected KSP workers."""

import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from agent import PPOAgent
from client import KSPClient
from environment import KSPEnvironment


def restore_training_state(agent, checkpoint, observation_names, expand_observations=False):
	loader = getattr(agent, "load_training_state", None)
	if loader is not None:
		return loader(checkpoint, observation_names, expand_observations=expand_observations)

	# Older local agent.py versions expose only standard PyTorch loading methods.
	names = list(observation_names)
	if int(checkpoint["observation_size"]) != agent.observation_size:
		raise ValueError("Checkpoint observation size changed. Restore the agent's observation-migration method before expanding an older checkpoint.")
	saved_names = checkpoint.get("observation_names")
	if saved_names is not None and list(saved_names) != names:
		raise ValueError("Checkpoint observation names/order do not match the environment.")
	agent.load_state_dict(checkpoint["model"])
	agent.optimizer.load_state_dict(checkpoint["optimizer"])
	return False


def reset_worker(worker):
	worker["connection"].krpc.paused = False
	state = worker["env"].reset()
	worker["client"].manual_control()
	if not np.all(np.isfinite(state)):
		raise ValueError(f"Worker {worker['index']} returned an invalid reset state.")
	return state.copy()


def step_worker(worker, action):
	state, reward, terminated, truncated, info = worker["env"].step(action)
	if not np.all(np.isfinite(state)) or not np.isfinite(reward):
		raise ValueError(f"Worker {worker['index']} returned an invalid transition.")
	# Preserve the final observation separately from the reset observation.
	final_state = state.copy()
	following_state = reset_worker(worker) if terminated or truncated else state.copy()
	return final_state, float(reward), bool(terminated), bool(truncated), info, following_state


def collect_rollout(agent, workers, executor, steps, on_step=None):
	count = len(workers)
	size = agent.observation_size
	actions = agent.log_std.numel()
	rollout = {
		"states": np.empty((steps, count, size), dtype=np.float32),
		"raw_actions": np.empty((steps, count, actions), dtype=np.float32),
		"old_log_probabilities": np.empty((steps, count), dtype=np.float32),
		"rewards": np.empty((steps, count), dtype=np.float32),
		"values": np.empty((steps, count), dtype=np.float32),
		"next_values": np.full((steps, count), np.nan, dtype=np.float32),
		"terminated": np.empty((steps, count), dtype=bool),
		"truncated": np.empty((steps, count), dtype=bool),
	}
	agent.eval()
	pending = {}

	def submit(i, t):
		worker = workers[i]
		decision = agent.act(worker["state"])
		rollout["states"][t, i] = worker["state"]
		rollout["raw_actions"][t, i] = decision["raw_action"]
		rollout["old_log_probabilities"][t, i] = decision["log_probability"]
		rollout["values"][t, i] = decision["value"]
		if t > 0 and np.isnan(rollout["next_values"][t - 1, i]):
			rollout["next_values"][t - 1, i] = decision["value"]
		future = executor.submit(step_worker, worker, decision["action"])
		pending[future] = (i, t)

	for i, worker in enumerate(workers):
		worker["connection"].krpc.paused = False
		submit(i, 0)

	while pending:
		ready, _ = wait(pending, return_when=FIRST_COMPLETED)
		rewards, infos, finished = [], [], []
		for future in ready:
			i, t = pending.pop(future)
			worker = workers[i]
			final_state, reward, terminated, truncated, info, following_state = future.result()
			rollout["rewards"][t, i] = reward
			rollout["terminated"][t, i] = terminated
			rollout["truncated"][t, i] = truncated
			if terminated:
				rollout["next_values"][t, i] = 0.0
			elif truncated:
				# Bootstrap from the final state, never from the next episode.
				rollout["next_values"][t, i] = agent.estimate_value(final_state)
			worker["state"] = following_state
			worker["episode_return"] += reward
			worker["episode_steps"] += 1
			rewards.append(reward)
			infos.append(info)
			if terminated or truncated:
				finished.append(
					{"worker": worker["index"], "return": worker["episode_return"], "length": worker["episode_steps"], "success": bool(info.get("success", False)), "reason": info.get("end_reason")}
				)
				worker["episode_return"] = 0.0
				worker["episode_steps"] = 0

			if t + 1 < steps:
				# Give this worker its next action without waiting for other games.
				submit(i, t + 1)
			else:
				# Its rollout is complete. Hold this game at the update boundary
				# while other workers finish collecting their own quotas.
				worker["connection"].krpc.paused = True

		if on_step is not None:
			on_step(np.asarray(rewards, dtype=np.float32), infos, finished)

	return rollout


def prepare_update(agent, workers, rollout, gamma, gae_lambda):
	# Caller pauses all games before this function and the optimizer update.
	for i, worker in enumerate(workers):
		if np.isnan(rollout["next_values"][-1, i]):
			rollout["next_values"][-1, i] = agent.estimate_value(worker["state"])

	advantages, returns = [], []
	for i in range(len(workers)):
		worker_advantages, worker_returns = agent.compute_gae(
			rewards=rollout["rewards"][:, i],
			values=rollout["values"][:, i],
			next_values=rollout["next_values"][:, i],
			terminated=rollout["terminated"][:, i],
			truncated=rollout["truncated"][:, i],
			gamma=gamma,
			gae_lambda=gae_lambda,
		)
		advantages.append(worker_advantages)
		returns.append(worker_returns)

	# Time-major order must match across every flattened array.
	return {
		"states": rollout["states"].reshape(-1, agent.observation_size),
		"raw_actions": rollout["raw_actions"].reshape(-1, agent.log_std.numel()),
		"old_log_probabilities": rollout["old_log_probabilities"].reshape(-1),
		"advantages": torch.stack(advantages, dim=1).reshape(-1),
		"returns": torch.stack(returns, dim=1).reshape(-1),
	}


def save_checkpoint(path, agent, env, args, counters, wandb_metadata):
	numpy_rng = np.random.get_state()
	payload = {
		"model": agent.state_dict(),
		"optimizer": agent.optimizer.state_dict(),
		"observation_size": agent.observation_size,
		"observation_names": env.observation_names,
		"action_size": agent.log_std.numel(),
		"hidden_size": agent.actor[0].out_features,
		"wandb": wandb_metadata,
		"config": vars(args).copy(),
		"rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if agent.device.type == "cuda" else [], "numpy": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]]},
		**counters,
	}
	temporary = path.with_suffix(path.suffix + ".tmp")
	torch.save(payload, temporary)
	temporary.replace(path)


def train(args, workers):
	"""Train one policy; connections remain owned by the launcher."""
	if not workers:
		raise ValueError("At least one connected worker is required.")
	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable.")
	if min(args.updates, args.rollout_steps, args.epochs, args.batch_size) <= 0:
		raise ValueError("Updates, rollout steps, epochs and batch size must be positive.")

	path = Path(f"models/{args.checkpoint}")
	path.parent.mkdir(parents=True, exist_ok=True)
	checkpoint = torch.load(path, map_location="cpu", weights_only=True) if args.resume else None
	if checkpoint is not None:
		for name in ("model", "optimizer", "observation_size", "completed_updates", "total_steps"):
			if name not in checkpoint:
				raise ValueError(f"Checkpoint is missing {name}.")
		for name in ("seed", "rollout_steps", "episode_seconds", "epochs", "batch_size", "learning_rate", "gamma", "gae_lambda"):
			if name in checkpoint.get("config", {}):
				setattr(args, name, checkpoint["config"][name])
		if checkpoint.get("action_size", 5) != 5:
			raise ValueError("Checkpoint must use five actions.")

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	saved = checkpoint or {}
	total_steps = int(saved.get("total_steps", 0))
	episode_count = int(saved.get("episode_count", 0))
	successful_episodes = int(saved.get("successful_episodes", 0))
	start_update = int(saved.get("completed_updates", 0))
	elapsed = float(saved.get("elapsed_seconds", 0.0))
	wandb_metadata = dict(saved.get("wandb") or {})
	run, completed = None, False
	try:
		for worker in workers:
			client = worker.get("client")
			if client is None:
				client = KSPClient(connection=worker["connection"])
				worker["client"] = client
			worker["connection"].krpc.paused = False
			worker["env"] = KSPEnvironment(client, task="orbit", episode_seconds=args.episode_seconds)
			worker["state"] = reset_worker(worker)
			worker["episode_return"], worker["episode_steps"] = 0.0, 0

		env = workers[0]["env"]
		for worker in workers:
			if worker["env"].observation_names != env.observation_names:
				raise ValueError("Workers have different observation layouts.")
		agent = PPOAgent(observation_size=len(env.last_state), action_size=5, hidden_size=saved.get("hidden_size", 64), device=args.device, learning_rate=args.learning_rate)
		if checkpoint is not None:
			expanded = restore_training_state(agent, checkpoint, env.observation_names, expand_observations=args.expand_observations)
			if expanded:
				backup = path.with_name(path.stem + ".before-stage-inputs" + path.suffix)
				if not backup.exists():
					shutil.copy2(path, backup)
			args.learning_rate = agent.optimizer.param_groups[0]["lr"]
			rng = saved.get("rng", {})
			if "torch" in rng:
				torch.set_rng_state(rng["torch"])
			if "numpy" in rng:
				state = rng["numpy"]
				np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
			if agent.device.type == "cuda" and rng.get("cuda"):
				if len(rng["cuda"]) == torch.cuda.device_count():
					torch.cuda.set_rng_state_all(rng["cuda"])

		wandb_offset = 0
		if args.wandb:
			import wandb

			run_id = wandb_metadata.get("id") or args.wandb_run_id
			if run_id and not args.resume:
				raise ValueError("Continuing a W&B run requires --resume.")
			if args.resume and args.wandb_mode != "online":
				raise ValueError("W&B resume requires online mode.")
			if args.resume and not run_id and saved.get("config", {}).get("wandb", False):
				raise ValueError("Checkpoint has no W&B ID; supply --wandb-run-id.")
			run = wandb.init(
				project=wandb_metadata.get("project") or args.wandb_project,
				entity=wandb_metadata.get("entity") or args.wandb_entity,
				name=args.wandb_name,
				mode=args.wandb_mode,
				config=None if run_id else vars(args),
				id=run_id,
				resume="must" if run_id else None,
			)
			wandb_metadata = {"id": run.id, "project": run.project, "entity": run.entity}
			wandb_offset = max(0, int(run.step) - total_steps)

		training_start = time.monotonic() - elapsed
		samples_per_update = args.rollout_steps * len(workers)
		print(f"Shared {agent.device} policy; {samples_per_update} transitions per update.")
		with ThreadPoolExecutor(max_workers=len(workers)) as executor, tqdm(total=args.updates * samples_per_update, desc="Collecting", unit="transition") as progress:
			for update_index in range(start_update, start_update + args.updates):
				collection_start = time.perf_counter()
				finished_episodes = []

				def on_step(rewards, infos, finished):
					nonlocal total_steps, episode_count, successful_episodes
					total_steps += len(rewards)
					finished_episodes.extend(finished)
					log = {"train/environment_steps": total_steps, "train/elapsed_seconds": time.monotonic() - training_start, "rollout/reward": float(np.mean(rewards))}
					names = set().union(*(info.get("reward_components", {}) for info in infos))
					for name in names:
						value = np.mean([info.get("reward_components", {}).get(name, 0.0) for info in infos])
						log[f"reward/{name}"] = -float(value) if name.endswith("_penalty") else float(value)
					for episode in finished:
						episode_count += 1
						successful_episodes += int(episode["success"])
						tqdm.write(f"Worker {episode['worker']} episode: {episode['reason']} | return={episode['return']:.3f} | length={episode['length']}")
					if finished:
						log.update(
							{
								"episode/return_mean": float(np.mean([e["return"] for e in finished])),
								"episode/length_mean": float(np.mean([e["length"] for e in finished])),
								"episode/success_rate": successful_episodes / episode_count,
								"train/episodes": episode_count,
							}
						)
					for name in ("altitude", "apoapsis", "periapsis"):
						values = [
							float(info["telemetry"][name]) for info in infos if not info.get("telemetry_stale", False) and name in info.get("telemetry", {}) and np.isfinite(info["telemetry"][name])
						]

						if values:
							log[f"telemetry/{name}"] = float(np.mean(values))
					if run is not None:
						run.log(log, step=total_steps + wandb_offset)
					progress.update(len(rewards))

				rollout = collect_rollout(agent, workers, executor, args.rollout_steps, on_step)
				collection_seconds = time.perf_counter() - collection_start
				try:
					for worker in workers:
						worker["connection"].krpc.paused = True
					progress.set_description("Updating PPO")
					if agent.device.type == "cuda":
						torch.cuda.synchronize(agent.device)
					update_start = time.perf_counter()
					batch = prepare_update(agent, workers, rollout, args.gamma, args.gae_lambda)
					metrics = agent.update(**batch, epochs=args.epochs, batch_size=args.batch_size)
					if agent.device.type == "cuda":
						torch.cuda.synchronize(agent.device)
					update_seconds = time.perf_counter() - update_start
					save_checkpoint(
						path,
						agent,
						env,
						args,
						{
							"total_steps": total_steps,
							"completed_updates": update_index + 1,
							"episode_count": episode_count,
							"successful_episodes": successful_episodes,
							"elapsed_seconds": time.monotonic() - training_start,
							"worker_count": len(workers),
						},
						wandb_metadata,
					)
					if run is not None:
						run.log(
							{
								"train/update": update_index + 1,
								"train/environment_steps": total_steps,
								"train/elapsed_seconds": time.monotonic() - training_start,
								"train/update_seconds": update_seconds,
								"rollout/collection_seconds": collection_seconds,
								"rollout/reward_mean": float(rollout["rewards"].mean()),
								"rollout/episodes_finished": len(finished_episodes),
								**{f"ppo/{name}": value for name, value in metrics.items() if value is not None},
							},
							step=total_steps + wandb_offset + 1,
						)
						# Reserve the history step used by the update log.
						wandb_offset += 1
					tqdm.write(f"Update {update_index + 1}: samples={samples_per_update} | collection={collection_seconds:.1f}s | update={update_seconds:.2f}s")
				finally:
					for worker in workers:
						worker["connection"].krpc.paused = False
					progress.set_description("Collecting")
		completed = True
	except KeyboardInterrupt:
		tqdm.write("Interrupted; the latest completed-update checkpoint is preserved.")
	finally:
		for worker in workers:
			client = worker.get("client")
			if client is not None:
				try:
					worker["connection"].krpc.paused = False
					client.stop_controls()
					if client.sc.can_revert_to_launch():
						client.reset()
				except Exception as error:
					tqdm.write(f"Worker {worker['index']} cleanup failed: {error}")
		if run is not None:
			run.finish(exit_code=0 if completed else 1)
