import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from agent import PPOAgent
from client import KSPClient
from environment import KSPEnvironment


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("--mode", choices=["train", "test"], default="train")
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--updates", type=int, default=10, help="Updates to run in this session; additional updates when resuming.")
	parser.add_argument("--rollout-steps", type=int, default=256)
	parser.add_argument("--episode-seconds", type=float, default=200.0)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--checkpoint", default="ppo_checkpoint.pt")
	parser.add_argument("--resume", action="store_true", help="Continue from --checkpoint, restoring its training settings.")

	parser.add_argument("--wandb", action="store_true")
	parser.add_argument("--wandb-project", default="ksp-rl")
	parser.add_argument("--wandb-entity", default=None)
	parser.add_argument("--wandb-name", default=None)
	parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
	parser.add_argument("--wandb-run-id", default=None, help="Existing W&B run ID, needed for older checkpoints without one.")

	args = parser.parse_args()

	if min(args.updates, args.rollout_steps, args.epochs, args.batch_size) <= 0:
		parser.error("Updates, rollout steps, epochs and batch size must be positive.")
	if args.wandb_run_id and not (args.resume and args.wandb):
		parser.error("--wandb-run-id requires --resume and --wandb.")
	if args.resume and args.wandb and args.wandb_mode != "online":
		parser.error("This resume command requires --wandb-mode online to continue the existing W&B run.")
	if args.resume and args.mode != "train":
		parser.error("--resume is only supported in training mode.")

	return args


def train(args):
	client = None
	run = None
	completed = False
	checkpoint = None
	checkpoint_path = Path("models/", args.checkpoint)
	if args.resume:
		checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
		for name in ("model", "optimizer", "observation_size", "completed_updates", "total_steps"):
			if name not in checkpoint:
				raise ValueError(f"Checkpoint is missing required field: {name}")
		for name in ("seed", "rollout_steps", "episode_seconds", "epochs", "batch_size", "learning_rate", "gamma", "gae_lambda"):
			if name in checkpoint.get("config", {}):
				setattr(args, name, checkpoint["config"][name])
	start_update = int(checkpoint["completed_updates"]) if checkpoint is not None else 0
	end_update = start_update + args.updates
	total_steps = int(checkpoint["total_steps"]) if checkpoint is not None else 0
	episode_count = int(checkpoint.get("episode_count", 0)) if checkpoint is not None else 0
	successful_episodes = int(checkpoint.get("successful_episodes", 0)) if checkpoint is not None else 0
	wandb_metadata = dict(checkpoint.get("wandb") or {}) if checkpoint is not None else {}
	wandb_step_offset = 0

	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable.")

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	try:
		if args.wandb:
			import wandb  # type: ignore

			saved_run_id = wandb_metadata.get("id")
			if saved_run_id and args.wandb_run_id and saved_run_id != args.wandb_run_id:
				raise ValueError("--wandb-run-id differs from the run saved in the checkpoint.")
			run_id = saved_run_id or args.wandb_run_id
			if args.resume and not run_id and checkpoint.get("config", {}).get("wandb", False):
				raise ValueError("This older checkpoint has no W&B ID. Supply --wandb-run-id from the existing run URL.")
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
			wandb_step_offset = max(0, int(run.step) - total_steps)

		client = KSPClient(name="Custom PPO Training")

		env = KSPEnvironment(client, task="orbit", episode_seconds=args.episode_seconds)

		if checkpoint is not None:
			if checkpoint.get("action_size", 5) != 5:
				raise ValueError("Checkpoint does not use this environment's five-action layout.")

		agent = PPOAgent(
			observation_size=len(env.last_state),
			action_size=checkpoint.get("action_size", 5) if checkpoint is not None else 5,
			hidden_size=checkpoint.get("hidden_size", 64) if checkpoint is not None else 64,
			device=args.device,
			learning_rate=args.learning_rate,
		)
		if checkpoint is not None:
			agent.load_state_dict(checkpoint["model"])
			agent.optimizer.load_state_dict(checkpoint["optimizer"])
			args.learning_rate = agent.optimizer.param_groups[0]["lr"]
			state = env.reset()
			rng = checkpoint.get("rng", {})
			if "torch" in rng:
				torch.set_rng_state(rng["torch"])
			if "numpy" in rng:
				numpy_rng = rng["numpy"]
				np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), *numpy_rng[2:]))
			if agent.device.type == "cuda" and rng.get("cuda"):
				if len(rng["cuda"]) == torch.cuda.device_count():
					torch.cuda.set_rng_state_all(rng["cuda"])
				else:
					print("CUDA device count changed; CUDA random states were not restored.")
			print(f"Resumed {checkpoint_path}: {start_update} completed updates, {total_steps} environment steps.")
			if "episode_count" not in checkpoint:
				print("Older checkpoint has no episode totals; episode count and success rate restart from zero.")
		else:
			state = env.last_state.copy()

		needs_reset = False

		episode_return = 0.0
		episode_steps = 0

		checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		collection_start = time.perf_counter()

		reset_seconds = 0.0
		env_step_seconds = 0.0
		action_inference_seconds = 0.0
		simulated_seconds = 0.0

		reset_count = 0
		bootstrap_evaluations = 0

		previous_ut = client.telemetry["ut"]()

		elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0)) if checkpoint is not None else 0.0
		training_start = time.monotonic() - elapsed_seconds

		print("Training starts")

		with tqdm(total=args.updates * args.rollout_steps, desc="Collecting", unit="step", dynamic_ncols=True) as progress:
			for update_index in range(start_update, end_update):
				rollout = {name: [] for name in ("states", "raw_actions", "old_log_probabilities", "rewards", "values", "next_values", "terminated", "truncated")}
				finished_episodes = []

				collection_start = time.perf_counter()

				reset_seconds = 0.0
				env_step_seconds = 0.0
				action_inference_seconds = 0.0
				simulated_seconds = 0.0

				reset_count = 0
				bootstrap_evaluations = 0

				previous_ut = client.telemetry["ut"]()

				# ----------------------------------------------------------
				# Collect a rollout with a fixed policy.
				# ----------------------------------------------------------
				for _ in range(args.rollout_steps):
					if needs_reset:
						reset_start = time.perf_counter()

						state = env.reset()

						reset_seconds += time.perf_counter() - reset_start
						reset_count += 1
						needs_reset = False

						previous_ut = env.previous_observation["ut"]

					inference_start = time.perf_counter()

					result = agent.act(state)

					action_inference_seconds += time.perf_counter() - inference_start

					if rollout["next_values"] and rollout["next_values"][-1] is None:
						rollout["next_values"][-1] = result["value"]

					step_start = time.perf_counter()

					next_state, reward, terminated, truncated, info = env.step(result["action"])

					env_step_seconds += time.perf_counter() - step_start

					if not info.get("telemetry_stale", False):
						current_ut = info["telemetry"]["ut"]

						simulated_seconds += max(0.0, current_ut - previous_ut)

						previous_ut = current_ut

					if terminated:
						next_value = 0.0

					elif truncated:
						next_value = agent.estimate_value(next_state)
						bootstrap_evaluations += 1

					else:
						next_value = None

					rollout["states"].append(state.copy())
					rollout["raw_actions"].append(result["raw_action"].copy())
					rollout["old_log_probabilities"].append(result["log_probability"])
					rollout["rewards"].append(float(reward))
					rollout["values"].append(result["value"])
					rollout["next_values"].append(next_value)
					rollout["terminated"].append(bool(terminated))
					rollout["truncated"].append(bool(truncated))

					state = next_state.copy()
					needs_reset = terminated or truncated

					total_steps += 1
					if run is not None:
						metrics = {"rollout/reward": reward, "train/environment_steps": total_steps, "train/elapsed_seconds": time.monotonic() - training_start}
						for name, value in info["reward_components"].items():
							metrics[f"reward/{name}"] = -float(value) if name.endswith("_penalty") else float(value)
						if not info.get("telemetry_stale", False):
							telemetry = info.get("telemetry", {})
							for name in ("altitude", "apoapsis", "periapsis", "mass"):
								if name in telemetry:
									metrics[f"telemetry/{name}"] = telemetry[name]
						run.log(metrics, step=total_steps + wandb_step_offset, commit=False)

					episode_steps += 1
					episode_return += float(reward)

					if needs_reset:
						episode_count += 1

						successful_episodes += int(info.get("success", False))
						if run is not None:
							run.log(
								{"episode/return": episode_return, "episode/length": episode_steps, "episode/success_rate": successful_episodes / episode_count},
								step=total_steps + wandb_step_offset,
								commit=False,
							)

						finished_episodes.append({"return": episode_return, "length": episode_steps, "success": bool(info.get("success", False))})

						tqdm.write(f"Episode {episode_count}: {info['end_reason']} | Return={episode_return:.4f} | Steps={episode_steps}")

						episode_return = 0.0
						episode_steps = 0

					progress.set_postfix(update=f"{update_index + 1}/{end_update}", episodes=episode_count, reward=f"{reward:.4f}", refresh=False)
					progress.update(1)

				collection_seconds = time.perf_counter() - collection_start

				# ----------------------------------------------------------
				# Pause for bootstrapping, GAE, optimization and logging.
				# ----------------------------------------------------------
				client.conn.krpc.paused = True

				try:
					progress.set_description("Updating PPO")

					if agent.device.type == "cuda":
						torch.cuda.synchronize(agent.device)

					update_start = time.perf_counter()

					if rollout["next_values"][-1] is None:
						rollout["next_values"][-1] = agent.estimate_value(state)
						bootstrap_evaluations += 1

					advantages, returns = agent.compute_gae(
						rewards=rollout["rewards"],
						values=rollout["values"],
						next_values=rollout["next_values"],
						terminated=rollout["terminated"],
						truncated=rollout["truncated"],
						gamma=args.gamma,
						gae_lambda=args.gae_lambda,
					)

					metrics = agent.update(
						states=rollout["states"],
						raw_actions=rollout["raw_actions"],
						old_log_probabilities=(rollout["old_log_probabilities"]),
						advantages=advantages,
						returns=returns,
						epochs=args.epochs,
						batch_size=args.batch_size,
					)

					if agent.device.type == "cuda":
						torch.cuda.synchronize(agent.device)

					update_seconds = time.perf_counter() - update_start

					numpy_rng = np.random.get_state()
					temporary_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
					torch.save(
						{
							"model": agent.state_dict(),
							"optimizer": agent.optimizer.state_dict(),
							"observation_size": agent.observation_size,
							"observation_names": env.observation_names,
							"action_size": agent.log_std.numel(),
							"hidden_size": agent.actor[0].out_features,
							"total_steps": total_steps,
							"completed_updates": update_index + 1,
							"episode_count": episode_count,
							"successful_episodes": successful_episodes,
							"elapsed_seconds": time.monotonic() - training_start,
							"wandb": wandb_metadata,
							"rng": {
								"torch": torch.get_rng_state(),
								"cuda": torch.cuda.get_rng_state_all() if agent.device.type == "cuda" else [],
								"numpy": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
							},
							"config": vars(args),
						},
						temporary_checkpoint,
					)
					temporary_checkpoint.replace(checkpoint_path)

					log_data = {
						"train/update": update_index + 1,
						"train/episodes": episode_count,
						"train/environment_steps": total_steps,
						"train/elapsed_seconds": time.monotonic() - training_start,
						"train/update_seconds": update_seconds,
						"rollout/reward_mean": float(np.mean(rollout["rewards"])),
						"rollout/reset_count": reset_count,
						"rollout/bootstrap_evaluations": (bootstrap_evaluations),
						**{f"ppo/{name}": value for name, value in metrics.items() if value is not None},
					}

					if run is not None:
						run.log(log_data, step=total_steps + wandb_step_offset)

					tqdm.write(
						f"Update {update_index + 1}: "
						f"policy={metrics['policy_loss']:.4f}, "
						f"value={metrics['value_loss']:.4f}, "
						f"KL={metrics['approx_kl']:.5f}, "
						f"collection={collection_seconds:.2f}s, "
						f"reset={reset_seconds:.2f}s, "
						f"update={update_seconds:.2f}s"
					)

				finally:
					client.conn.krpc.paused = False
					progress.set_description("Collecting")
		completed = True

	except KeyboardInterrupt:
		tqdm.write("Training interrupted; latest completed update is saved.")

	finally:
		try:
			if client is not None:
				try:
					client.stop_controls()
				finally:
					try:
						client.conn.krpc.paused = False
						client.revert_to_launch(delay=5)
					finally:
						client.close()
		finally:
			if run is not None:
				run.finish(exit_code=0 if completed else 1)


def test():
	connection = KSPClient(name="Environment Reward Test")

	try:
		env = KSPEnvironment(connection, task="orbit", episode_seconds=10)
		state = env.reset()
		agent = PPOAgent(observation_size=len(state))
		result = agent.act(state)
		print(result)
		print("Launching in 5 seconds. Switch to KSP.")
		time.sleep(5)
		for episode in range(2):
			print(f"\nEpisode {episode + 1}")

			while True:
				state, reward, terminated, truncated, info = env.step(result["action"])
				observation = info["telemetry"]

				print(f"State size: {len(state)} | Altitude: {observation['altitude']:.1f}m | Reward: {reward:+.6f}")

				if terminated or truncated:
					print(f"Episode ended: {info['end_reason']}")
					break

			state = env.reset()
			print("Reset complete.")
	except KeyboardInterrupt:
		print("Test interrupted.")


def main():
	args = parse_args()
	if args.mode == "test":
		test()
	else:
		train(args)


if __name__ == "__main__":
	main()
