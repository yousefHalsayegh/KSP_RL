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

	parser.add_argument("--updates", type=int, default=10)
	parser.add_argument("--rollout-steps", type=int, default=256)
	parser.add_argument("--episode-seconds", type=float, default=200.0)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--checkpoint", default="ppo_checkpoint.pt")

	parser.add_argument("--wandb", action="store_true")
	parser.add_argument("--wandb-project", default="ksp-rl")
	parser.add_argument("--wandb-entity", default=None)
	parser.add_argument("--wandb-name", default=None)
	parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")

	args = parser.parse_args()

	if min(args.updates, args.rollout_steps, args.epochs, args.batch_size) <= 0:
		parser.error("Updates, rollout steps, epochs and batch size must be positive.")

	return args


def train(args):
	client = None
	run = None
	completed = False

	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable.")

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	try:
		# No W&B import or initialization unless explicitly enabled.
		if args.wandb:
			import wandb  # type: ignore

			run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, mode=args.wandb_mode, config=vars(args))

		client = KSPClient(name="Custom PPO Training")

		env = KSPEnvironment(client, task="orbit", episode_seconds=args.episode_seconds)

		agent = PPOAgent(observation_size=len(env.last_state), device=args.device, learning_rate=args.learning_rate)

		state = env.last_state.copy()
		needs_reset = False

		total_steps = 0
		episode_count = 0
		episode_return = 0.0
		episode_steps = 0

		checkpoint_path = Path(args.checkpoint)
		checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
		collection_start = time.perf_counter()

		reset_seconds = 0.0
		env_step_seconds = 0.0
		action_inference_seconds = 0.0
		simulated_seconds = 0.0

		reset_count = 0
		bootstrap_evaluations = 0

		previous_ut = client.telemetry["ut"]()

		print("Training starts")

		with tqdm(total=args.updates * args.rollout_steps, desc="Collecting", unit="step", dynamic_ncols=True) as progress:
			for update_index in range(args.updates):
				rollout = {name: [] for name in ("states", "raw_actions", "old_log_probabilities", "rewards", "values", "next_values", "terminated", "truncated")}
				finished_episodes = []

				# Timing and throughput measurements for this rollout.
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

						# Revert changes game time, so start a new baseline.
						previous_ut = env.previous_observation["ut"]

					inference_start = time.perf_counter()

					result = agent.act(state)

					action_inference_seconds += time.perf_counter() - inference_start

					# Reuse V(state) as the previous transition's next value.
					if rollout["next_values"] and rollout["next_values"][-1] is None:
						rollout["next_values"][-1] = result["value"]

					step_start = time.perf_counter()

					next_state, reward, terminated, truncated, info = env.step(result["action"])

					env_step_seconds += time.perf_counter() - step_start

					# Measure simulated time without spanning episode resets.
					if not info.get("telemetry_stale", False):
						current_ut = info["telemetry"]["ut"]

						simulated_seconds += max(0.0, current_ut - previous_ut)

						previous_ut = current_ut

					if terminated:
						# True terminal states have zero future value.
						next_value = 0.0

					elif truncated:
						# Evaluate the final state before resetting.
						next_value = agent.estimate_value(next_state)
						bootstrap_evaluations += 1

					else:
						# Filled by the next iteration's agent.act(state).
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
					episode_steps += 1
					episode_return += float(reward)

					if needs_reset:
						episode_count += 1

						finished_episodes.append({"return": episode_return, "length": episode_steps, "success": bool(info.get("success", False))})

						tqdm.write(f"Episode {episode_count}: {info['end_reason']} | Return={episode_return:.4f} | Steps={episode_steps}")

						episode_return = 0.0
						episode_steps = 0

					progress.set_postfix(update=f"{update_index + 1}/{args.updates}", episodes=episode_count, reward=f"{reward:.4f}", refresh=False)
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

					# An unfinished rollout has no following action to
					# provide its final next-state value.
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
							"config": vars(args),
						},
						checkpoint_path,
					)

					active_collection_seconds = max(collection_seconds - reset_seconds, 1e-8)

					log_data = {
						"train/update": update_index + 1,
						"train/episodes": episode_count,
						"train/update_seconds": update_seconds,
						"timing/collection_seconds": collection_seconds,
						"timing/reset_seconds": reset_seconds,
						"timing/env_step_seconds": env_step_seconds,
						"timing/action_inference_seconds": (action_inference_seconds),
						"rollout/reward_mean": float(np.mean(rollout["rewards"])),
						"rollout/reset_count": reset_count,
						"rollout/simulated_seconds": simulated_seconds,
						"rollout/bootstrap_evaluations": (bootstrap_evaluations),
						"throughput/steps_per_second": (args.rollout_steps / max(collection_seconds, 1e-8)),
						"throughput/sim_seconds_per_wall_second": (simulated_seconds / max(collection_seconds, 1e-8)),
						"throughput/active_sim_seconds_per_wall_second": (simulated_seconds / active_collection_seconds),
						**{f"ppo/{name}": value for name, value in metrics.items() if value is not None},
					}

					if finished_episodes:
						log_data.update(
							{
								"episode/return_mean": float(np.mean([item["return"] for item in finished_episodes])),
								"episode/length_mean": float(np.mean([item["length"] for item in finished_episodes])),
								"episode/success_rate": float(np.mean([item["success"] for item in finished_episodes])),
							}
						)

					if run is not None:
						run.log(log_data, step=total_steps)

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


if __name__ == "__main__":
	args = parse_args()

	if args.mode == "test":
		test()  # Your existing test function.
	else:
		train(args)
