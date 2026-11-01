# agent.py
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PPOAgent(nn.Module):
	def __init__(self, observation_size, action_size=5, hidden_size=64, device="cuda", learning_rate=3e-4):
		super().__init__()

		self.device = torch.device(device)
		self.observation_size = observation_size

		# Actor: produces the mean of each action's distribution.
		self.actor = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_size))

		# Learnable exploration level for each action.
		self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

		# Critic: produces one state-value estimate.
		self.critic = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

		self.to(self.device)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

	def distribution(self, states):
		mean = self.actor(states)
		std = self.log_std.clamp(-5.0, 2.0).exp()

		return Normal(mean, std)

	def action_log_probability(self, distribution, raw_actions):
		# Correct the probability for the tanh transformation.
		log_jacobian = 2.0 * (math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions))

		return (distribution.log_prob(raw_actions) - log_jacobian).sum(dim=-1)

	@torch.no_grad()
	def act(self, state, deterministic=False):
		state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

		if state.shape != (self.observation_size,):
			raise ValueError("Unexpected observation shape.")

		if not torch.isfinite(state).all():
			raise ValueError("Observation contains invalid values.")

		distribution = self.distribution(state)

		raw_action = distribution.mean if deterministic else distribution.sample()

		# Convert unbounded samples into environment actions in [-1, 1].
		action = torch.tanh(raw_action)

		log_probability = self.action_log_probability(distribution, raw_action)

		value = self.critic(state).squeeze(-1)

		return {"action": action.cpu().numpy(), "raw_action": raw_action.cpu().numpy(), "log_probability": log_probability.item(), "value": value.item()}

	def evaluate_actions(self, states, raw_actions):
		"""Used later to recompute probabilities during PPO updates."""
		states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
		raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32, device=self.device)

		distribution = self.distribution(states)

		log_probabilities = self.action_log_probability(distribution, raw_actions)
		values = self.critic(states).squeeze(-1)

		return log_probabilities, values

	def tensor(self, data, dtype=torch.float32):
		if isinstance(data, (list, tuple)):
			data = np.asarray(data)

		return torch.as_tensor(data, dtype=dtype, device=self.device).detach()

	@torch.no_grad()
	def compute_gae(self, rewards, values, next_values, terminated, truncated, gamma=0.99, gae_lambda=0.95):

		rewards = self.tensor(rewards, torch.float32)
		values = self.tensor(values, torch.float32)
		next_values = self.tensor(next_values, torch.float32)
		terminated = self.tensor(terminated, torch.bool)
		truncated = self.tensor(truncated, torch.bool)

		arrays = (rewards, values, next_values, terminated, truncated)

		if rewards.ndim != 1 or rewards.numel() == 0:
			raise ValueError("Expected a nonempty, one-dimensional rollout.")

		if any(array.shape != rewards.shape for array in arrays):
			raise ValueError("All rollout arrays must have the same shape.")

		if not all(torch.isfinite(array).all().item() for array in (rewards, values, next_values)):
			raise ValueError("Rollout contains non-finite values.")

		# True termination has no future state value.
		bootstrap_values = torch.where(terminated, torch.zeros_like(next_values), next_values)

		deltas = rewards + gamma * bootstrap_values - values

		advantages = torch.zeros_like(rewards)
		gae = torch.zeros((), device=self.device)

		for t in reversed(range(len(rewards))):
			# Neither termination nor truncation may connect to the next episode.
			continue_episode = (~(terminated[t] | truncated[t])).float()

			gae = deltas[t] + gamma * gae_lambda * continue_episode * gae

			advantages[t] = gae

		returns = advantages + values

		return advantages, returns

	@torch.no_grad()
	def estimate_value(self, state):
		state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

		return self.critic(state).squeeze(-1).item()

	def update(
		self,
		states,
		raw_actions,
		old_log_probabilities,
		advantages,
		returns,
		epochs=10,
		batch_size=64,
		clip_ratio=0.2,
		value_coefficient=0.5,
		entropy_coefficient=0.01,
		max_grad_norm=0.5,
		target_kl=0.02,
	):

		states = self.tensor(states)
		raw_actions = self.tensor(raw_actions)
		old_log_probabilities = self.tensor(old_log_probabilities)
		advantages = self.tensor(advantages)
		returns = self.tensor(returns)

		count = states.shape[0]

		if count == 0 or epochs <= 0 or batch_size <= 0:
			raise ValueError("Rollout, epochs and batch size must be positive.")

		if states.shape != (count, self.observation_size):
			raise ValueError("Unexpected state shape.")

		if raw_actions.shape != (count, self.log_std.numel()):
			raise ValueError("Unexpected raw-action shape.")

		for data in (old_log_probabilities, advantages, returns):
			if data.shape != (count,):
				raise ValueError("Expected one scalar per rollout step.")

		for data in (states, raw_actions, old_log_probabilities, advantages, returns):
			if not torch.isfinite(data).all().item():
				raise ValueError("Rollout contains non-finite values.")

		# Normalize actor advantages only; leave critic targets unchanged.
		advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

		metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}

		samples_updated = 0
		updates = 0
		early_stopped = False
		stop_kl = None

		self.train()

		for _ in range(epochs):
			indices = torch.randperm(count, device=self.device)

			for start in range(0, count, batch_size):
				batch = indices[start : start + batch_size]

				distribution = self.distribution(states[batch])

				log_probabilities = self.action_log_probability(distribution, raw_actions[batch])
				predicted_values = self.critic(states[batch]).squeeze(-1)

				log_ratio = log_probabilities - old_log_probabilities[batch]
				ratio = log_ratio.exp()

				if not torch.isfinite(ratio).all().item():
					raise RuntimeError("Non-finite PPO probability ratio.")

				# Estimate how far the new policy moved from the old policy.
				with torch.no_grad():
					approx_kl = ((ratio - 1.0) - log_ratio).mean()

				if target_kl is not None and approx_kl.item() > target_kl:
					early_stopped = True
					stop_kl = approx_kl.item()
					break

				unclipped = ratio * advantages[batch]
				clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[batch]

				# Negative because the optimizer minimizes loss.
				policy_loss = -torch.minimum(unclipped, clipped).mean()

				value_loss = F.mse_loss(predicted_values, returns[batch])

				if entropy_coefficient != 0:
					entropy_samples = distribution.rsample()
					entropy = -self.action_log_probability(distribution, entropy_samples).mean()
				else:
					entropy = torch.zeros((), device=self.device)

				loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy

				if not torch.isfinite(loss).item():
					raise RuntimeError("Non-finite PPO loss.")

				self.optimizer.zero_grad(set_to_none=True)
				loss.backward()

				nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm, error_if_nonfinite=True)

				self.optimizer.step()

				with torch.no_grad():
					clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()

				batch_count = batch.numel()

				for name, value in {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy, "approx_kl": approx_kl, "clip_fraction": clip_fraction}.items():
					metrics[name] += value.detach().item() * batch_count

				samples_updated += batch_count
				updates += 1

			if early_stopped:
				break

		if samples_updated:
			metrics = {name: value / samples_updated for name, value in metrics.items()}

		metrics["updates"] = updates
		metrics["early_stopped"] = early_stopped
		metrics["stop_kl"] = stop_kl

		return metrics
