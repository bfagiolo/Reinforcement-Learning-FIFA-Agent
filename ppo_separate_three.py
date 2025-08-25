# ppo_separate_three_entropy.py
from typing import Optional
import torch
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.utils import explained_variance


class PPOSeparateThreeEntropy(PPO):
    def __init__(self, *args, ent_coef_gauss: float = 0.15, ent_coef_beta: float = 0.02, ent_coef_finesse: float = 0.01, **kwargs):
        # Peek the env that is about to be given to SB3
        env = kwargs.get("env", None)
        if env is None and len(args) >= 2:
            env = args[1]
        if env is not None:
            print("PPO3 DEBUG incoming env.action_space:", getattr(env, "action_space", None),
                getattr(getattr(env, "action_space", None), "shape", None))
            assert getattr(getattr(env, "action_space", None), "shape", None) == (4,), \
                f"PPO3 expected 4-D action space from env, got {getattr(getattr(env, 'action_space', None), 'shape', None)}"


            # NEW: guard against accidentally using old 6-dim obs env/checkpoints
            obs_shape = getattr(getattr(env, "observation_space", None), "shape", None)
            print("PPO3 DEBUG incoming env.observation_space:", getattr(env, "observation_space", None), obs_shape)
            assert obs_shape == (7,), \
                f"PPO3 expects 7-D normalized observations ([ax,ay,gx,gy,d,cos,sin]); got {obs_shape}"




        super().__init__(*args, **kwargs)


        # After SB3 wires things up:
        print("PPO3 DEBUG self.action_space:", self.action_space, getattr(self.action_space, "shape", None))
        self.ent_coef_gauss   = ent_coef_gauss
        self.ent_coef_beta    = ent_coef_beta
        self.ent_coef_finesse = ent_coef_finesse


    def train(self) -> None:
        self.policy.set_training_mode(True)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None


        entropy_g_mean = None
        entropy_b_mean = None
        entropy_f_mean = None
        approx_kl_divs = []
        clip_fractions = []


        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, tuple):
                    actions = tuple(a for a in actions)
                elif self.action_space.__class__.__name__ == "Discrete":
                    actions = actions.long()


                values, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                dist = self.policy.get_distribution(rollout_data.observations)
                if hasattr(dist, "split_entropy"):
                    ent_g, ent_b, ent_f = dist.split_entropy()
                else:
                    # Fallback: treat all entropy as Gaussian head (won't break training)
                    ent = dist.entropy()
                    ent_g, ent_b, ent_f = ent, torch.zeros_like(ent), torch.zeros_like(ent)




                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()


                if clip_range_vf is None:
                    value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values)
                else:
                    values_clipped = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    value_loss_unclipped = (values - rollout_data.returns) ** 2
                    value_loss_clipped = (values_clipped - rollout_data.returns) ** 2
                    value_loss = 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))


                entropy_bonus = (
                    self.ent_coef_gauss   * ent_g.mean()
                  + self.ent_coef_beta    * ent_b.mean()
                  + self.ent_coef_finesse * ent_f.mean()
                )


                loss = policy_loss + self.vf_coef * value_loss - entropy_bonus


                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()


                # logging
                if entropy_g_mean is None:
                    entropy_g_mean, entropy_b_mean, entropy_f_mean = ent_g.mean().detach(), ent_b.mean().detach(), ent_f.mean().detach()
                else:
                    entropy_g_mean += ent_g.mean().detach()
                    entropy_b_mean += ent_b.mean().detach()
                    entropy_f_mean += ent_f.mean().detach()


                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl)
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).cpu().numpy()
                    clip_fractions.append(clip_fraction)


        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        n_batches = self.n_epochs * (self.rollout_buffer.buffer_size // self.batch_size)
        if entropy_g_mean is not None:
            entropy_g_mean /= n_batches
            entropy_b_mean /= n_batches
            entropy_f_mean /= n_batches


        self.logger.record("train/entropy_gauss", entropy_g_mean)
        self.logger.record("train/entropy_beta",  entropy_b_mean)
        self.logger.record("train/entropy_finesse", entropy_f_mean)
        self.logger.record("train/approx_kl", sum(approx_kl_divs) / len(approx_kl_divs))
        self.logger.record("train/clip_fraction", sum(clip_fractions) / len(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)



