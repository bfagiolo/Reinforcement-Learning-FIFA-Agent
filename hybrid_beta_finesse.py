import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Bernoulli
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution, SquashedDiagGaussianDistribution


class HybridGaussianBetaBernoulli(Distribution):
    def __init__(self):
        super().__init__()
        self.gauss: Optional[SquashedDiagGaussianDistribution] = None
        self.beta: Optional[Beta] = None
        self.finesse: Optional[Bernoulli] = None
        self._half_log_jac = math.log(0.5)
        self._eps = 1e-6

    def proba_distribution(self, mean_actions_xy, log_std_xy, alpha, beta, finesse_logit):
        self.gauss = SquashedDiagGaussianDistribution(mean_actions_xy.shape[-1])
        self.gauss = self.gauss.proba_distribution(mean_actions=mean_actions_xy, log_std=log_std_xy)
        self.beta = Beta(alpha.clamp_min(self._eps), beta.clamp_min(self._eps))
        self.finesse = Bernoulli(logits=finesse_logit)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        a_xy = actions[..., :2]
        a_z = actions[..., 2:3]
        a_f = actions[..., 3:4]

        logp_g = self.gauss.log_prob(a_xy).sum(dim=-1)

        y = torch.clamp((a_z + 1.0) / 2.0, self._eps, 1.0 - self._eps)
        logp_b = self.beta.log_prob(y).squeeze(-1) + self._half_log_jac

        logp_f = self.finesse.log_prob(a_f).squeeze(-1)

        return logp_g + logp_b + logp_f

    def entropy(self) -> torch.Tensor:
        ent_g = self.gauss.distribution.entropy().sum(dim=-1)
        ent_b = self.beta.entropy().squeeze(-1)
        ent_f = self.finesse.entropy().squeeze(-1)
        return ent_g + ent_b + ent_f

    def split_entropy(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ent_g = self.gauss.distribution.entropy().sum(dim=-1)
        ent_b = self.beta.entropy().squeeze(-1)
        ent_f = self.finesse.entropy().squeeze(-1)
        return ent_g, ent_b, ent_f

    def sample(self) -> torch.Tensor:
        s_xy = self.gauss.sample()
        s_b = self.beta.sample()
        a_z = s_b * 2.0 - 1.0
        s_f = self.finesse.sample()
        return torch.cat([s_xy, a_z, s_f], dim=-1)

    def mode(self) -> torch.Tensor:
        m_xy = self.gauss.distribution.mean

        a = self.beta.concentration1
        b = self.beta.concentration0
        denom = (a + b - 2.0).clamp_min(self._eps)
        y_mode = ((a - 1.0) / denom).clamp(self._eps, 1.0 - self._eps)
        a_z = y_mode * 2.0 - 1.0
        if a_z.ndim == 1:
            a_z = a_z.unsqueeze(-1)

        p = self.finesse.probs
        f_mode = (p >= 0.5).float()
        return torch.cat([m_xy, a_z, f_mode], dim=-1)

    def proba_distribution_net(self, *args, **kwargs):
        return None, None

    def actions_from_params(
        self,
        mean_actions_xy: torch.Tensor,
        log_std_xy: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        finesse_logit: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.proba_distribution(mean_actions_xy, log_std_xy, alpha, beta, finesse_logit)
        actions = self.mode() if deterministic else self.sample()
        return actions, self.log_prob(actions)

    def log_prob_from_params(
        self,
        mean_actions_xy: torch.Tensor,
        log_std_xy: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        finesse_logit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.proba_distribution(mean_actions_xy, log_std_xy, alpha, beta, finesse_logit)
        actions = self.mode()
        return actions, self.log_prob(actions)


class HybridBetaFinessePolicy(ActorCriticPolicy):
    def __init__(self, *args, log_std_init: float = -0.5, **kwargs):
        super().__init__(*args, **kwargs)
        print("POLICY DEBUG v1 — action_space:", self.action_space, "shape:", getattr(self.action_space, "shape", None))
        assert self.action_space.shape[0] == 4

        pi_dim = self.mlp_extractor.latent_dim_pi

        self.mu_xy = nn.Linear(pi_dim, 2)
        nn.init.zeros_(self.mu_xy.bias)
        self.log_std_xy = nn.Parameter(torch.ones(2) * log_std_init)

        self.alpha_head = nn.Linear(pi_dim, 1)
        self.beta_head = nn.Linear(pi_dim, 1)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.beta_head.bias)

        self.finesse_head = nn.Linear(pi_dim, 1)
        nn.init.zeros_(self.finesse_head.bias)

        self._hybrid_dist = HybridGaussianBetaBernoulli()

        for m in [self.mu_xy, self.alpha_head, self.beta_head, self.finesse_head]:
            m.weight.data.mul_(0.01)

        self._build(args[2] if len(args) >= 3 else kwargs.get("lr_schedule", None))

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        mean_xy = self.mu_xy(latent_pi)

        alpha_raw = self.alpha_head(latent_pi)
        beta_raw = self.beta_head(latent_pi)
        alpha = F.softplus(alpha_raw) + 1.2
        beta = F.softplus(beta_raw) + 1.2

        finesse_logit = self.finesse_head(latent_pi)

        log_std_xy = self.log_std_xy.expand_as(mean_xy)

        return self._hybrid_dist.proba_distribution(mean_xy, log_std_xy, alpha, beta, finesse_logit)
