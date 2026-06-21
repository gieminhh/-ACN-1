from typing import Any

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.am.policy import AttentionModelPolicy


class AttentionModel(REINFORCE):
    """Attention Model based on REINFORCE: https://arxiv.org/abs/1803.08475.
    Check :class:`REINFORCE` and :class:`rl4co.models.RL4COLitModule` for more details such as additional parameters  including batch size.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: AttentionModelPolicy | None = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs: dict[str, Any] | None = None,
        baseline_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if policy_kwargs is None:
            policy_kwargs = {}
        if baseline_kwargs is None:
            baseline_kwargs = {}

        if policy is None:
            if "env_name" in policy_kwargs:
                policy = AttentionModelPolicy(**policy_kwargs)
            else:
                policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
