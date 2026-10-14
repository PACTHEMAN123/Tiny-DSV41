"""Teacher-forced action probabilities at each actual multi-turn prefix."""
from __future__ import annotations


def trajectory_log_probs(model, trajectory, temperature=1.0):
    from .algorithm import response_log_probs
    from rollout_manager.training import TrainingTrajectory
    if not isinstance(trajectory, TrainingTrajectory):
        # Retain format-1 / original Sample compatibility.
        return response_log_probs(model, trajectory.prompt_ids, trajectory.response_ids, temperature)
    if temperature != trajectory.policy.sampling["temperature"]:
        raise ValueError("learner temperature differs from recorded behavior sampling")
    import torch
    return torch.cat([response_log_probs(model, turn.context_token_ids, turn.action_token_ids, temperature)
                      for turn in trajectory.turns])
