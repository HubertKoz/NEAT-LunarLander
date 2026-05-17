"""
Three reward shaping strategies for LunarLander-v3.

Each function has the signature:
    func(gym_reward, info, step_count, observation) -> float

observation layout (LunarLander-v3, 8 values):
  [0] x position
  [1] y position
  [2] x velocity
  [3] y velocity
  [4] angle (radians)
  [5] angular velocity
  [6] left leg contact  (0.0 or 1.0)
  [7] right leg contact (0.0 or 1.0)
"""


def default_fitness(gym_reward, info, step_count, observation):
    """Raw gymnasium reward with no modification."""
    return gym_reward


def penalty_time_fitness(gym_reward, info, step_count, observation):
    """Subtract a time penalty to incentivize faster landings.

    Coefficient 0.2 per step nudges the agent to land quickly without
    dominating the shaped landing rewards from the environment.
    """
    return gym_reward - 0.2 * step_count


def penalty_angle_fitness(gym_reward, info, step_count, observation):
    """Penalize tilt when either leg makes ground contact.

    A coefficient of 50.0 on abs(angle) is meaningful because maximum tilt
    at contact is roughly pi/4 (~0.785 rad), giving a max penalty of ~39
    per contact step — comparable to the environment's landing rewards.
    """
    angle = observation[4]
    on_ground = observation[6] or observation[7]
    angle_penalty = 50.0 * abs(angle) if on_ground else 0.0
    return gym_reward - angle_penalty


FITNESS_STRATEGIES = {
    "default": default_fitness,
    "penalty_time": penalty_time_fitness,
    "penalty_angle": penalty_angle_fitness,
}
