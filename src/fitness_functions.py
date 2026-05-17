"""
Fitness strategies for evaluating NEAT agents in LunarLander-v3.

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
    """
    Apply a constant time penalty at each simulation step.

    This encourages the agent to finish the episode faster without making
    the penalty grow with the step index.
    """
    return gym_reward - 0.2


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

def landing_quality_fitness(gym_reward, info, step_count, observation):
    """Reward stable, centered and low-velocity landing."""
    x_pos = observation[0]
    x_vel = observation[2]
    y_vel = observation[3]
    angle = observation[4]
    leg_l = observation[6]
    leg_r = observation[7]

    distance_penalty = 10.0 * abs(x_pos)
    velocity_penalty = 5.0 * (abs(x_vel) + abs(y_vel))
    angle_penalty = 20.0 * abs(angle)

    legs_bonus = 20.0 if leg_l and leg_r else 0.0

    return gym_reward - distance_penalty - velocity_penalty - angle_penalty + legs_bonus


def centered_landing_fitness(gym_reward, info, step_count, observation):
    """Reward staying close to the center of the landing area."""
    x_pos = observation[0]
    x_vel = observation[2]
    y_vel = observation[3]
    angle = observation[4]

    return gym_reward - 15.0 * abs(x_pos) - 3.0 * abs(x_vel) - 3.0 * abs(y_vel) - 10.0 * abs(angle)


FITNESS_STRATEGIES = {
    "default": default_fitness,
    "penalty_time": penalty_time_fitness,
    "penalty_angle": penalty_angle_fitness,
    "landing_quality": landing_quality_fitness,
    "centered_landing": centered_landing_fitness,
}
