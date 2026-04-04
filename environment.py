import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, Any, Tuple


class UPIFraudEnv(gym.Env):
    def __init__(self):
        super(UPIFraudEnv, self).__init__()

        # Actions: 0=Approve, 1=Reject, 2=Flag
        self.action_space = spaces.Discrete(3)

        # State: [amount, time, location_risk, device_risk, user_trust, failed_attempts]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([100000.0, 23.0, 1.0, 1.0, 1.0, 10.0]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        # Initial safe transaction
        self.state = np.array([500.0, 12.0, 0.1, 0.2, 0.8, 0.0], dtype=np.float32)
        self.step_count = 0
        self.max_steps = 100
        return self.state.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # Determine if transaction is fraudulent based on state
        fraud_score = self.state[2] + self.state[3] + (1 - self.state[4]) + (self.state[5] * 0.1)
        correct_action = 0 if fraud_score < 1.2 else 2 if fraud_score < 2.0 else 1

        # Reward calculation
        if action == correct_action:
            reward = 1.0
        elif abs(action - correct_action) == 1:
            reward = 0.5
        else:
            reward = -1.0

        # Generate next state
        self.state = np.array([
            random.uniform(100, 50000),
            random.uniform(0, 23),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0.1, 1.0),
            random.randint(0, 3)
        ], dtype=np.float32)

        terminated = self.step_count >= self.max_steps
        truncated = False

        return self.state.copy(), reward, terminated, truncated, {}

    def state(self) -> np.ndarray:
        return self.state.copy()