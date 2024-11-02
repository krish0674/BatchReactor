import torch


class Memory:
    def __init__(self, num_environments, short_horizon_length, observation_length, action_length):
        self.observations = torch.zeros((num_environments, short_horizon_length, observation_length))
        self.next_obs     = torch.zeros((num_environments, short_horizon_length, observation_length))
        self.actions      = torch.zeros((num_environments, short_horizon_length, action_length))
        self.rewards      = torch.zeros((num_environments, short_horizon_length, 1))
        self.penalties    = torch.zeros((num_environments, short_horizon_length, 1))

    def store(self, ienv, istep: int, observation, next_obs, action, reward, penalty):
        self.observations[ienv, istep, :] = observation
        self.next_obs[ienv, istep, :]     = next_obs
        self.actions[ienv, istep, :]      = action
        self.rewards[ienv, istep]         = reward
        self.penalties[ienv, istep]       = penalty

    def clear(self):
        self.observations = torch.zeros_like(self.observations)
        self.next_obs     = torch.zeros_like(self.next_obs)
        self.actions      = torch.zeros_like(self.actions)
        self.rewards      = torch.zeros_like(self.rewards)
        self.penalties    = torch.zeros_like(self.penalties)