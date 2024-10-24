import torch

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        sec_per_hour = 60.0 * 60.0

        # Initialize neural network layers using only state information
        self.state_input_layer = torch.nn.Linear(2, 15)  # Input layer for state, 2 inputs for Tr and Tj

        self.input_layer = torch.nn.Linear(15, 50)  # Consolidated input layer
        self.hidden_layer = torch.nn.Linear(50, 50)  # Hidden layer
        self.output_layer = torch.nn.Linear(50, 1)   # Output layer for scalar value, the value function

        # Action bounds not used directly in the critic, mentioned for completeness
        self.max_action = torch.tensor([1.2 / sec_per_hour, 700.0 / sec_per_hour])
        self.min_action = torch.tensor([0.8 / sec_per_hour, 0.0 / sec_per_hour])

        # State bounds for scaling state inputs
        self.max_state = torch.tensor([1.1 * 0.1367, 0.8])  # c_upper, T_upper
        self.min_state = torch.tensor([0.9 * 0.1367, 0.6])  # c_lower, T_lower

    def forward(self, observation) -> torch.Tensor:
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)

        # Process state part of the observation
        state = self.scale_state(observation[:, :2])
        state = torch.tanh(self.state_input_layer(state))

        x = torch.tanh(self.input_layer(state))
        x = torch.tanh(self.hidden_layer(x))

        value = self.output_layer(x)  # Output the value of the state
        return value

    def scale_state(self, state):
        min_state = self.min_state.view(1, -1)  # reshape to [1, 2]
        max_state = self.max_state.view(1, -1)  # reshape to [1, 2]
        scaled_state = (state - min_state) / (max_state - min_state)
        return scaled_state
