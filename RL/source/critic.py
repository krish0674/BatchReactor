import torch


#remove price and storage 


# TODO: think how to optimize critic NN
class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        sec_per_hour = 60.0 * 60.0

        self.state_input_layer   = torch.nn.Linear(2, 15)
        self.storage_input_layer = torch.nn.Linear(1, 15)
        self.prices_input_layer  = torch.nn.Linear(9, 20)

        self.input_layer  = torch.nn.Linear(50, 50)  # len(observation) = 15
        self.hidden_layer = torch.nn.Linear(50, 50)
        self.output_layer = torch.nn.Linear(50, 1)  # just a score

        self.max_action = torch.tensor([1.2 / sec_per_hour, 700.0 / sec_per_hour])
        self.min_action = torch.tensor([0.8 / sec_per_hour, 0.0 / sec_per_hour])
        self.max_state  = torch.tensor([1.1 * 0.1367, 0.8])  # c_upper, T_upper
        self.min_state  = torch.tensor([0.9 * 0.1367, 0.6])  # c_lower, T_lower

    # use consistent activation functions
    def forward(self, observation) -> torch.Tensor:
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)

        # extract different parts of the current environment state
        state   = self.scale_state(observation[:, :2])
        storage = observation[:, 2].unsqueeze(1)
        prices  = observation[:, 3:]

        state   = torch.tanh(self.state_input_layer(state))
        storage = torch.tanh(self.storage_input_layer(storage))
        prices  = torch.tanh(self.prices_input_layer(prices))

        x = torch.concat((state, storage, prices), dim=1)
        x = torch.tanh(self.input_layer(x))
        x = torch.tanh(self.hidden_layer(x))

        value = self.output_layer(x)  # just a score, can be negative, hence rid activation func. V(s)
        return value

    #def scale_state(self, state):
    #    # Ensure min_state and max_state are broadcastable to state's shape
    #    state = (state - self.min_state.unsqueeze(0)) / (self.max_state - self.min_state).unsqueeze(0)
    #    return state
    def scale_state(self, state):
        min_state = self.min_state.view(1, -1)  # reshape to [1, 2]
        max_state = self.max_state.view(1, -1)  # reshape to [1, 2]
        scaled_state = (state - min_state) / (max_state - min_state)
        return scaled_state
