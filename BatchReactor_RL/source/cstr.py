import torch
from torchdiffeq import odeint

SEC_PER_HR = 60.0 * 60.0

class CSTRParameters:
    def __init__(self, integration_method='rk4'):
        self.V = 20.0  # Volume of the reactor
        self.k = 300.0 / SEC_PER_HR  # Rate constant
        self.N = 5.0  # Reaction order
        self.T_f = 0.3947  # Feed temperature
        self.alpha_c = 1.95e-04  # Heat transfer coefficient
        self.T_c = 0.3816  # Coolant temperature
        self.tau_1 = 4.84  # Time constant
        self.tau_2 = 14.66  # Time constant

        self.integration_method = integration_method
        if self.integration_method == 'rk4':
            self.odeint_options = {'step_size': SEC_PER_HR}
        elif self.integration_method == 'dopri5':
            self.odeint_options = None
        else:
            raise ValueError("Invalid integration method!")

class CSTR(torch.nn.Module):
    def __init__(self, params: CSTRParameters):
        super().__init__()
        self.p = params

    def cstr_ode(self, t, xu):
        c, T = xu[:2]  # Only consider state variables: concentration and temperature
        roh, Fc = xu[2:]  # Control inputs: production rate and coolant flow rate

        ddt = torch.zeros_like(xu)
        ddt[0] = (1 - c) * roh / self.p.V - c * self.p.k * torch.exp(-self.p.N / T)  # dC/dt
        ddt[1] = (self.p.T_f - T) * roh / self.p.V + c * self.p.k * torch.exp(-self.p.N / T) - Fc * self.p.alpha_c * (T - self.p.T_c)  # dT/dt

        return ddt[:2]  # Return only derivatives of state variables

    def forward(self, X0: torch.tensor, U: torch.tensor, delta_t: float):
        initial_state = torch.cat((X0, U), dim=0)  # Combine state and control inputs
        times = torch.tensor([0.0, delta_t], dtype=torch.float32)
        X1U = odeint(self.cstr_ode, initial_state, times, method=self.p.integration_method, options=self.p.odeint_options)
        return X1U[1, :2]  # Return only the new state at delta_t
