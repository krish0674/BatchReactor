import torch
from torchdiffeq import odeint

#change all formulars to ours ,ask sir for sample code

SEC_PER_HR = 60.0 * 60.0


class CSTRParameters:
    # default: 'dopri5', options: 'dopri5', 'rk4' and more, see: https://github.com/rtqichen/torchdiffeq
    def __init__(self, integration_method='rk4'):
        self.V       = 20.0
        self.k       = 300.0 / SEC_PER_HR
        self.N       = 5.0
        self.T_f     = 0.3947
        self.alpha_c = 1.95e-04
        self.T_c     = 0.3816
        self.tau_1   = 4.84
        self.tau_2   = 14.66

        self.integration_method = integration_method
        if self.integration_method == 'rk4':
            self.odeint_options = dict(step_size = SEC_PER_HR)
        elif self.integration_method == 'dopri5':
            self.odeint_options = None
        else:
            raise ValueError("Invalid integration method!")


class CSTR(torch.nn.Module):
    def __init__(self, params: CSTRParameters):
        super().__init__()
        self.p = params

    def cstr_ode(self, t, xu):
        c, T, roh, Fc = xu

        ddt = torch.zeros_like(xu)
        ddt[0] = (1 - c) * roh / self.p.V - c * self.p.k * torch.exp(-self.p.N / T)
        ddt[1] = (self.p.T_f - T) * roh / self.p.V + c * self.p.k * torch.exp(-self.p.N / T) - Fc * self.p.alpha_c * (T - self.p.T_c)

        return ddt

    def forward(self, X0: torch.tensor, U: torch.tensor, delta_t: float):
        X1U = odeint(self.cstr_ode, torch.cat((X0, U)), torch.tensor([0.0, delta_t]), method=self.p.integration_method, options=self.p.odeint_options)
        return X1U[1, :2]  # only return new state
