import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from timeit import default_timer as timer
from networks import Koopman
from main import get_settings
import warnings

sec_per_hour = 60 * 60 
settings=get_settings()


class MPC_Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.koopman_model = Koopman()

        self.load_CSTR1_koopman_model(path='/kaggle/working/best_val_model.pth')
        self.optlayer = get_CSTR1_optlayer(self.koopman_model.Az.weight)

        sec_per_hour = 60 * 60
        self.max_action = torch.tensor([1.2 / sec_per_hour, 700.0 / sec_per_hour])
        self.min_action = torch.tensor([0.8 / sec_per_hour, 0.0 / sec_per_hour])
        self.max_state = torch.tensor([1.1 * 0.1367, 1.2 * 0.7293])
        self.min_state = torch.tensor([0.9 * 0.1367, 0.8 * 0.7293])  

    def load_CSTR1_koopman_model(self, path):
        model_state_dict = torch.load(path, map_location=torch.device(settings['device']))  #
        self.koopman_model.load_state_dict(model_state_dict)
        print("Koopman model loaded successfully.")

    def forward(self, state):
        # state = self.scale_state(state)
        z_init = self.koopman_model.encoder(state)

        try:
            action = self.optlayer(self.koopman_model.Az.weight, self.koopman_model.Au.weight, self.koopman_model.decoder.weight, z_init)[0]
        except:
            warnings.warn("Fallback to SCS solver in optlayer due to error.")
            action = self.optlayer(self.koopman_model.Az.weight, self.koopman_model.Au.weight, self.koopman_model.decoder.weight, z_init, solver_args={'solve_method':'SCS'})[0]
        # action = self.scale_action(action)
        return action

    def scale_action(self, action):
        action = 0.5 * (action + 1.0)
        action = action * (self.max_action - self.min_action) + self.min_action
        return action

    def scale_state(self, state):
        state = (state - self.min_state) / (self.max_state - self.min_state)
        state = 2.0 * state - 1.0
        return state

def get_CSTR1_optlayer(Az):
    starttime = timer()

    # settings
    nominal_production = 0.0
    num_timesteps = int( 9*4 )
    n_const_cntrl = int( 4 )
    num_x, num_z, num_u = 2, Az.shape[0], 2
    
    # get cvxpy parameters to set up optimization problem
    Az_cp = cp.Parameter(shape=(num_z,num_z))
    Au_cp = cp.Parameter(shape=(num_z,num_u))
    ZtoX_cp = cp.Parameter(shape=(num_x,num_z))
    z_init_cp = cp.Parameter(shape=(num_z,))
    storage_init_cp = cp.Parameter(shape=(1,))
    prices = cp.Parameter(shape=(num_timesteps,))
    
    # variables
    Z = dict()                              # latent space variables
    U = dict()                              # control variables
    storage = dict()                        # quantity of stored product
    for t in range(num_timesteps):
        Z[t] = cp.Variable(shape=num_z)
    for t in range(num_timesteps-1):
        U[t] = cp.Variable(shape=num_u)
    for t in range(num_timesteps):
        storage[t] = cp.Variable(shape=1)
    M_slack = np.array([10_000.0, 10_000.0]) # M_soft_constraints
    X_slack = dict()
    for t in range(num_timesteps):
        X_slack[t] = cp.Variable(shape=num_x, nonneg=True)
    M_storage_slack = 10_000.0               # M_storage
    storage_slack = dict()
    for t in range(num_timesteps):
        storage_slack[t] = cp.Variable(shape=1, nonneg=True)

    # constraints
    constraints = []
    # initial state
    constraints.append(Z[0] == z_init_cp)
    constraints.append(storage[0] == storage_init_cp)

    # upper and lower bounds of X
    for t in range(num_timesteps):
        constraints.append( ZtoX_cp @ Z[t] + X_slack[t] >= np.array([-1., -1.]) )
        constraints.append( ZtoX_cp @ Z[t] - X_slack[t] <= np.array([1., 1.]) )

    # upper and lower bounds of U
    for t in range(num_timesteps-1):
        constraints.append( U[t] >= np.array([-1., -1.]) )
        constraints.append( U[t] <= np.array([1., 1.]) )

    # constraints to enforce that U only changes every n timesteps
    for t in range(1,num_timesteps-1):
        if t % n_const_cntrl != 0:
            constraints.append( U[t] == U[t-1] )

    # system evolution constraints
    for t in range(1,num_timesteps):
        constraints.append( Az_cp @ Z[t-1] + Au_cp @ U[t-1] == Z[t] )

    # storage evolution constraints
    for t in range(1,num_timesteps):
        constraints.append( storage[t] == storage[t-1] +\
                            (U[t-1][0] - nominal_production) / n_const_cntrl * 0.2 )

    # upper and lower bounds of storage
    for t in range(1,num_timesteps):
        constraints.append( storage[t] + storage_slack[t] >= 0.0 )
        constraints.append( storage[t] - storage_slack[t] <= 6.0 )
    constraints.append( storage[num_timesteps-1] + storage_slack[num_timesteps-1] >= 1.0 ) # target_storage == 1.0

    # set objective
    objective = sum( (U[t][1]+1.0) * prices[t] for t in range(num_timesteps-1) )
    # add quadratic slack penalty
    objective += sum( X_slack[t]**2 @ M_slack for t in range(num_timesteps) )
    objective += sum( storage_slack[t]**2 * M_storage_slack for t in range(num_timesteps) )
    # minimize objective
    objective = cp.Minimize(objective)

    # formulate the problem
    prob = cp.Problem(objective, constraints)

    # make list of parameters w.r.t. which the solution of the optimization will be differentiated
    parameters = [Az_cp, Au_cp, ZtoX_cp, z_init_cp, storage_init_cp, prices]

    # create the PyTorch interface
    optlayer = CvxpyLayer(prob,
                          parameters=parameters,
                          variables=[U[0]])

    optlayer.num_timesteps = num_timesteps
    optlayer.num_x = num_x
    optlayer.num_z = num_z
    optlayer.num_u = num_u
    optlayer.n_const_cntrl = n_const_cntrl

    time = round((timer()-starttime), 2)
    print('\nTime taken to set up economic OptLayer for MPC Policy: ' + str(time) + ' seconds\n')
    return optlayer


def test_single_state():
    policy = MPC_Policy()

    test_state = torch.tensor([45.0, 40.0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_state = test_state.to(device)

    predicted_action = policy(test_state)

    print(f"Predicted action (Fc) for state [Tr, Tj] = [45, 40]: {predicted_action.cpu().detach().numpy()}")

if __name__ == "__main__":
    test_single_state()
