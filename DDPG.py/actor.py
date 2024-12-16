import torch
import numpy as np
import warnings
import torch as T
import torch.nn as nn
from torch.distributions import Normal
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from timeit import default_timer as timer
from networks import Koopman
#change optlayer based on cvxp documentation
#change from two control to one 
#optlayer constraints almost same ,se paper 


class MPC_Policy(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self):
        super().__init__()

        self.load_CSTR1_koopman_model(
            path='best_val_model.pth')
        self.optlayer = get_CSTR1_optlayer(self.Az)

        self.max_action = torch.tensor([0.125000])
        self.min_action = torch.tensor([0.998277])
        self.max_state  = torch.tensor([34.946917, 30.485979])  # Tr,Tj
        self.min_state  = torch.tensor([90.839534,60.022752])  

    def load_CSTR1_koopman_model(self, path):
        state_dict = torch.load(path)
        model=Koopman()
        model.load_state_dict(state_dict)        
        self.Az = model.Az.weight
        self.Au = model.Au.weight
        self.ZtoX = model.decoder.weight
        self.XtoZ = model.encoder
        return None

    def forward(self, observation) -> torch.Tensor:
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)

        state   = self.scale_state(observation)

        z_init = self.XtoZ(state)

        try:
            action = self.optlayer(
                self.Az, self.Au, self.ZtoX,
                z_init, 
                T.repeat_interleave( repeats=self.optlayer.n_const_cntrl, dim=1))[0]
        except:
            warnings.warn("Fallback to SCS solver in optlayer because specified solver errored.")
            action = self.optlayer(
                self.Az, self.Au, self.ZtoX,
                z_init,
                T.repeat_interleave( repeats=self.optlayer.n_const_cntrl, dim=1),
                solver_args={'solve_method':'SCS'})[0]

        return action

    # any values of state between its extremes is scaled proportionally between 0 and 1.
    def scale_state(self, state):
        # scale state from [min_state, max_state] to [-1, 1]
        state = (state - self.min_state) / (self.max_state - self.min_state)
        state = 2.0 * state - 1.0
        return state

def get_CSTR1_optlayer(Az): #do for our reactor
    starttime = timer()

    # settings
    nominal_production = 0.0
    num_timesteps = int( 7201 )
    n_const_cntrl = int( 4 )
    num_x, num_z, num_u = 2, Az.shape[0], 1
    
    # get cvxpy parameters to set up optimization problem
    Az_cp = cp.Parameter(shape=(num_z,num_z))
    Au_cp = cp.Parameter(shape=(num_z,num_u))
    ZtoX_cp = cp.Parameter(shape=(num_x,num_z))
    z_init_cp = cp.Parameter(shape=(num_z,))
    # storage_init_cp = cp.Parameter(shape=(1,))
    # prices = cp.Parameter(shape=(num_timesteps,))
    
    # variables
    Z = dict()                              # latent space variables
    U = dict()                              # control variables
    storage = dict()                        # quantity of stored product
    for t in range(num_timesteps):
        Z[t] = cp.Variable(shape=num_z)
    for t in range(num_timesteps-1):
        U[t] = cp.Variable(shape=num_u)
    # for t in range(num_timesteps):
    #     storage[t] = cp.Variable(shape=1)
    M_slack = np.array([10_000.0, 10_000.0]) # M_soft_constraints
    X_slack = dict()
    for t in range(num_timesteps):
        X_slack[t] = cp.Variable(shape=num_x, nonneg=True)
    M_storage_slack = 10_000.0               # M_storage
    # storage_slack = dict()
    # for t in range(num_timesteps):
    #     storage_slack[t] = cp.Variable(shape=1, nonneg=True)

    # constraints
    constraints = []
    # initial state
    constraints.append(Z[0] == z_init_cp)
    # constraints.append(storage[0] == storage_init_cp)

    # upper and lower bounds of X
    for t in range(num_timesteps):
        constraints.append( ZtoX_cp @ Z[t] + X_slack[t] >= np.array([-1., -1.]) )
        constraints.append( ZtoX_cp @ Z[t] - X_slack[t] <= np.array([1., 1.]) )

    # upper and lower bounds of U
    for t in range(num_timesteps-1):
        constraints.append( U[t] >= np.array([-1.]) )
        constraints.append( U[t] <= np.array([1.]) )

    # constraints to enforce that U only changes every n timesteps
    for t in range(1,num_timesteps-1):
        if t % n_const_cntrl != 0:
            constraints.append( U[t] == U[t-1])

    # system evolution constraints
    for t in range(1,num_timesteps):
        constraints.append( Az_cp @ Z[t-1] + Au_cp @ U[t-1] == Z[t] )

    # storage evolution constraints
    # for t in range(1,num_timesteps):
    #     constraints.append( storage[t] == storage[t-1] +\
    #                         (U[t-1][0] - nominal_production) / n_const_cntrl * 0.2 )

    # upper and lower bounds of storage
    # for t in range(1,num_timesteps):
    #     constraints.append( storage[t] + storage_slack[t] >= 0.0 )
    #     constraints.append( storage[t] - storage_slack[t] <= 6.0 )
    # constraints.append( storage[num_timesteps-1] + storage_slack[num_timesteps-1] >= 1.0 ) # target_storage == 1.0

    # set objective
    # objective = sum( (U[t][1]+1.0) * prices[t] for t in range(num_timesteps-1) )
    # add quadratic slack penalty
    objective += sum( X_slack[t]**2 @ M_slack for t in range(num_timesteps) )
    # objective += sum( storage_slack[t]**2 * M_storage_slack for t in range(num_timesteps) )
    # minimize objective
    objective = cp.Minimize(objective)

    # formulate the problem
    prob = cp.Problem(objective, constraints)

    # make list of parameters w.r.t. which the solution of the optimization will be differentiated
    parameters = [Az_cp, Au_cp, ZtoX_cp, z_init_cp]

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