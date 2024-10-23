import torch
import numpy as np
from networks import Koopman

# Define dynamic system equations (based on the image provided)
def br(x, u, Ad, Ed, Ap, Ep, deltaHp, UA, Qc, Qs, V, Tc, Cpc, R, alpha, beta, epsilon, theta, m1, cp1, mjCpj, cp2, cp3, cp4, m5, cp5, m6, cp6, M0, I0, Tamb):
    F = u * 16.667  # convert control input Fc to flow rate
    Ii = x[0]
    M = x[1]
    Tr = x[2]
    Tj = x[3]

    # Reaction rates
    Ri = Ad * Ii * (np.exp(-Ed / (R * (Tr + 273.15))))
    Rp = Ap * (Ii ** epsilon) * (M ** theta) * (np.exp(-Ep / (R * (Tr + 273.15))))

    # Heat capacities and heat loss terms
    mrCpr = m1 * cp1 + Ii * cp2 * V + M * cp3 * V + (M0 - M) * cp4 * V + m5 * cp5 + m6 * cp6
    Qpr = alpha * (Tr - Tc) ** beta
    Qr = Rp * V * (-deltaHp)
    Qloss = alpha * (Tr - Tamb) ** beta

    # Dynamic equations
    dy1_dt = -Ri
    dy2_dt = -Rp
    dy3_dt = (Qr - UA * (Tr - Tj) + Qc + Qs - Qloss) / mrCpr
    dy4_dt = (UA * (Tr - Tj) - F * Cpc * (Tj - Tc)) / mjCpj

    xdot = np.zeros(4)
    xdot[0] = dy1_dt  # d[I]/dt
    xdot[1] = dy2_dt  # d[M]/dt
    xdot[2] = dy3_dt  # d[Tr]/dt
    xdot[3] = dy4_dt  # d[Tj]/dt

    return xdot

# Calculate flow control rate (Fc) based on system state and temperature changes
def calculate_Fc(Tr, Tj, dTr, dTj, UA, Tc, Cpc, mjCpj):
    # Use the dynamic heat balance equations for Tj to calculate Fc
    numerator = (UA * (Tr - Tj) - mjCpj * dTj)
    denominator = Cpc * (Tj - Tc)

    if abs(denominator) < 1e-5:  # Avoid division by zero
        return 0.0
    else:
        return numerator / denominator

# Example constants
Ad = 4.4e16
Ed = 140.06e3
Ap = 1.7e11 / 60
Ep = 16.9e3 / 0.239
deltaHp = -82.2e3
UA = 33.3083
Qc = 650
Qs = 12.41e-2
V = 0.5
Tc = 27
Tamb = 27
Cpc = 4.184
R = 8.3145
alpha = 1.212827
beta = 0.000267
epsilon = 0.5
theta = 1.25
m1 = 450
cp1 = 4.184
mjCpj = (18 * 4.184) + (240 * 0.49)
cp2 = 187
cp3 = 110.58
cp4 = 84.95
m5 = 220
cp5 = 0.49
m6 = 7900
cp6 = 0.49
M0 = 0.7034
I0 = 4.5e-3

# Initialize the Koopman Model (this assumes you've already loaded your Koopman model)
class KoopmanModel:
    def __init__(self, model_path,settings):
        self.model = self.load_koopman_model(model_path)
        self.settings=settings
    def load_koopman_model(self, path):
        model_state_dict = torch.load(path, map_location='cpu')
        model = Koopman(self.settings)  # Assuming your Koopman class is available
        model.load_state_dict(model_state_dict)
        model.eval()
        return model

    def predict_next_state(self, current_state, control_input):
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32)
        control_input_tensor = torch.tensor(control_input, dtype=torch.float32)
        next_state = self.model(current_state_tensor, control_input_tensor)
        return next_state.detach().numpy()

# Instantiate the Koopman model
def testit(settings):
    koopman_model = KoopmanModel('/path/to/pretrained/koopman_model.pth',settings=settings)

    # Initial conditions (current state and control input)
    Tr_initial = 45.0
    Tj_initial = 35.0
    Fc_initial = 0.5
    initial_state = [I0, M0, Tr_initial, Tj_initial]

    # Time step size and number of iterations for simulation
    time_step = 0.1
    num_iterations = 10

    # Main loop for running predictions
    current_state = initial_state
    current_Fc = Fc_initial

    for i in range(num_iterations):
        # Get the next state using the Koopman model
        next_state = koopman_model.predict_next_state(current_state[2:], current_Fc)
        Tr_next, Tj_next = next_state[0], next_state[1]

        # Calculate the change in temperatures (dTr, dTj)
        dTr = Tr_next - current_state[2]
        dTj = Tj_next - current_state[3]

        # Use the system dynamics to calculate xdot (state evolution)
        xdot = br(current_state, current_Fc, Ad, Ed, Ap, Ep, deltaHp, UA, Qc, Qs, V, Tc, Cpc, R, alpha, beta, epsilon, theta, m1, cp1, mjCpj, cp2, cp3, cp4, m5, cp5, m6, cp6, M0, I0, Tamb)

        # Update the control input (Fc) based on the temperature change
        new_Fc = calculate_Fc(current_state[2], current_state[3], dTr, dTj, UA, Tc, Cpc, mjCpj)

        # Update the state using Euler's method
        current_state[0] += xdot[0] * time_step  # Update Ii
        current_state[1] += xdot[1] * time_step  # Update M
        current_state[2] += xdot[2] * time_step  # Update Tr
        current_state[3] += xdot[3] * time_step  # Update Tj

        # Update the flow control (Fc)
        current_Fc = new_Fc

        print(f"Iteration {i+1}: Tr = {Tr_next:.2f}, Tj = {Tj_next:.2f}, Fc = {current_Fc:.2f}, dTr = {dTr:.2f}, dTj = {dTj:.2f}")
